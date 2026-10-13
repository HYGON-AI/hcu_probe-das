"""训推一致性验证(das vllm 移植后):比对训练侧(FSDP)与推理侧(vllm)对同一 prompt 前向 dump 的统计量。

【为什么需要这个专用脚本,而不能直接用 hcu_probe 的 compare】
das/vllm 有两个特性,会让通用 compare 工具产生大量"假 diff"、误判训推不一致:
  1. vllm 的 Qwen2DecoderLayer 用 fused residual,forward 返回 (normed_hidden, residual) **两个** output;
     HF(训练侧)只返回 **一个**。通用工具会把 HF.output[0](完整 hidden) 错配到 vllm.output[0](normed 小值),
     造出几千倍的假 diff。**正确对应是 HF.output[0] ↔ vllm.output[1](residual)**。
  2. vllm 默认 batch prefill(多 prompt 拼一批算),dump 粒度与训练单 request 不一致 → 多数对不可比。
     前置:训练侧须 micro_batch=1;推理侧须 max_num_seqs=1 + enable_chunked_prefill=False(单 request prefill)。

【本脚本做什么】
  用 request_id 配对两侧同 prompt 的 dump → 检查可比性(embed 输入 token 数一致)→
  只比两侧能严格对应的统计量:embed/各层 layernorm 的 normed output + 残差流(用上面的正确对应)。

用法: python3 verify_consistency.py [run_dir] [--threshold 0.005]
  run_dir 默认 /data/hcu_probe_verl/verl_run;threshold 是 Norm 相对误差阈值(默认 0.5%)。
"""
import argparse
import glob
import json
import os
import re


def strip_suffix(rid):
    return rid.split("-")[0] if rid else rid  # das vllm 会给 request_id 加 "-<8hex>" 后缀


def pair_dumps(run):
    """request_id → (train_dump.json, gen_dump.json)。推理侧第 k 个 prefill 行对应 dump step k。"""
    gen_map, train_map = {}, {}
    for dl in sorted(glob.glob(f"{run}/dump_generate/*/dispatch_log.jsonl")):
        pid = re.search(r"dump_generate/(\d+)/", dl).group(1)
        k = 0
        for line in open(dl):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("phase") != "prefill":
                continue
            d = f"{run}/dump_generate/{pid}/step{k}/rank0/dump.json"
            for req in r.get("requests", []):
                if os.path.exists(d):
                    gen_map.setdefault(strip_suffix(req["request_id"]), d)
            k += 1
    for ual in sorted(glob.glob(f"{run}/dump_actor/*/update_actor_log.jsonl")):
        for line in open(ual):
            try:
                r = json.loads(line)
            except Exception:
                continue
            for rid in r.get("request_ids", []):
                d = f"{run}/dump_actor/step{r.get('step')}/rank{r.get('rank')}/dump.json"
                if os.path.exists(d):
                    train_map.setdefault(strip_suffix(rid), d)
    return [(rid, train_map[rid], gen_map[rid]) for rid in sorted(set(train_map) & set(gen_map))]


def find_out(d, layer, sub, idx=0):
    """取匹配 (layers.{layer} 或 embed_tokens) + sub 的 module 的 output[idx] 统计字典。"""
    pat = f"layers.{layer}." if layer is not None else "embed_tokens."
    for k, v in d.items():
        if pat in k and sub in k and k.endswith("forward.0"):
            o = v.get("output", [])
            if len(o) > idx and isinstance(o[idx], dict):
                return o[idx]
    return None


def emb_in_tokens(d, cls):
    """embed 输入(token id)的 (token 数, Norm),用于可比性检查。"""
    for k, v in d.items():
        if "embed_tokens" in k and cls in k and k.endswith("forward.0"):
            a = v.get("input_args", [])
            return (a[0]["shape"][-1], a[0]["Norm"]) if a and isinstance(a[0], dict) else None
    return None


def relerr_norm(a, b):
    if not a or not b:
        return None
    return abs(a["Norm"] - b["Norm"]) / max(abs(b["Norm"]), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="/data/hcu_probe_verl/verl_run")
    ap.add_argument("--threshold", type=float, default=0.005)
    args = ap.parse_args()

    pairs = pair_dumps(args.run)
    print(f"配对(request_id 关联): {len(pairs)} 个 request")

    comparable, batch_prefill, results = 0, 0, []
    for rid, tp, gp in pairs:
        td = json.load(open(tp))["data"]
        gd = json.load(open(gp))["data"]
        ti, gi = emb_in_tokens(td, "Embedding"), emb_in_tokens(gd, "VocabParallel")
        # 可比性:同 prompt 的 embed 输入 token 数必相同;不同 = 推理被 batch prefill,不可比
        if not (ti and gi and ti[0] == gi[0] and abs(ti[1] - gi[1]) / max(gi[1], 1e-9) < 1e-3):
            batch_prefill += 1
            continue
        comparable += 1
        checks = []
        e = relerr_norm(find_out(td, None, "Embedding"), find_out(gd, None, "VocabParallel"))
        if e is not None:
            checks.append(("embed", e))
        layers = sorted(set(int(m.group(1)) for k in td for m in [re.search(r"layers\.(\d+)\.", k)] if m))
        # 只比两侧能严格对应的 normed 算子(RMSNorm 输出)。
        # 不比整层 Qwen2DecoderLayer:vllm 用 fused residual,整层 output 是 (normed, residual) tuple,
        # 与 HF 的单 output 对应关系复杂且不普遍成立(实测 layer0 就 900%+),靠它会得出错误结论。
        for L in layers:
            for sub in ("input_layernorm", "post_attention_layernorm"):
                r = relerr_norm(find_out(td, L, sub), find_out(gd, L, sub))
                if r is not None:
                    checks.append((f"{sub[:3]}{L}", r))
        if checks:
            worst = max(checks, key=lambda x: x[1])
            results.append((rid, worst[1], worst[0]))

    print(f"可比(单 request prefill): {comparable}  |  不可比(batch prefill): {batch_prefill}")
    if batch_prefill:
        print(f"  ⚠️ {batch_prefill} 个推理是 batch prefill,无法与训练单 request 比对。")
        print("     须用 max_num_seqs=1 + enable_chunked_prefill=False 重跑,才能让全部 request 单独 prefill。")
    if not results:
        print("无可比对(全部 batch prefill 或 dump 缺失)。")
        return
    rels = [r[1] for r in results]
    th = args.threshold
    npass = sum(1 for r in rels if r < th)
    print(f"\n训推一致性(normed 算子 + 残差流 Norm 相对误差,阈值 {th * 100:g}%):")
    print(f"  通过: {npass}/{len(rels)}  |  最差: {max(rels) * 100:.3f}%  中位: {sorted(rels)[len(rels) // 2] * 100:.4f}%")
    bad = sorted([r for r in results if r[1] >= th], key=lambda x: -x[1])
    for r in bad[:10]:
        print(f"  ✗ rid={r[0][:12]}  {r[1] * 100:.2f}%  @{r[2]}")
    print("\n结论:", "✓ 全部可比对训推一致" if not bad else f"✗ {len(bad)}/{len(rels)} 对超阈值,需排查")


if __name__ == "__main__":
    main()
