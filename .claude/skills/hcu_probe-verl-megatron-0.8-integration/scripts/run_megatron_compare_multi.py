#!/usr/bin/env python3
"""megatron 训推一致 多样本 compare:对每个 token 完全匹配的样本各跑一次 pt_compare,
严格排除整层(TransformerLayer/TransformerBlock/GPTModel)+ IdentityOp,逐样本 + 汇总 <1% 率。
用法: python3 run_megatron_compare_multi.py <RUN目录> [样本数N=8]"""
import glob, json, re, os, sys, statistics
RUN = sys.argv[1] if len(sys.argv) > 1 else "./verl_run"  # 示例默认;实际用 argv 传 <RUN目录>(含 dump_actor/dump_generate)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def strip(rid): return rid.split("-")[0] if rid else rid


gen = {}
for dl in sorted(glob.glob("%s/dump_generate/*/dispatch_log.jsonl" % RUN)):
    pid = re.search(r"dump_generate/(\d+)/", dl).group(1); k = 0
    for line in open(dl):
        try: r = json.loads(line)
        except: continue
        if r.get("phase") != "prefill": continue
        reqs = r.get("requests", [])
        if len(reqs) == 1:
            cand = sorted(glob.glob("%s/dump_generate/%s/step%d/rank*/dump.json" % (RUN, pid, k)))
            rid = strip(reqs[0].get("request_id"))
            if cand and rid: gen.setdefault(rid, cand[0])
        k += 1
train = {}
for ual in sorted(glob.glob("%s/dump_actor/*/update_actor_log.jsonl" % RUN)):
    for line in open(ual):
        try: r = json.loads(line)
        except: continue
        if r.get("num_requests") != 1: continue
        for rid in r.get("request_ids", []):
            rid = strip(rid); d = "%s/dump_actor/step%s/rank%s/dump.json" % (RUN, r.get("step"), r.get("rank"))
            if os.path.exists(d) and rid: train.setdefault(rid, d)


def tok(d, subs):
    dd = json.load(open(d)).get("data", {})
    for k, v in dd.items():
        if all(s in k for s in subs) and k.endswith("forward.0"):
            a = v.get("input_args", [])
            return a[0].get("shape", [None])[-1] if a and isinstance(a[0], dict) else None
    return None


common = sorted(set(gen) & set(train))
good = [rid for rid in common if tok(train[rid], ["word_embeddings", "VocabParallel"]) == tok(gen[rid], ["embed_tokens", "VocabParallel"])]
print("总配对:", len(common), " token 完全匹配(可用样本):", len(good))
if not good:
    raise SystemExit("无完整匹配配对")

from hcu_probe.pytorch.compare.pt_compare import pt_compare
import pandas as pd


def p2f(x):
    try: return float(str(x).replace("%", ""))
    except: return None


def real_of(xlsx):
    df = pd.read_excel(xlsx)
    df["nerr"] = df["NormRelativeErr"].apply(p2f)
    nm = df["HCU Name"].astype(str)
    isout = nm.str.contains("output", na=False)
    iswrap = nm.str.contains("TransformerLayer|IdentityOp|TransformerBlock|GPTModel", na=False, regex=True)
    return df[isout & ~iswrap & df["nerr"].notna()]


rows = []
for i, rid in enumerate(good[:N]):
    out = "%s/compare_multi/s%d" % (RUN, i)
    os.makedirs(out, exist_ok=True)
    for x in glob.glob(out + "/*.xlsx"): os.remove(x)
    try:
        pt_compare({"npu_path": train[rid], "bench_path": gen[rid]}, out, consistent_check=True, backend="megatron")
        xs = sorted(glob.glob(out + "/*.xlsx"))
        if not xs:
            print(f"sample {i} rid={rid[:12]} 无 xlsx 产出"); continue
        real = real_of(xs[-1])
        n = len(real); lt1 = int((real["nerr"] < 1).sum()); med = real["nerr"].median(); mx = real["nerr"].max()
        tk = tok(train[rid], ["word_embeddings", "VocabParallel"])
        rows.append((rid, n, lt1, med, mx))
        print(f"sample {i} rid={rid[:12]} token={tk} 可比算子={n} <1%={lt1}/{n} 中位={med:.4g}% max={mx:.4g}%")
    except Exception as e:
        print(f"sample {i} rid={rid[:12]} compare 报错: {type(e).__name__}: {e}")

print("\n========== 多样本汇总 ==========")
print("成功样本数:", len(rows))
if rows:
    rates = [lt1 / n for _, n, lt1, _, _ in rows if n]
    meds = [med for _, _, _, med, _ in rows]
    mxs = [mx for _, _, _, _, mx in rows]
    print("每样本 <1% 率:", ", ".join(f"{r:.1%}" for r in rates))
    print("平均 <1% 率:", f"{statistics.mean(rates):.1%}" if rates else "n/a")
    print("每样本中位误差:", ", ".join(f"{m:.4g}%" for m in meds))
    print("每样本最大误差:", ", ".join(f"{m:.4g}%" for m in mxs))
    print("总体: 所有样本可比算子合计", sum(n for _, n, _, _, _ in rows), " 其中 <1% 合计", sum(lt1 for _, _, lt1, _, _ in rows))
