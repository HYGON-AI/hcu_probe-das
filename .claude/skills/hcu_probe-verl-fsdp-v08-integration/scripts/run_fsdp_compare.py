import glob, json, re, os, sys

def _read_result(out_dir):
    # 新版 hcuprobe(hcu_probe-das)默认输出 compare_result_*.csv、列名为 "NPU Name/NPU l2norm"(上游命名);
    # 旧版 dcu_probe 输出 *.xlsx、列名 "HCU Name"。两者兼容:统一读最新结果并把 "NPU " 列名前缀归一为 "HCU "。
    import glob as _g, os as _o, pandas as _pd
    fs = sorted(_g.glob(_o.path.join(out_dir, "compare_result_*.csv")) + _g.glob(_o.path.join(out_dir, "*.xlsx")), key=_o.path.getmtime)
    if not fs:
        raise SystemExit("compare 无结果文件: " + out_dir)
    df = _pd.read_csv(fs[-1]) if fs[-1].endswith(".csv") else _pd.read_excel(fs[-1])
    return df.rename(columns=lambda c: "HCU " + c[4:] if c.startswith("NPU ") else c)

# 用法: python3 run_fsdp_compare.py [run目录]。run目录 = 你的产物目录(其下有 dump_actor/ 和 dump_generate/),
# 须 == config_*.json 里 dump_path 的父目录。容器内跑(dump.json 是容器 root 0750,宿主读不到)。
#
# FSDP 后端 vs Megatron 后端的差异(本脚本相对 run_megatron_compare.py):
#   1. pt_compare(..., backend="fsdp"):pt_compare 自动用 FSDP_MODULE_MAP(q/k/v_proj->qkv_proj 等),
#      而非 MEGATRON_MODULE_MAP。
#   2. FSDP 训练侧 module 是 HF 风格命名(embed_tokens / q_proj / norm),不是 mcore 风格
#      (word_embeddings / linear_qkv / final_layernorm)。故 token 匹配、首尾检查的 key 全部改 HF 风格。
#      训练侧与推理侧两边都用 ["embed_tokens"] 做 token 匹配(FSDP 训练侧 == HF == 推理侧命名一致)。
# ⚠️ 下面的 HF module key(embed_tokens / norm)是基于 HF 风格的最佳推断,用宽松子串匹配。
#    实跑后请按实际 dump.json 里的 key 微调(很可能带前缀,如 model.embed_tokens / model.norm)。
RUN = sys.argv[1] if len(sys.argv) > 1 else ""
if not RUN or not os.path.isdir(RUN + "/dump_actor"):
    raise SystemExit("用法: python3 run_fsdp_compare.py <RUN目录>（== config dump_path 父目录,其下有 dump_actor/+dump_generate/；容器内跑,需 PYTHONPATH 含 Megatron-LM + pip install openpyxl）")
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
# token 匹配挑同 prompt 完整 prefill 样本:FSDP 训练侧 == HF == 推理侧,两边都用 ["embed_tokens"]。
# ⚠️ 若实际 dump key 形如 model.embed_tokens,["embed_tokens"] 子串仍可命中,无需改;若命名变化再调。
good = [rid for rid in common if tok(train[rid], ["embed_tokens"]) == tok(gen[rid], ["embed_tokens"])]
print("总配对:", len(common), " token数完全匹配(真·同prompt完整prefill):", len(good))
if not good:
    raise SystemExit("无完整匹配配对")
rid = good[0]
print("选 rid=%s  token=%s" % (rid, tok(train[rid], ["embed_tokens"])))

from hcuprobe.pytorch.compare.pt_compare import pt_compare
out = "%s/compare_out" % RUN
os.makedirs(out, exist_ok=True)
for x in glob.glob(out + "/*.xlsx") + glob.glob(out + "/compare_result_*.csv"): os.remove(x)
pt_compare({"npu_path": train[rid], "bench_path": gen[rid]}, out, consistent_check=True, backend="fsdp")

import pandas as pd
pd.set_option("display.max_columns", None); pd.set_option("display.width", 340)
df = _read_result(out)
m = df[df["HCU Name"].notna() & df["Bench Name"].notna()].copy()
def p2f(x):
    try: return float(str(x).replace("%", ""))
    except: return None
m["nerr"] = m["NormRelativeErr"].apply(p2f)
nm = m["HCU Name"].astype(str)
is_out = nm.str.contains("output", na=False)
# 排除项(都不是"训推可比的叶子算子",列出来保持透明):
#   - 整层 wrapper(Qwen2DecoderLayer/MLP/Attention):output=残差和,两侧结构/捕获点不同,relative err 失真(megatron 口径也不比 TransformerLayer)
#   - IdentityOp:融合占位算子
#   - act_fn(SiLU/SiluAndMul):融合算子不可比 —— HF act_fn=silu(gate) 单独,vllm act_fn=SiluAndMul=silu(gate)·up,算的是不同张量。
#     真正等价的比较点是 down_proj(两侧都吃 silu(gate)·up)。
EXCLUDE = "Qwen2DecoderLayer|Qwen2MLP|Qwen2Attention|IdentityOp|act_fn|SiLU"
is_excl = nm.str.contains(EXCLUDE, na=False, regex=True)
real = m[is_out & ~is_excl & m["nerr"].notna()]
print("=== 双边match:", len(m), " Result:", df["Result"].value_counts().to_dict())
print("=== 排除项(整层/IdentityOp/融合act_fn)output 数:", (is_out & is_excl).sum(),
      " (其中 act_fn:", (is_out & nm.str.contains("act_fn|SiLU", na=False, regex=True)).sum(), ") —— 这些是结构性不可比,不计入一致率")
print("=== 可比叶子算子 output NormRelativeErr 分布(已排除整层+IdentityOp+act_fn融合) ===")
print(real["nerr"].describe().to_string())
print("<1%%: %d/%d  <0.5%%: %d  <0.1%%: %d" % ((real["nerr"] < 1).sum(), real["nerr"].notna().sum(), (real["nerr"] < 0.5).sum(), (real["nerr"] < 0.1).sum()))
import re as _re
def _mod(n):
    for k in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","input_layernorm","post_attention","model.norm","embed"]:
        if k.lower() in str(n).lower(): return k
    return "other"
realc = real.copy(); realc["mod"] = realc["HCU Name"].apply(_mod)
print("=== 各叶子类型 中位/最大 误差 ===")
print(realc.groupby("mod")["nerr"].agg(["count","median","max"]).sort_values("max",ascending=False).to_string())
big = realc[realc["nerr"] > 1]
if len(big):
    print("=== 可比叶子里仍 >1% 的(尾巴,需追因)===")
    print(big[["HCU Name","nerr"]].sort_values("nerr",ascending=False).head(10).to_string())
# 首尾检查:FSDP HF 风格 = embed_tokens(Embedding) + model.norm(RMSNorm)。
# ⚠️ HF Qwen 首层是 model.embed_tokens(torch.nn.Embedding),末层 norm 是 model.norm(Qwen2RMSNorm/RMSNorm);
#    实跑后按实际 dump.json 的 module 类名/前缀微调(可能是 embed_tokens.Embedding / norm.Qwen2RMSNorm 等)。
print("=== 首尾 embed/norm ===")
for kw in ["embed_tokens", "norm.RMSNorm"]:
    sub = m[m["HCU Name"].astype(str).str.contains(kw.replace(".", "\\."), na=False, regex=True)]
    if len(sub): print(sub[["HCU l2norm", "Bench l2norm", "NormRelativeErr"]].to_string())
