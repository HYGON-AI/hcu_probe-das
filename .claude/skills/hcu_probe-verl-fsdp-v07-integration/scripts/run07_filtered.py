# 官方 pt_compare --backend fsdp 复核 verl 0.7 既有 dump(过滤 das vllm 0.18.1 交错模块后)。
# 用法: python3 run07_filtered.py [run目录]  (run目录下须有 dump_actor/ 和 dump_generate/)
# 背景: 0.7 用 das vllm 0.18.1 会多 dump 72 个训练侧无对应的交错模块(Triton.kernel_unified_attention ×24
#       + reshape_and_cache ×24 + rotary_emb ×24),vanilla 官方 pt_compare 的 dual_monotonic_sort 会因两侧
#       module 位置无法同时单调递增而 raise。过滤掉这些 bench-only 交错模块后即可正常跑官方 compare。
import sys, json, glob, re, os, shutil
RUN = sys.argv[1] if len(sys.argv) > 1 else "/data/hcu_probe_verl/dump_frozen"
FILT = RUN.rstrip("/") + "_filtered"
def strip(r): return r.split("-")[0] if r else r
gen = {}
for dl in sorted(glob.glob(RUN + "/dump_generate/*/dispatch_log.jsonl")):
    pid = re.search(r"dump_generate/(\d+)/", dl).group(1); k = 0
    for line in open(dl):
        try: r = json.loads(line)
        except: continue
        if r.get("phase") != "prefill": continue
        reqs = r.get("requests", [])
        if len(reqs) == 1:
            cand = sorted(glob.glob(RUN + "/dump_generate/%s/step%d/rank*/dump.json" % (pid, k)))
            rid = strip(reqs[0].get("request_id"))
            if cand and rid: gen.setdefault(rid, cand[0])
        k += 1
train = {}
for ual in sorted(glob.glob(RUN + "/dump_actor/*/update_actor_log.jsonl")):
    for line in open(ual):
        try: r = json.loads(line)
        except: continue
        if r.get("num_requests") != 1: continue
        for rid in r.get("request_ids", []):
            rid = strip(rid); d = RUN + "/dump_actor/step%s/rank%s/dump.json" % (r.get("step"), r.get("rank"))
            if os.path.exists(d) and rid: train.setdefault(rid, d)
def tok(d, subs):
    dd = json.load(open(d)).get("data", {})
    for k, v in dd.items():
        if all(s in k for s in subs) and k.endswith("forward.0"):
            a = v.get("input_args", [])
            return a[0].get("shape", [None])[-1] if a and isinstance(a[0], dict) else None
    return None
common = sorted(set(gen) & set(train))
good = [rid for rid in common if tok(train[rid], ["embed_tokens"]) == tok(gen[rid], ["embed_tokens"])]
print("总配对:", len(common), " token匹配:", len(good))
rid = good[0]; tp = train[rid]; gp = gen[rid]
print("选 rid=%s token=%s" % (rid, tok(train[rid], ["embed_tokens"])))

# --- 整目录拷贝 gen rank 目录, 过滤 dump.json 里 bench-only 交错模块 ---
gdir = os.path.dirname(gp)
fdir = FILT + "/gen_rank"
if os.path.exists(fdir): shutil.rmtree(fdir)
shutil.copytree(gdir, fdir)
DROP = re.compile(r"Triton\.|rotary_emb|reshape_and_cache|unified_attention")
d = json.load(open(fdir + "/dump.json")); data = d.get("data", {})
kept = {k: v for k, v in data.items() if not DROP.search(k)}
print("bench 模块: %d -> 过滤后 %d (剔除 %d 个 bench-only 交错模块)" % (len(data), len(kept), len(data) - len(kept)))
d["data"] = kept
json.dump(d, open(fdir + "/dump.json", "w"))
fbench = fdir + "/dump.json"

from hcuprobe.pytorch.compare.pt_compare import pt_compare
out = FILT + "/compare_out"; os.makedirs(out, exist_ok=True)
for x in glob.glob(out + "/*.xlsx"): os.remove(x)
pt_compare({"npu_path": tp, "bench_path": fbench}, out, consistent_check=True, backend="fsdp")

import pandas as pd
pd.set_option("display.max_columns", None); pd.set_option("display.width", 340)
df = pd.read_excel(sorted(glob.glob(out + "/*.xlsx"))[-1])
m = df[df["HCU Name"].notna() & df["Bench Name"].notna()].copy()
def p2f(x):
    try: return float(str(x).replace("%", ""))
    except: return None
m["nerr"] = m["NormRelativeErr"].apply(p2f)
nm = m["HCU Name"].astype(str)
is_out = nm.str.contains("output", na=False)
EXCLUDE = "Qwen2DecoderLayer|Qwen2MLP|Qwen2Attention|IdentityOp|act_fn|SiLU"
is_excl = nm.str.contains(EXCLUDE, na=False, regex=True)
real = m[is_out & ~is_excl & m["nerr"].notna()]
print("=== 双边match:", len(m), " Result:", df["Result"].value_counts().to_dict())
print("=== 排除项(整层/IdentityOp/act_fn)output 数:", int((is_out & is_excl).sum()))
print("=== 可比叶子算子 output NormRelativeErr 分布 ===")
print(real["nerr"].describe().to_string())
print("<1%%: %d/%d  <0.5%%: %d  <0.1%%: %d" % ((real["nerr"] < 1).sum(), real["nerr"].notna().sum(), (real["nerr"] < 0.5).sum(), (real["nerr"] < 0.1).sum()))
def _mod(n):
    for k in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","input_layernorm","post_attention","model.norm","embed"]:
        if k.lower() in str(n).lower(): return k
    return "other"
realc = real.copy(); realc["mod"] = realc["HCU Name"].apply(_mod)
print("=== 各叶子类型 中位/最大 误差 ===")
print(realc.groupby("mod")["nerr"].agg(["count","median","max"]).sort_values("max",ascending=False).to_string())
big = realc[realc["nerr"] > 1]
if len(big):
    print("=== 可比叶子里仍 >1%% 的 ===")
    print(big[["HCU Name","nerr"]].sort_values("nerr",ascending=False).head(10).to_string())
