import glob, json, re, os, sys
# 用法: python3 run_megatron_compare.py [run目录]。run目录 = 你的产物目录(其下有 dump_actor/ 和 dump_generate/),
# 须 == config_*.json 里 dump_path 的父目录。容器内跑(dump.json 是容器 root 0750,宿主读不到)。
RUN = sys.argv[1] if len(sys.argv) > 1 else "/path/to/shared/verl_run_v08"
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
print("总配对:", len(common), " token数完全匹配(真·同prompt完整prefill):", len(good))
if not good:
    raise SystemExit("无完整匹配配对")
rid = good[0]
print("选 rid=%s  token=%s" % (rid, tok(train[rid], ["word_embeddings", "VocabParallel"])))

from hcu_probe.pytorch.compare.pt_compare import pt_compare
out = "%s/compare_out" % RUN
os.makedirs(out, exist_ok=True)
for x in glob.glob(out + "/*.xlsx"): os.remove(x)
pt_compare({"npu_path": train[rid], "bench_path": gen[rid]}, out, consistent_check=True, backend="megatron")

import pandas as pd
pd.set_option("display.max_columns", None); pd.set_option("display.width", 340)
df = pd.read_excel(sorted(glob.glob(out + "/*.xlsx"))[-1])
m = df[df["HCU Name"].notna() & df["Bench Name"].notna()].copy()
def p2f(x):
    try: return float(str(x).replace("%", ""))
    except: return None
m["nerr"] = m["NormRelativeErr"].apply(p2f)
real = m[m["HCU Name"].astype(str).str.contains("output", na=False) & ~m["HCU Name"].astype(str).str.contains("IdentityOp", na=False)]
print("=== 双边match:", len(m), " Result:", df["Result"].value_counts().to_dict())
print("=== 真算子 output NormRelativeErr 分布(排除IdentityOp融合) ===")
print(real["nerr"].describe().to_string())
print("<1%%: %d/%d  <0.5%%: %d" % ((real["nerr"] < 1).sum(), real["nerr"].notna().sum(), (real["nerr"] < 0.5).sum()))
print("=== 首尾 embed/final_norm ===")
for kw in ["word_embeddings.VocabParallelEmbedding.forward.0.output.0", "final_layernorm.RMSNorm.forward.0.output.0"]:
    sub = m[m["HCU Name"].astype(str).str.contains(kw.replace(".", "\\."), na=False, regex=True)]
    if len(sub): print(sub[["HCU l2norm", "Bench l2norm", "NormRelativeErr"]].to_string())
