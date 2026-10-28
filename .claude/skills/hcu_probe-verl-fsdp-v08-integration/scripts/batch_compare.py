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

import pandas as pd
# 用法: python3 batch_compare.py [run目录]。run目录 = 产物目录(其下有 dump_actor/ dump_generate/),== config dump_path 父目录。容器内跑。
#
# FSDP 后端版(相对 megatron 版的差异):
#   1. pt_compare(..., backend="fsdp"):自动用 FSDP_MODULE_MAP。
#   2. FSDP 训练侧 HF 风格命名,token 匹配两边都用 ["embed_tokens"](训练侧 == 推理侧)。
# ⚠️ HF module key(embed_tokens)为最佳推断,用宽松子串匹配;实跑后按实际 dump.json key 微调
#    (可能带前缀,如 model.embed_tokens / model.norm)。
RUN = sys.argv[1] if len(sys.argv) > 1 else "/path/to/shared/verl_run_v08"
def strip(rid): return rid.split("-")[0] if rid else rid
gen = {}
for dl in sorted(glob.glob("%s/dump_generate/*/dispatch_log.jsonl" % RUN)):
    pid = re.search(r"dump_generate/(\d+)/", dl).group(1); k = 0
    for line in open(dl):
        try: r = json.loads(line)
        except: continue
        if r.get("phase") != "prefill": continue
        if len(r.get("requests", [])) == 1:
            cand = sorted(glob.glob("%s/dump_generate/%s/step%d/rank*/dump.json" % (RUN, pid, k)))
            rid = strip(r["requests"][0].get("request_id"))
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
# token 匹配两边都用 ["embed_tokens"](FSDP 训练侧 == HF == 推理侧)。⚠️ 实跑后按实际 dump key 微调。
good = [rid for rid in common if tok(train[rid], ["embed_tokens"]) == tok(gen[rid], ["embed_tokens"])]
print("token完全匹配配对:", len(good))

from hcuprobe.integrations.verl import consistent_compare  # 内含 statistics squeeze 容差 + 剔除 Triton 条目,不改上游 core
def p2f(x):
    try: return float(str(x).replace("%", ""))
    except: return None
allres = []
for i, rid in enumerate(good[:8]):
    out = "%s/cmp_%d" % (RUN, i)
    os.makedirs(out, exist_ok=True)
    for x in glob.glob(out + "/*.xlsx") + glob.glob(out + "/compare_result_*.csv"): os.remove(x)
    try:
        consistent_compare(train[rid], gen[rid], out, backend="fsdp")
    except Exception as e:
        print("rid", rid[:8], "compare err", e); continue
    df = _read_result(out)
    m = df[df["HCU Name"].notna() & df["Bench Name"].notna()].copy()
    m["nerr"] = m["NormRelativeErr"].apply(p2f)
    # 真算子:output 且非整层 TransformerLayer/DecoderLayer、非 IdentityOp、非整块 attention。
    # ⚠️ FSDP HF 风格的整层/整块名(Qwen2DecoderLayer / Qwen2Attention 等)与 megatron(TransformerLayer/
    #    SelfAttention)不同;此处保留 megatron 关键字并补 HF 关键字,实跑后按实际 dump 类名增删。
    real = m[m["HCU Name"].astype(str).str.contains("output", na=False)
             & ~m["HCU Name"].astype(str).str.contains("TransformerLayer", na=False)
             & ~m["HCU Name"].astype(str).str.contains("DecoderLayer", na=False)
             & ~m["HCU Name"].astype(str).str.contains("IdentityOp", na=False)
             & ~m["HCU Name"].astype(str).str.contains("SelfAttention", na=False)
             & ~m["HCU Name"].astype(str).str.contains("Attention.forward", na=False)]
    n = real["nerr"].notna().sum()
    p1 = (real["nerr"] < 1).sum()
    med = real["nerr"].median()
    tk = tok(train[rid], ["embed_tokens"])
    allres.append((rid[:10], tk, n, p1, med))
    print("rid=%s tok=%s 真算子=%d <1%%=%d 中位=%.3f%%" % (rid[:10], tk, n, p1, med))
print("=== 汇总 %d 个配对 ===" % len(allres))
tot_n = sum(r[2] for r in allres); tot_p = sum(r[3] for r in allres)
print("真算子总数=%d  <1%%=%d (%.1f%%)" % (tot_n, tot_p, 100.0*tot_p/max(tot_n,1)))
