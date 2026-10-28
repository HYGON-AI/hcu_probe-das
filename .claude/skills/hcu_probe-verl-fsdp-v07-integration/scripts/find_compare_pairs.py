"""用 request_id 精确配对两侧同 prompt 的 dump.json。容器内跑。
   推理侧:dispatch_log 的第 k 个 prefill 行(strip 后缀的 request_id)↔ dump_generate/{pid}/step{k}/rank0/dump.json
   训练侧:update_actor_log 行(request_id + step + rank)↔ dump_actor/step{step}/rank{rank}/dump.json
   交集 request_id → (train_dump, gen_dump) 对,写第一对到 compare_pair.txt。"""
import glob
import json
import os
import re

RUN = "/data/hcu_probe_verl/verl_run"


def strip_suffix(rid):
    return rid.split("-")[0] if rid else rid  # das vllm 的 X-<8hex> -> X


# 推理侧:request_id -> gen dump.json(按 prefill 行序对应 dump step)
gen_map = {}
for dl in sorted(glob.glob(f"{RUN}/dump_generate/*/dispatch_log.jsonl")):
    pid = re.search(r"dump_generate/(\d+)/", dl).group(1)
    prefill_k = 0
    for line in open(dl):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("phase") != "prefill":
            continue
        dump = f"{RUN}/dump_generate/{pid}/step{prefill_k}/rank0/dump.json"
        for req in r.get("requests", []):
            rid = strip_suffix(req["request_id"])
            if os.path.exists(dump):
                gen_map.setdefault(rid, dump)
        prefill_k += 1

# 训练侧:request_id -> train dump.json
train_map = {}
for ual in sorted(glob.glob(f"{RUN}/dump_actor/*/update_actor_log.jsonl")):
    for line in open(ual):
        try:
            r = json.loads(line)
        except Exception:
            continue
        step, rank = r.get("step"), r.get("rank")
        for rid in r.get("request_ids", []):
            rid = strip_suffix(rid)
            dump = f"{RUN}/dump_actor/step{step}/rank{rank}/dump.json"
            if os.path.exists(dump):
                train_map.setdefault(rid, dump)

common = sorted(set(train_map) & set(gen_map))
print(f"train reqs={len(train_map)}  gen reqs={len(gen_map)}  common={len(common)}")
for rid in common[:5]:
    print(f"  {rid[:16]}.. train={train_map[rid].split('verl_run/')[-1]}  gen={gen_map[rid].split('verl_run/')[-1]}")

if common:
    rid = common[0]
    with open(f"{RUN}/compare_pair.txt", "w") as f:
        f.write(f"{train_map[rid]}\n{gen_map[rid]}\n{rid}\n")
    print(f"\n第一对 → {RUN}/compare_pair.txt:\n  rid={rid}\n  train={train_map[rid]}\n  gen={gen_map[rid]}")
