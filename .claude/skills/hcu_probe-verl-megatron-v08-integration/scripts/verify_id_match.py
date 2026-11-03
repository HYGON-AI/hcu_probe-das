"""训推 id 同源核对(megatron 0.8.0):
   推理侧 dispatch_log.jsonl 的 prefill 行 request_id(strip das 加的 -8hex 后缀)
   ↔ 训练侧 update_actor_log.jsonl 的 request_ids,取交集。
   交集非空 = 同一 prompt 的训推 dump 可配对(compare 前置)。
   容器内跑:python3 verify_id_match.py [run_dir]"""
import glob
import json
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else "/path/to/shared/verl_run_v08"


def strip_suffix(rid):
    return rid.split("-")[0] if rid else rid  # das vllm 的 X-<8hex> -> X


# 推理侧:dispatch_log prefill 行的 request_id
gen_ids = set()
gen_prefill_rows = 0
for dl in glob.glob(f"{RUN}/dump_generate/*/dispatch_log.jsonl"):
    for line in open(dl):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("phase") != "prefill":
            continue
        gen_prefill_rows += 1
        for req in r.get("requests", []):
            rid = strip_suffix(req.get("request_id"))
            if rid:
                gen_ids.add(rid)

# 训练侧:update_actor_log 的 request_ids
train_ids = set()
train_rows = 0
for ual in glob.glob(f"{RUN}/dump_actor/*/update_actor_log.jsonl"):
    for line in open(ual):
        try:
            r = json.loads(line)
        except Exception:
            continue
        train_rows += 1
        for rid in r.get("request_ids", []):
            rid = strip_suffix(rid)
            if rid:
                train_ids.add(rid)

common = gen_ids & train_ids
print(f"推理侧 prefill 行: {gen_prefill_rows}  | 唯一 request_id: {len(gen_ids)}")
print(f"训练侧 update_actor 行: {train_rows}  | 唯一 request_id: {len(train_ids)}")
print(f"交集(训推同源 request_id): {len(common)}")
print(f"  推理样例: {list(gen_ids)[:4]}")
print(f"  训练样例: {list(train_ids)[:4]}")
print(f"  交集样例: {list(common)[:4]}")
if common:
    print(f"\n✓ id 同源验证通过:{len(common)} 个 request_id 训推两侧都有,dump 可配对")
elif not train_ids:
    print("\n⏳ 训练侧 update_actor_log 暂无 request_id(compute_log_prob/update 阶段尚未写),稍后重跑")
elif not gen_ids:
    print("\n⏳ 推理侧暂无 prefill request_id(rollout 尚未产出),稍后重跑")
else:
    print("\n✗ 交集为空:两侧 id 不同源,需排查 request_id 注入链路")
