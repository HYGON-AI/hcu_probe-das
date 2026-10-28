"""读 verl tensorboard event,打印关键训练指标趋势(L4 收敛验证用)。
用法:python3 read_tb.py [event_file 或 glob,默认取 tensorboard 目录最新]"""
import sys, glob
from tensorboard.backend.event_processing import event_accumulator as EA

pat = sys.argv[1] if len(sys.argv) > 1 else "/data/hcu_probe_verl/verl_run/tensorboard/*"
files = sorted(glob.glob(pat))
if not files:
    print("NO event file matching:", pat); sys.exit(1)
f = files[-1]  # 最新(按文件名排序,时间戳递增)
print("event:", f)
ea = EA.EventAccumulator(f, size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]
# 关心的收敛指标(key 名可能因 verl 版本略异,下面再列全量兜底)
want = ["actor/pg_loss", "critic/rewards/mean", "critic/score/mean", "actor/grad_norm",
        "actor/kl_loss", "actor/entropy", "response_length/mean", "actor/ppo_kl"]
for k in want:
    if k in tags:
        s = [(v.step, round(v.value, 4)) for v in ea.Scalars(k)]
        print(f"{k}: {s}")
print("--- 全量 scalar tags(确认 key 名)---")
print(sorted(tags)[:50])
