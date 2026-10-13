#!/bin/bash
# 读 baseline live run 的 tensorboard,打印健康关键指标趋势。容器内跑。
set +e
TB="${1:-<RUN>/tensorboard}"
python3 - "$TB" <<'PY'
import sys, glob, math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
d = sys.argv[1]
ea = EventAccumulator(d, size_guidance={'scalars': 0})
ea.Reload()
tags = ea.Tags().get('scalars', [])
want = ['actor/grad_norm','actor/ppo_kl','actor/pg_loss','critic/rewards/mean','critic/score/mean',
        'response_length/mean','prompt_length/mean']
# 名字可能有前缀差异,做模糊匹配
def find(t):
    for x in tags:
        if x == t or x.endswith(t) or t.split('/')[-1] in x and t.split('/')[0] in x:
            return x
    return None
print("总步数(按 critic/rewards/mean):", end=" ")
r = find('critic/rewards/mean')
n = len(ea.Scalars(r)) if r else 0
print(n)
print(f"{'指标':26s} n  首值      末值      min       max")
for t in want:
    x = find(t)
    if not x:
        print(f"{t:26s} (未找到)"); continue
    vals = [s.value for s in ea.Scalars(x)]
    if not vals:
        print(f"{t:26s} (空)"); continue
    nan = sum(1 for v in vals if math.isnan(v) or math.isinf(v))
    print(f"{t:26s} {len(vals):2d}  {vals[0]:<9.4g} {vals[-1]:<9.4g} {min(vals):<9.4g} {max(vals):<9.4g}  NaN/Inf={nan}")
print()
print("可用 scalar tags(前25):")
for x in tags[:25]: print("  ", x)
PY
