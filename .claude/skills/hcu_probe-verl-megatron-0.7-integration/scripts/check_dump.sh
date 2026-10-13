#!/bin/bash
# 验证 ③:dump 模块数 + NaN + update_actor_log 的 request_ids 非空。容器内跑。
set +e
D="${1:-<RUN>/dump_actor}"
echo "=== rank0 dump.json 模块数 + NaN ==="
python3 - "$D/step0/rank0/dump.json" <<'PY'
import json, sys, math
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("READ_FAIL:", e); sys.exit(0)
data = d.get("data", {})
print("模块数 =", len(data))
nan=inf=tot=0
def w(o):
    global nan,inf,tot
    if isinstance(o,dict):
        for v in o.values(): w(v)
    elif isinstance(o,list):
        for v in o: w(v)
    elif isinstance(o,float):
        tot+=1
        if math.isnan(o): nan+=1
        if math.isinf(o): inf+=1
    elif isinstance(o,int): tot+=1
for v in data.values(): w(v)
print(f"数值字段={tot} NaN={nan} Inf={inf}")
PY
echo
echo "=== update_actor_log 各 pid 的 request_ids ==="
for f in "$D"/*/update_actor_log.jsonl; do
  [ -f "$f" ] || continue
  n=$(wc -l < "$f")
  echo "--- $f  行数=$n ---"
  head -3 "$f"
done
echo
echo "=== 汇总:所有 update_actor_log 里非空 request_ids 计数 ==="
cat "$D"/*/update_actor_log.jsonl 2>/dev/null | python3 -c "
import sys, json
tot=0; nonempty=0; ids=set()
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    try: r=json.loads(line)
    except: continue
    tot+=1
    rr=r.get('request_ids',[])
    if rr: nonempty+=1; ids.update(rr)
print(f'记录总数={tot} 非空request_ids记录={nonempty} 唯一request_id数={len(ids)}')
for i in list(ids)[:5]: print('  sample:', i)
"
