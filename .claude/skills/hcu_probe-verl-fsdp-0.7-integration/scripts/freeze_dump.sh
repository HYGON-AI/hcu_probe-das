#!/bin/bash
# 等 dump.json 落盘后立即停训练 + 把 dump 冻结到持久备份(防训练 step 推进时 hcu_probe 清/覆盖旧 step)
# 后台跑:docker exec -d <container> bash /data/hcu_probe_verl/freeze_dump.sh
# 进度写 freeze.log,完成标志 "DONE"
RUN=/data/hcu_probe_verl/verl_run
LOG=/data/hcu_probe_verl/verl_run.log
BACKUP=/data/hcu_probe_verl/dump_frozen
FLOG=/data/hcu_probe_verl/freeze.log
echo "$(date +%H:%M:%S) freeze_dump 启动,等 dump 落盘..." > $FLOG
for i in $(seq 1 48); do
  ta=$(find "$RUN/dump_actor" -name dump.json 2>/dev/null | wc -l)
  tg=$(find "$RUN/dump_generate" -name dump.json 2>/dev/null | wc -l)
  # update_actor_log 须非空:verify 靠它的 request_ids 配对;append fp 写的,要等 NFS 同步(否则 pkill -9 后 0 字节、配对 0)
  ual=$(cat "$RUN"/dump_actor/*/update_actor_log.jsonl 2>/dev/null | wc -c)
  if [ "${ta:-0}" -gt 0 ] && [ "${tg:-0}" -gt 0 ] && [ "${ual:-0}" -gt 0 ]; then
    echo "$(date +%H:%M:%S) DUMP_READY actor=$ta generate=$tg ual_bytes=$ual (${i}x10s)" >> $FLOG
    break
  fi
  grep -qE "RayTaskError|EXIT=1" "$LOG" 2>/dev/null && echo "$(date +%H:%M:%S) WARN: crash/exit marker seen" >> $FLOG
  sleep 10
done
# 立即停训练防 step 推进清 dump([x] trick 防 pkill 自杀)
for p in 'main_pp[o]' 'EngineCor[e]' 'vLLMHtt[p]' 'ray::[I]'; do pkill -9 -f "$p" 2>/dev/null; done
sleep 3
# 冻结到持久备份
rm -rf "$BACKUP"; mkdir -p "$BACKUP"
cp -r "$RUN/dump_actor" "$BACKUP/" 2>/dev/null
cp -r "$RUN/dump_generate" "$BACKUP/" 2>/dev/null
cp "$LOG" "$BACKUP/run.log" 2>/dev/null
chmod -R 755 "$BACKUP" 2>/dev/null   # 容器 root 建的 0o750,放开让 host 可读
{
  echo "$(date +%H:%M:%S) === 冻结完成 → $BACKUP ==="
  echo "actor_dump.json=$(find "$BACKUP/dump_actor" -name dump.json 2>/dev/null | wc -l)"
  echo "generate_dump.json=$(find "$BACKUP/dump_generate" -name dump.json 2>/dev/null | wc -l)"
  echo "update_actor_log=$(find "$BACKUP/dump_actor" -name update_actor_log.jsonl 2>/dev/null | wc -l)"
  echo "dispatch_log=$(find "$BACKUP/dump_generate" -name dispatch_log.jsonl 2>/dev/null | wc -l)"
  echo "DONE"
} >> $FLOG
