#!/bin/bash
# 容器内 verl dump 实验管理(scp 到 /data/hcu_probe_verl/,docker exec bash manage.sh <action>)
# 封装杀进程/清盘/启动,避开三层 ssh 引号地狱 + pkill -f 自杀(用 [x] trick 让 pattern 不匹配自身)
set +e
A=/data/hcu_probe_verl/hcu_verl/verl/verl
RUN=/data/hcu_probe_verl

# 从 L0 statistics base config 派生 dump 变体(改 level/task),避免一堆近重复 config 文件
# 用法:gen_cfg <suffix> <level> <task> → 生成 config_{actor,generate}_<suffix>.json
gen_cfg() {
  local suffix="$1" level="$2" task="$3"
  for side in actor generate; do
    python3 - "$RUN/config_${side}.json" "$RUN/config_${side}_${suffix}.json" "$level" "$task" <<'PYEOF'
import json, sys
src, dst, level, task = sys.argv[1:5]
cfg = json.load(open(src))
cfg["level"], cfg["task"] = level, task
json.dump(cfg, open(dst, "w"), indent=4, ensure_ascii=False)
print(f"[gen_cfg] {dst} level={level} task={task}")
PYEOF
  done
}

case "$1" in
  status)
    echo "PROC_main_ppo=$(ps -ef | grep 'main_pp[o]' | grep -v grep | wc -l)"
    echo "PROC_engine=$(ps -ef | grep 'EngineCor[e]' | grep -v grep | wc -l)"
    echo "PYC_actor=$(ls $A/workers/actor/__pycache__/ 2>/dev/null | wc -l)"
    echo "dp_actor_request_id_hits=$(grep -c request_id $A/workers/actor/dp_actor.py)"
    echo "reqid_inject=$(grep -cF 'output.extra_fields["request_id"] = vllm_request_id' $A/experimental/agent_loop/agent_loop.py)"
    echo "update_actor_logs=$(find $RUN/verl_run/dump_actor -name update_actor_log.jsonl 2>/dev/null | wc -l)"
    ;;
  clean)
    pkill -9 -f 'main_pp[o]' 2>/dev/null
    pkill -9 -f 'EngineCor[e]' 2>/dev/null
    pkill -9 -f 'WorkerDic[t]' 2>/dev/null
    pkill -9 -f 'ray::[I]' 2>/dev/null
    sleep 3
    rm -rf $A/workers/actor/__pycache__ $A/experimental/agent_loop/__pycache__
    rm -rf $RUN/verl_run/dump_actor $RUN/verl_run/dump_generate $RUN/verl_run/ckpts
    echo "CLEANED PYC_actor_after=$(ls $A/workers/actor/__pycache__/ 2>/dev/null | wc -l)"
    ;;
  start)
    # CLI flag 化:start [--hcu-probe <off|L0|L1|tensor|monitor|combined>] 选采集模式(不传=off=纯训练)
    # verl 是手动钩子(非 megatron Feature 0 行代码),故 flag 在脚本层解析→设 env;接入代码(dp_actor/constants_ppo)读 env
    MODE=off
    shift  # 去掉 "start"
    while [ $# -gt 0 ]; do
      case "$1" in
        --hcu-probe) MODE="$2"; shift 2;;
        --hcu-probe=*) MODE="${1#*=}"; shift;;
        *) shift;;
      esac
    done
    export GPUS_PER_NODE=8
    # 按 mode 设 hcu_probe env(透传 ray worker 见 hcu_verl constants_ppo.py PPO_RAY_RUNTIME_ENV)
    case "$MODE" in
      off)
        echo "[hcu-probe] mode=off 纯训练,不 dump 不 monitor(verl 原行为)";;
      L0)
        export DUMP_ON=1 PROMPTS_ONLY=1 DUMP_ACTOR_CONFIG=$RUN/config_actor.json DUMP_CONFIG_PATH=$RUN/config_generate.json
        echo "[hcu-probe] mode=L0 训推一致 dump(L0 module 级 statistics,裁 response 对齐推理 prefill)";;
      L1)
        gen_cfg L1 L1 statistics
        export DUMP_ON=1 PROMPTS_ONLY=1 TORCHDYNAMO_DISABLE=1 DUMP_ACTOR_CONFIG=$RUN/config_actor_L1.json DUMP_CONFIG_PATH=$RUN/config_generate_L1.json
        echo "[hcu-probe] mode=L1 训推一致 dump(L1 API 级 + TORCHDYNAMO_DISABLE);verl 上未实测、机制就绪(HF 模型理论无 is F.silu 检查、应无需 patch yaml;megatron 才需)";;
      tensor)
        gen_cfg tensor L0 tensor
        export DUMP_ON=1 PROMPTS_ONLY=1 DUMP_ACTOR_CONFIG=$RUN/config_actor_tensor.json DUMP_CONFIG_PATH=$RUN/config_generate_tensor.json
        echo "[hcu-probe] mode=tensor 训推一致 dump(逐元素 tensor,易爆盘);verl 上未系统实测,⚠️ 推理侧逐元素 dump 极慢(可 infer-unpatch 关推理钩子再采训练侧)";;
      monitor)
        export MONITOR_CONFIG=$RUN/config_monitor.json
        echo "[hcu-probe] mode=monitor 训练三通道监控(weight_grad/optimizer/param,挂 dp_actor.update_policy;正常训练不裁 response)";;
      combined)
        export DUMP_ON=1 DUMP_ACTOR_CONFIG=$RUN/config_actor.json DUMP_CONFIG_PATH=$RUN/config_generate.json MONITOR_CONFIG=$RUN/config_monitor.json
        echo "[hcu-probe] mode=combined 正常训练(不裁 response,PROMPTS_ONLY=0)+ 前向统计 dump + monitor 三通道叠加";;
      *)
        echo "[hcu-probe] ERROR unknown mode '$MODE' — 可选 off|L0|L1|tensor|monitor|combined"; exit 1;;
    esac
    echo "[hcu-probe] env: DUMP_ON=${DUMP_ON:-} PROMPTS_ONLY=${PROMPTS_ONLY:-} MONITOR_CONFIG=${MONITOR_CONFIG:-} TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-}"
    exec bash $RUN/run_0.5b.sh > $RUN/verl_run.log 2>&1
    ;;
  reqids)
    # 抽样所有 rank 的 update_actor_log 首行,看 request_ids 是否非空(④ 验证)
    for f in $RUN/verl_run/dump_actor/*/update_actor_log.jsonl; do
      head -1 "$f" 2>/dev/null
    done
    ;;
  dbglog)
    grep -E 'hcu-dbg|request_id_in_rollout' $RUN/verl_run.log | tail -5
    ;;
  waitdump)
    # 阻塞等 update_actor_log 出现(≤120s,留 master 余量),然后抓 ④ 验证关键信息
    for i in $(seq 1 12); do
      n=$(find $RUN/verl_run/dump_actor -name update_actor_log.jsonl 2>/dev/null | wc -l)
      [ "$n" -gt 0 ] && { echo "DUMP_READY after $((i*10))s logs=$n"; break; }
      sleep 10
    done
    echo "--- dbglog(rollout batch 有无 request_id)---"; grep -E 'request_id_in_rollout' $RUN/verl_run.log | tail -2
    echo "--- reqids 首行(④ 验证)---"; for f in $RUN/verl_run/dump_actor/*/update_actor_log.jsonl; do head -1 "$f"; break; done
    echo "--- 崩没崩 ---"; grep -E 'Traceback|Training Progress' $RUN/verl_run.log | tail -2
    ;;
  infer-deploy)
    # ② 容器内:动态定位 vllm_hcu/v1(不写死 py 版本)+ 备份 runner + 落 dispatch_logger + 注入 patch + py_compile 自检
    V1=$(python3 -c 'import vllm_hcu,os;print(os.path.join(os.path.dirname(vllm_hcu.__file__),"v1"))')
    P=$V1/hcu_model_runner.py
    D=$V1/dispatch_logger.py
    cp -n $P ${P}.bak_hcu && echo "backup ok: ${P}.bak_hcu"
    cp $RUN/dispatch_logger.py $D 2>/dev/null && echo "dispatch_logger deployed: $D" || echo "WARN: $RUN/dispatch_logger.py 不在,先放进 \$RUN 再跑"
    python3 $RUN/patch_hcu_dump.py $P
    echo "--- verify ---"
    python3 -m py_compile $P && echo "PYCOMPILE_runner_OK" || echo "PYCOMPILE_runner_FAIL"
    [ -f $D ] && { python3 -m py_compile $D && echo "PYCOMPILE_dispatch_OK" || echo "PYCOMPILE_dispatch_FAIL"; } || echo "dispatch_logger NOT FOUND"
    echo "inject_marker=$(grep -c _hcu_debugger $P)"
    ;;
  infer-unpatch)
    # 还原:hcu_model_runner 复原 + 删 dispatch_logger
    V1=$(python3 -c 'import vllm_hcu,os;print(os.path.join(os.path.dirname(vllm_hcu.__file__),"v1"))')
    P=$V1/hcu_model_runner.py
    [ -f ${P}.bak_hcu ] && cp ${P}.bak_hcu $P && echo "restored hcu_model_runner" || echo "no backup"
    rm -f $V1/dispatch_logger.py && echo "removed dispatch_logger"
    python3 -m py_compile $P && echo "restored_PYCOMPILE_OK"
    ;;
  wait-infer)
    # 阻塞等 vllm worker 推理 dump 初始化([hcu_probe-infer])或捕获错误(≤120s)
    for i in $(seq 1 12); do
      grep -q 'hcu_probe-infer' $RUN/verl_run.log 2>/dev/null && { echo "INFER_INIT_SEEN after $((i*10))s"; break; }
      grep -qiE 'Traceback|hcu_model_runner.py\", line' $RUN/verl_run.log 2>/dev/null && { echo "ERROR_DETECTED after $((i*10))s"; break; }
      sleep 10
    done
    echo "--- infer init([hcu_probe-infer])---"; grep 'hcu_probe-infer' $RUN/verl_run.log | tail -3
    echo "--- errors ---"; grep -iE 'traceback|hcu_model_runner|precisiondebugger|hcu_probe.*fail' $RUN/verl_run.log | tail -5
    echo "--- tail ---"; tail -2 $RUN/verl_run.log
    ;;
  id-match)
    # id 同源决定性验证:训练侧 request_id(32位) 与 推理侧 strip 后缀的 32 位 求交集
    tt=$(mktemp); ti=$(mktemp)
    for f in $RUN/verl_run/dump_actor/*/update_actor_log.jsonl; do grep -oE '[0-9a-f]{32}' "$f" 2>/dev/null; done | sort -u > $tt
    for f in $RUN/verl_run/dump_generate/*/dispatch_log.jsonl; do grep -oE '[0-9a-f]{32}' "$f" 2>/dev/null; done | sort -u > $ti
    echo "训练侧唯一 request_id 数: $(wc -l < $tt)"
    echo "推理侧唯一 request_id 数(strip 后缀): $(wc -l < $ti)"
    echo "交集数(同源证明,应=训练侧数): $(comm -12 $tt $ti | wc -l)"
    echo "训练侧独有(推理侧没有)数: $(comm -23 $tt $ti | wc -l)"
    echo "交集样例:"; comm -12 $tt $ti | head -3
    rm -f $tt $ti
    ;;
  ids-both)
    # 对比训练侧(纯 uuid4.hex)与推理侧(疑似 -后缀)的 request_id 前 32 位是否一一对应(id 同源)
    echo "=== 训练侧 update_actor request_ids ==="
    f=$(find $RUN/verl_run/dump_actor -name update_actor_log.jsonl 2>/dev/null | head -1)
    [ -n "$f" ] && grep -oE '[0-9a-f]{32}(-[0-9a-f]+)?' "$f" | sort -u | head -5
    echo "=== 推理侧 prefill dispatch request_ids ==="
    grep -h '"phase": "prefill"' $RUN/verl_run/dump_generate/*/dispatch_log.jsonl 2>/dev/null | grep -oE '[0-9a-f]{32}(-[0-9a-f]+)?' | sort -u | head -5
    echo "=== 推理侧前 32 位(strip 后缀后) ==="
    grep -h '"phase": "prefill"' $RUN/verl_run/dump_generate/*/dispatch_log.jsonl 2>/dev/null | grep -oE '[0-9a-f]{32}' | sort -u | head -5
    ;;
  infer-check)
    # 推理 dump 产物 + id 同源检查
    echo "PROC_main_ppo=$(ps -ef | grep 'main_pp[o]' | grep -v grep | wc -l)"
    echo "dump_generate_dispatch_logs=$(find $RUN/verl_run/dump_generate -name dispatch_log.jsonl 2>/dev/null | wc -l)"
    echo "dump_generate_dumpjson=$(find $RUN/verl_run/dump_generate -name dump.json 2>/dev/null | wc -l)"
    echo "infer_init_log:"; grep 'hcu_probe-infer' $RUN/verl_run.log | tail -2
    echo "prefill dispatch 样例(request_id):"
    for f in $RUN/verl_run/dump_generate/*/dispatch_log.jsonl; do grep '"phase": "prefill"' "$f" 2>/dev/null | head -1; break; done
    ;;
  compare-env)
    # compare 前置探查:dump 产物 + hcu_probe 安装/CLI/函数 API 可用性(R2)
    echo "train_dump_json=$(find $RUN/verl_run/dump_actor -name dump.json 2>/dev/null | wc -l)"
    echo "gen_dump_json=$(find $RUN/verl_run/dump_generate -name dump.json 2>/dev/null | wc -l)"
    echo "=== hcu_probe pip ==="; pip show hcu_probe 2>/dev/null | grep -E 'Name|Version|Location'
    echo "=== import hcu_probe ==="; python3 -c 'import hcu_probe; print("import OK")' 2>&1 | tail -3
    echo "=== hcu-probe CLI (R2?) ==="; hcu-probe --help 2>&1 | tail -4
    echo "=== 函数 API 路径探测 ==="
    python3 -c 'from hcu_probe.pytorch import compare; print("hcu_probe.pytorch.compare OK:", [x for x in dir(compare) if not x.startswith("_")][:8])' 2>&1 | tail -2
    python3 -c 'from hcu_probe.pytorch.compare.pt_compare import compare as f; print("pt_compare OK")' 2>&1 | tail -1
    python3 -c 'from hcu_probe.pytorch import compare_distributed; print("compare_distributed OK")' 2>&1 | tail -1
    ;;
  *)
    echo "usage: manage.sh {status|clean|start [--hcu-probe MODE]|reqids|dbglog|waitdump|infer-deploy|infer-unpatch|infer-check|ids-both|id-match|compare-env}"
    echo "  start --hcu-probe <off|L0|L1|tensor|monitor|combined>  采集模式(不传=off=纯训练):"
    echo "    off      纯训练,不 dump 不 monitor(verl 原行为)"
    echo "    L0       训推一致 dump,L0 module 级 statistics(裁 response 对齐 prefill)  ✅ 已验证"
    echo "    L1       训推一致 dump,L1 API 级(+TORCHDYNAMO_DISABLE)               ⚠️ verl 未实测(机制就绪;compare 不可行)"
    echo "    tensor   训推一致 dump,逐元素(易爆盘)                                 ⚠️ verl 未系统实测(推理逐元素极慢)"
    echo "    monitor  训练三通道监控(weight_grad/optimizer/param,挂 update_policy) ✅ 已验证(8rank×3通道 csv)"
    echo "    combined 正常训练 + 前向统计 dump(不裁 response)+ monitor 三通道叠加  ✅ 已验证"
    ;;
esac
