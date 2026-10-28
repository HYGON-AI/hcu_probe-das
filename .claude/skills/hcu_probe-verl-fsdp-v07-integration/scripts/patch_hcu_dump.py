#!/usr/bin/env python3
# vllm_hcu 推理侧 dump ②:注入 GPUModelRunner.__init__ + execute_model
# 容器内跑:python3 /data/hcu_probe_verl/patch_hcu_dump.py
# idempotent(marker _hcu_debugger);锚点核对失败则报错不改。
# 落点(das vllm_hcu 0.18.1;下列行号仅示意、会漂——靠字符串锚点不靠行号):
#   __init__ 末尾 self.layerwise_nvtx_hooks_registered = False(:880)→ 建 debugger + DispatchLogger
#   execute_model 内 model_output = self._model_forward(...)(:4149,with 块内 12 空格缩进)→ 前后 start/stop/step
# 关键设计:
#   - dump_cfg 走 env DUMP_CONFIG_PATH(das runner 无 additional_config,grep 零命中)
#   - 多 vllm worker 进程 → dump_path 按 pid 隔离({dump_generate}/{pid}/...)
#   - 只对 prefill 做 debugger dump(is_prefill=len(scheduled_new_reqs)>0),避免每个 decode step 都 dump 爆盘
#     (等价华为 token_range 限 prefill,但在钩子层门控,不依赖 das vllm 的 token_range 参数)
#   - PrecisionDebugger API 同训练侧 ③(已验证):config_path / .config.dump_path / .start(model=) / .stop() / .step() / seed_all
import sys, os

# 默认:动态定位容器内 vllm_hcu/v1/hcu_model_runner.py(不写死 py 版本);传参可指向源码仓的文件(规范化 patch 用)
def _default_target():
    import vllm_hcu
    return os.path.join(os.path.dirname(vllm_hcu.__file__), "v1", "hcu_model_runner.py")
F = sys.argv[1] if len(sys.argv) > 1 else _default_target()
s = open(F).read()

if "_hcu_debugger" in s:
    print("ALREADY")
    raise SystemExit(0)

# --- 注入 1:__init__ 末尾建 debugger + DispatchLogger ---
INIT_ANCHOR = "        self.layerwise_nvtx_hooks_registered = False\n"
INIT_INJECT = INIT_ANCHOR + '''
        # === hcu_probe 训推一致 dump ②:env DUMP_CONFIG_PATH 开关(das 无 additional_config 故走 env)===
        import os as _os
        self._hcu_debugger = None
        self._hcu_dispatch_logger = None
        _hcu_cfg = _os.environ.get("DUMP_CONFIG_PATH")
        if _hcu_cfg:
            from hcuprobe.pytorch import PrecisionDebugger, seed_all
            seed_all(mode=True)
            self._hcu_debugger = PrecisionDebugger(config_path=_hcu_cfg)
            _hcu_pid = _os.getpid()
            _hcu_pid_path = _os.path.join(self._hcu_debugger.config.dump_path, str(_hcu_pid))
            self._hcu_debugger.config.dump_path = _hcu_pid_path  # 多 worker 按 pid 隔离
            try:
                self._hcu_debugger.service.config.dump_path = _hcu_pid_path
            except Exception:
                pass
            try:
                from vllm_hcu.v1.dispatch_logger import DispatchLogger
                self._hcu_dispatch_logger = DispatchLogger(
                    dump_path=_hcu_pid_path, pid=_hcu_pid,
                    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
            except Exception as _e:
                logger.warning(f"[hcu_probe] DispatchLogger init failed: {_e}")
            logger.warning(f"[hcu_probe-infer] debugger init pid={_hcu_pid} dump={_hcu_pid_path}")
'''

# --- 注入 2:execute_model 的 _model_forward 前后 start / log_step / stop / step ---
FWD_ANCHOR = '''            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )'''
FWD_INJECT = '''            _hcu_on = getattr(self, "_hcu_debugger", None) is not None
            _hcu_prefill = _hcu_on and len(scheduler_output.scheduled_new_reqs) > 0
            if _hcu_prefill:
                self._hcu_debugger.start(model=self.model)
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
            if _hcu_on and self._hcu_dispatch_logger is not None:
                try:
                    self._hcu_dispatch_logger.log_step(scheduler_output)
                except Exception as _e:
                    logger.warning(f"[hcu_probe] dispatch log_step failed: {_e}")
            if _hcu_prefill:
                self._hcu_debugger.stop()
                self._hcu_debugger.step()'''

for name, anchor in [("__init__", INIT_ANCHOR), ("execute_model._model_forward", FWD_ANCHOR)]:
    if anchor not in s:
        print(f"ANCHOR_NOT_FOUND: {name}")
        raise SystemExit(1)

s = s.replace(INIT_ANCHOR, INIT_INJECT, 1)
s = s.replace(FWD_ANCHOR, FWD_INJECT, 1)
open(F, "w").write(s)
print("PATCHED_HCU_DUMP")
