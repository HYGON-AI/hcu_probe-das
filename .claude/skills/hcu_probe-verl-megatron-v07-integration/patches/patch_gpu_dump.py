#!/usr/bin/env python3
# 推理侧 dump ②:注入 vllm 0.11 GPUModelRunner.__init__ + execute_model
# 容器内跑:python3 /path/to/shared/patch_gpu_dump.py
# idempotent(marker _hcu_debugger);锚点核对失败则报错不改。
# 落点(上游 vllm 0.11.0 / das1.7,无 vllm_hcu,2026-06-16 实测):
#   GPUModelRunner.__init__ 末尾 self.sampled_token_ids_pinned_cpu = torch.empty(...) → 建 debugger + DispatchLogger
#   execute_model 内 model_output = self.model(...)(with set_forward_context 块内,12 空格缩进)→ 前后 start/stop/step
# 关键设计同 FSDP:DUMP_CONFIG_PATH env(das runner 无 additional_config)/ dump_path 按 pid 隔离 / 只对 prefill dump
#   (is_prefill = len(scheduled_new_reqs)>0,避免每个 decode step dump 爆盘)
# PrecisionDebugger API 同训练侧(已验证):config_path / .config.dump_path / .start(model=) / .stop() / .step() / seed_all
import sys

# 默认改容器 site-packages;传参可指向源码副本(规范化 patch 用)
F = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_model_runner.py"
s = open(F).read()

if "_hcu_debugger" in s:
    print("ALREADY")
    raise SystemExit(0)

# --- 注入 1:GPUModelRunner.__init__ 末尾建 debugger + DispatchLogger ---
INIT_ANCHOR = """        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_model_len, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory)"""
INIT_INJECT = INIT_ANCHOR + '''

        # === hcu_probe 训推一致 dump ②:env DUMP_CONFIG_PATH 开关(无 additional_config 故走 env)===
        import os as _os
        self._hcu_debugger = None
        self._hcu_dispatch_logger = None
        _hcu_cfg = _os.environ.get("DUMP_CONFIG_PATH")
        if _hcu_cfg:
            from hcuprobe.pytorch import PrecisionDebugger, seed_all
            # mode=False:固定种子(seed=1234)但不开 use_deterministic_algorithms。
            # vllm 0.11 sampler 的 torch.cumsum 无确定性实现,mode=True 会让 vllm 起 engine 的
            # profile_run(dummy sampler 探显存)崩;前向 dump 比的是 layernorm 输出,固定种子已够可比。
            seed_all(mode=False)
            self._hcu_debugger = PrecisionDebugger(config_path=_hcu_cfg)
            _hcu_pid = _os.getpid()
            _hcu_pid_path = _os.path.join(self._hcu_debugger.config.dump_path, str(_hcu_pid))
            self._hcu_debugger.config.dump_path = _hcu_pid_path  # 多 worker 按 pid 隔离
            try:
                self._hcu_debugger.service.config.dump_path = _hcu_pid_path
            except Exception:
                pass
            try:
                from vllm.v1.worker.dispatch_logger import DispatchLogger
                self._hcu_dispatch_logger = DispatchLogger(
                    dump_path=_hcu_pid_path, pid=_hcu_pid,
                    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
            except Exception as _e:
                logger.warning(f"[hcu_probe] DispatchLogger init failed: {_e}")
            logger.warning(f"[hcu_probe-infer] debugger init pid={_hcu_pid} dump={_hcu_pid_path}")'''

# --- 注入 2:execute_model 的 self.model(...) 前后 start / log_step / stop / step ---
FWD_ANCHOR = """            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )"""
FWD_INJECT = '''            _hcu_on = getattr(self, "_hcu_debugger", None) is not None
            _hcu_prefill = _hcu_on and len(scheduler_output.scheduled_new_reqs) > 0
            if _hcu_prefill:
                # vllm 0.11 self.model 是 CUDAGraphWrapper(.runnable 才是真实 nn.Module);
                # hcu_probe check_model 只认 nn.Module,循环解包。prefill 走 eager(__call__ 直接调
                # self.runnable),hook 挂解包后的 module 会命中;decode 走 cudagraph 不 dump,无碍。
                _hcu_m = self.model
                while not isinstance(_hcu_m, torch.nn.Module) and hasattr(_hcu_m, "runnable"):
                    _hcu_m = _hcu_m.runnable
                self._hcu_debugger.start(model=_hcu_m)
            model_output = self.model(
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

for name, anchor in [("__init__", INIT_ANCHOR), ("execute_model.self.model", FWD_ANCHOR)]:
    if anchor not in s:
        print(f"ANCHOR_NOT_FOUND: {name}")
        raise SystemExit(1)

s = s.replace(INIT_ANCHOR, INIT_INJECT, 1)
s = s.replace(FWD_ANCHOR, FWD_INJECT, 1)
open(F, "w").write(s)
print("PATCHED_GPU_DUMP")
