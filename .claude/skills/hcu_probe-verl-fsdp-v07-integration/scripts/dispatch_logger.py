# vllm_hcu 推理侧 dump ①:DispatchLogger
# 部署:随 infer-deploy 自动落到容器 vllm_hcu/v1/(或手动 scp 进 site-packages/vllm_hcu/v1/)
# 作用:每次 execute_model 记一行调度元数据到 {dump_path}/dispatch_log.jsonl,
#       compare 时选 phase=prefill 且 requests 数=1 的行 → 拿 request_id 关联训练侧 update_actor_log。
# das 与华为(vllm-ascend)差异:
#   - is_prefill 用 len(scheduled_new_reqs)>0 替 AscendAttentionState(das v1 无 attn_state 入参)
#   - scheduled_cached_reqs 用 .resumed_req_ids(das 无华为假设的 .req_ids;decode 才有,prefill 为空)
import json
import time
from pathlib import Path


class DispatchLogger:
    def __init__(self, dump_path: str, pid: int, rank: int = 0):
        log_dir = Path(dump_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(log_dir / "dispatch_log.jsonl", "w")
        self._step_counter = 0
        self._pid = pid
        self._rank = rank

    @staticmethod
    def _req_id(req):
        # das v1 NewRequestData 的 id 字段名兜底(req_id / request_id)
        return getattr(req, "req_id", None) or getattr(req, "request_id", None)

    def log_step(self, scheduler_output) -> None:
        new_reqs = scheduler_output.scheduled_new_reqs
        is_prefill = len(new_reqs) > 0
        requests = []
        for req in new_reqs:
            rid = self._req_id(req)
            requests.append(
                {"request_id": rid, "type": "new",
                 "tokens": scheduler_output.num_scheduled_tokens.get(rid, 0)}
            )
        # decode 阶段的 cached reqs(prefill-only 首验通常为空;尽力而为)
        try:
            cached = scheduler_output.scheduled_cached_reqs
            for rid in getattr(cached, "resumed_req_ids", None) or []:
                requests.append(
                    {"request_id": rid, "type": "cached",
                     "tokens": scheduler_output.num_scheduled_tokens.get(rid, 0)}
                )
        except Exception:
            pass
        record = {
            "source": "dispatch_logger",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "pid": self._pid,
            "rank": self._rank,
            "step": self._step_counter,
            "phase": "prefill" if is_prefill else "decode",
            "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
            "requests": requests,
        }
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()
        self._step_counter += 1

    def close(self):
        if self._fp and not self._fp.closed:
            self._fp.close()
