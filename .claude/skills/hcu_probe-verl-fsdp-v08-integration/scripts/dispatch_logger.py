# 推理侧 dump ①:DispatchLogger(vllm 0.11.0 / das1.7,无 vllm_hcu,直接进上游 vllm)
# 部署:放进容器 /usr/local/lib/python3.10/dist-packages/vllm/v1/worker/dispatch_logger.py
# 作用:每次 execute_model 记一行调度元数据到 {dump_path}/dispatch_log.jsonl,
#       compare 时选 phase=prefill 的行 → 拿 request_id 关联训练侧 update_actor_log。
# vllm 0.11 SchedulerOutput 实测字段(2026-06-16):
#   - scheduled_new_reqs: list[NewRequestData],NewRequestData.req_id
#   - scheduled_cached_reqs: CachedRequestData.req_ids(decode 才有;prefill 为空)
#   - num_scheduled_tokens: dict[req_id, int]
import json
import time
from pathlib import Path


class DispatchLogger:
    def __init__(self, dump_path, pid, rank=0):
        log_dir = Path(dump_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(log_dir / "dispatch_log.jsonl", "w")
        self._step_counter = 0
        self._pid = pid
        self._rank = rank

    @staticmethod
    def _req_id(req):
        # NewRequestData 的 id 字段名兜底(vllm 0.11 = req_id)
        return getattr(req, "req_id", None) or getattr(req, "request_id", None)

    def log_step(self, scheduler_output):
        new_reqs = scheduler_output.scheduled_new_reqs
        is_prefill = len(new_reqs) > 0
        requests = []
        for req in new_reqs:
            rid = self._req_id(req)
            requests.append(
                {"request_id": rid, "type": "new",
                 "tokens": scheduler_output.num_scheduled_tokens.get(rid, 0)}
            )
        # decode 阶段的 cached reqs(prefill-only 验证通常为空;尽力而为)
        try:
            cached = scheduler_output.scheduled_cached_reqs
            for rid in getattr(cached, "req_ids", None) or []:
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
