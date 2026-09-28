#!/usr/bin/env python3
"""hcuprobe PyTorch 可用性冒烟测试。

把 scripts/ 下的 demo 各跑一遍，确认：进程退出码为 0、输出里没有 traceback、
在 dump_path 下确实产出了可解析的精度数据文件。

用法:
    python3 scripts/ci_smoke_test.py                  # 跑全部 demo
    python3 scripts/ci_smoke_test.py quick_start      # 只跑指定 demo
    python3 scripts/ci_smoke_test.py --require-device # 无加速器时判失败而非跳过
    python3 scripts/ci_smoke_test.py --keep-workdir   # 保留临时工作目录便于排查
    python3 scripts/ci_smoke_test.py --junit-xml=x.xml

退出码: 0 全部通过（含跳过）；1 存在失败。

环境说明:
    basic_dump.py 建模型时写死 torch.device("cuda")，需要 pytorch 在 cuda
    设备名下有可用的加速器（CUDA / torch_npu 等）。纯 CPU 机器上该 demo
    跑不起来。默认行为是跳过并给出提示，CI 里可以用 --require-device
    把"环境不支持"变成硬失败，避免跳过被当成通过。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
CONFIG_FILE = "config.json"
DEFAULT_TIMEOUT = 600

# demo 名 -> 脚本文件 / 期望的 step 目录数 / 该 demo 的额外断言
DEMOS = {
    "quick_start": {"script": "quick_start.py", "steps": 1},
    "basic_dump": {"script": "basic_dump.py", "steps": 3},
    "module_dump": {"script": "module_dump.py", "steps": 1, "module_keys": True},
    "train": {"script": "train.py", "steps": 1},
}

TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")


class Result:
    def __init__(self, name):
        self.name = name
        self.status = "FAIL"      # PASS / FAIL / SKIP
        self.reason = ""
        self.duration = 0.0


def log(msg):
    print(msg, flush=True)


def write_junit(path, results, elapsed):
    suite = ET.Element(
        "testsuite",
        name="hcuprobe-pytorch-smoke",
        tests=str(len(results)),
        failures=str(sum(1 for r in results if r.status == "FAIL")),
        skipped=str(sum(1 for r in results if r.status == "SKIP")),
        time=f"{elapsed:.2f}",
    )
    for r in results:
        case = ET.SubElement(suite, "testcase", name=r.name, time=f"{r.duration:.2f}")
        if r.status == "FAIL":
            ET.SubElement(case, "failure", message=r.reason)
        elif r.status == "SKIP":
            ET.SubElement(case, "skipped", message=r.reason)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def check_device_available():
    """返回 (可用, 原因)。

    demo 里用 torch.device("cuda")，在原生 CUDA、torch_npu 等场景下都映射到
    "cuda" 这个设备名，所以只认 torch.cuda.is_available()。
    """
    try:
        import torch
    except ImportError as exc:
        return False, f"torch 不可导入: {exc}"

    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return True, f"torch {torch.__version__}, {torch.cuda.device_count()} device(s)"
        return False, f"torch {torch.__version__} 无可用加速器（torch.cuda.is_available() 为 False）"
    except Exception as exc:                      # noqa: BLE001 - 设备层异常种类不可控
        return False, f"探测 torch.cuda 失败: {type(exc).__name__}: {exc}"


def prepare_workdir():
    """把 scripts/ 复制到临时目录，保证 CI 可重复且不污染工作区。

    demo 里的 dump_path 是相对路径，各自会 os.chdir 到脚本所在目录，
    所以必须连 config.json 一起复制，并清掉历史 results/ 避免误判。
    """
    workdir = Path(tempfile.mkdtemp(prefix="hcuprobe-smoke-"))
    for item in SCRIPTS_DIR.iterdir():
        if item.name in ("__pycache__", "results"):
            continue
        if item.is_file():
            shutil.copy2(item, workdir / item.name)
    if not (workdir / CONFIG_FILE).is_file():
        raise FileNotFoundError(f"缺少 {CONFIG_FILE}，无法运行 demo: {SCRIPTS_DIR}")
    return workdir


def run_demo(workdir, spec, timeout):
    script = workdir / spec["script"]
    env = os.environ.copy()
    # 让 demo 里 `from model import ModuleOP` 之类的同目录导入生效
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(workdir), env.get("PYTHONPATH", "")) if p
    )
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def collect_steps(workdir, dump_path):
    """返回 [(step 目录名, proc 目录, {json 文件名: 解析结果})]。"""
    root = Path(dump_path)
    if not root.is_absolute():
        root = workdir / root
    if not root.is_dir():
        return []

    steps = []
    for step_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        procs = sorted(p for p in step_dir.iterdir() if p.is_dir())
        if not procs:
            steps.append((step_dir.name, None, {}))
            continue
        # 多进程场景每个进程一个 proc<PID> 目录，逐个收集
        for proc in procs:
            parsed = {}
            for json_file in proc.glob("*.json"):
                try:
                    parsed[json_file.name] = json.loads(json_file.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    parsed[json_file.name] = exc
            steps.append((step_dir.name, proc, parsed))
    return steps


def validate(spec, workdir, dump_path):
    """返回错误信息列表，空列表代表校验通过。"""
    errors = []
    steps = collect_steps(workdir, dump_path)
    if not steps:
        return [f"{dump_path} 下没有任何 step 目录"]

    expected_steps = spec.get("steps")
    if expected_steps is not None and len(steps) != expected_steps:
        errors.append(f"期望 {expected_steps} 个 step 目录，实际 {len(steps)} 个")

    for step_name, proc, parsed in steps:
        where = f"{step_name}/{proc.name if proc else '?'}"
        if not parsed:
            errors.append(f"{where}: 没有产出任何 json 文件")
            continue

        dump = parsed.get("dump.json")
        if not isinstance(dump, dict):
            errors.append(f"{where}: dump.json 缺失或无法解析")
            continue

        data = dump.get("data")
        if not isinstance(data, dict) or not data:
            errors.append(f"{where}: dump.json 的 data 字段为空")
            continue

        # 抽样校验一个节点的结构，防止"文件在但内容是空壳"
        node = next(iter(data.values()))
        if isinstance(node, dict) and not any(
            k in node for k in ("input_args", "input", "output", "input_kwargs")
        ):
            errors.append(f"{where}: 数据节点缺少 input/output 字段")

        if spec.get("module_keys"):
            # module_dump 采集模块级，key 形如 Module.MyModuleOP.ModuleOP.forward.0
            if not any(k.startswith("Module.") for k in data):
                errors.append(f"{where}: 未发现 Module.* 形式的模块级 key")

    return errors


def run_one(name, spec, workdir, timeout, device_ok, device_reason, require_device):
    res = Result(name)
    if not device_ok and not require_device:
        res.status = "SKIP"
        res.reason = device_reason
        return res

    log(f"\n{'=' * 70}\n[ RUN ] {name} -> {spec['script']}\n{'=' * 70}")
    started = time.time()
    try:
        proc = run_demo(workdir, spec, timeout)
    except subprocess.TimeoutExpired:
        res.duration = time.time() - started
        res.reason = f"超时（>{timeout}s）"
        log(f"[TIMEOUT] {name}: {res.reason}")
        return res

    res.duration = time.time() - started
    log(proc.stdout)
    if proc.stderr:
        log("--- stderr ---")
        log(proc.stderr)

    if proc.returncode != 0:
        res.reason = f"退出码 {proc.returncode}"
        return res
    if TRACEBACK_RE.search(proc.stdout) or TRACEBACK_RE.search(proc.stderr):
        res.reason = "输出中出现 traceback"
        return res

    errors = validate(spec, workdir, spec.get("dump_path"))
    if errors:
        res.reason = "; ".join(errors)
        return res

    res.status = "PASS"
    return res


def report(results, junit_xml, skip_reason):
    passed = [r for r in results if r.status == "PASS"]
    failed = [r for r in results if r.status == "FAIL"]
    skipped = [r for r in results if r.status == "SKIP"]

    log(f"\n{'=' * 70}\n[ SUMMARY ]")
    for r in results:
        line = f"  {r.status:4}  {r.name:14} {r.duration:6.1f}s"
        if r.reason:
            line += f"  {r.reason}"
        log(line)

    if skipped and skip_reason:
        log(f"\n  跳过原因: {skip_reason}")
        log("  如需在无加速器环境强制判定，请使用 --require-device")

    log(f"\n  通过 {len(passed)} / 失败 {len(failed)} / 跳过 {len(skipped)}")
    if junit_xml:
        write_junit(junit_xml, results, sum(r.duration for r in results))
        log(f"  JUnit 报告: {junit_xml}")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="hcuprobe PyTorch 可用性冒烟测试")
    parser.add_argument("demos", nargs="*", choices=sorted(DEMOS) + [[]], default=[],
                        help="只跑指定 demo，默认全部")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"单个 demo 超时秒数，默认 {DEFAULT_TIMEOUT}")
    parser.add_argument("--require-device", action="store_true",
                        help="无可用加速器时判失败而非跳过")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="保留临时工作目录（默认跑完删除）")
    parser.add_argument("--junit-xml", metavar="PATH", default=None,
                        help="输出 JUnit XML 报告")
    args = parser.parse_args()

    # config.json 决定 dump_path；解析失败说明配置本身有问题，直接报错
    try:
        dump_path = json.loads((SCRIPTS_DIR / CONFIG_FILE).read_text())["dump_path"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        log(f"[FATAL] 无法从 {CONFIG_FILE} 读取 dump_path: {exc}")
        return 1

    device_ok, device_reason = check_device_available()
    log(f"[ENV] python {sys.version.split()[0]}")
    log(f"[ENV] {device_reason}")

    selected = args.demos or list(DEMOS)

    if not device_ok and args.require_device:
        log(f"[FATAL] --require-device 已指定，但环境不可用: {device_reason}")
        return 1

    # 无加速器时跑 demo 只会得到设备侧报错，直接给出明确结论，
    # 不去制造误导性的失败信息。
    if not device_ok:
        results = [run_one(name, DEMOS[name], SCRIPTS_DIR, args.timeout,
                           False, device_reason, False)
                   for name in selected]
        return report(results, args.junit_xml, device_reason)

    workdir = None
    try:
        workdir = prepare_workdir()
        log(f"[ENV] workdir {workdir}")
        results = []
        for name in selected:
            spec = dict(DEMOS[name])
            spec["dump_path"] = dump_path
            results.append(run_one(name, spec, workdir, args.timeout,
                                   device_ok, device_reason, True))
        return report(results, args.junit_xml, "")
    finally:
        if workdir and not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        elif workdir:
            log(f"[ENV] 保留临时目录: {workdir}")


if __name__ == "__main__":
    sys.exit(main())
