import os
import shutil
import subprocess
import sys

from msprobe.core.common.log import logger


def run_ut():
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    cov_dir = os.path.dirname(f"{cur_dir}/../../python/msprobe")
    report_dir = os.path.join(cur_dir, "report")
    cov_config_path = os.path.join(cur_dir, ".coveragerc")
    final_xml_path = os.path.join(report_dir, "final.xml")
    html_cov_report = os.path.join(report_dir, "htmlcov")
    xml_cov_report = os.path.join(report_dir, "coverage.xml")

    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)

    all_ut_dirs = []
    for item in os.listdir(cur_dir):
        item_path = os.path.join(cur_dir, item)
        if os.path.isdir(item_path) and not item.startswith("."):
            all_ut_dirs.append(item_path)

    mindspore_ut_path = os.path.join(cur_dir, "mindspore_ut")
    other_ut_paths = []
    mindspore_exists = False

    for dir_path in all_ut_dirs:
        if dir_path == mindspore_ut_path:
            mindspore_exists = True
        else:
            other_ut_paths.append(dir_path)

    # mindspore_ut在最后跑
    ut_paths = other_ut_paths
    if mindspore_exists:
        ut_paths.append(mindspore_ut_path)
    logger.info(f"UT execution order: {ut_paths}")

    pytest_cmd = [
                     "python3", "-m", "pytest",
                     *ut_paths,
                     f"--junitxml={final_xml_path}",
                     f"--cov-config={cov_config_path}",
                     f"--cov={cov_dir}",
                     "--cov-branch",
                     f"--cov-report=html:{html_cov_report}",
                     f"--cov-report=xml:{xml_cov_report}",
                     "-v"
                 ]

    try:
        with subprocess.Popen(
                pytest_cmd,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
        ) as proc:
            for line in proc.stdout:
                logger.info(line.strip())

            proc.wait()

            if proc.returncode == 0:
                logger.info("Unit tests executed successfully.")
                return True
            else:
                logger.error("Unit tests execution failed.")
                return False
    except Exception as e:
        logger.error(f"An error occurred during test execution: {e}")
        return False


if __name__ == "__main__":
    if run_ut():
        sys.exit(0)
    else:
        sys.exit(1)
