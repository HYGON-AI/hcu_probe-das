import os
import shutil
import subprocess
import sys

from msprobe.core.common.log import logger


def run_ut():
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    ut_path = cur_dir
    cov_dir = os.path.dirname(cur_dir)
    report_dir = os.path.join(cur_dir, "report")
    cov_config_path = os.path.join(cur_dir, ".coveragerc")
    final_xml_path = os.path.join(report_dir, "final.xml")
    html_cov_report = os.path.join(report_dir, "htmlcov")
    xml_cov_report = os.path.join(report_dir, "coverage.xml")

    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)

    pytest_cmd = [
                     "python3", "-m", "pytest",
                     ut_path,
                     f"--junitxml={final_xml_path}",
                     f"--cov-config={cov_config_path}",
                     f"--cov={cov_dir}",
                     "--cov-branch",
                     f"--cov-report=html:{html_cov_report}",
                     f"--cov-report=xml:{xml_cov_report}",
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
