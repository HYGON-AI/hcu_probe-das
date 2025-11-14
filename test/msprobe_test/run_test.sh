#!/bin/bash
CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..
TEST_DIR=${TOP_DIR}/"msprobe_test"
SRC_DIR=${TOP_DIR}/../python/

install_pytest() {
    if ! pip show pytest &> /dev/null; then
        echo "pytest not found, trying to install..."
        pip install pytest
    fi

    if ! pip show pytest-cov &> /dev/null; then
        echo "pytest-cov not found, trying to install..."
        pip install pytest-cov
    fi
}

run_ut() {
    install_pytest

    export PYTHONPATH=${SRC_DIR}:${PYTHONPATH}
    python3 run_ut.py
}

main() {
    cd ${TEST_DIR} && run_ut
}

main $@
