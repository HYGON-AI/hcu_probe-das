#!/bin/bash

set -e

BUILD_PATH=$(pwd)

BUILD_ARGS=$(getopt -o ha:v:j:ft --long help,release,debug,arch:python-version:,CANN-path:,jobs:,force-rebuild,local,test-cases -- "$@")
eval set -- "${BUILD_ARGS}"

ARCH_TYPE=$(uname -m)
BUILD_TYPE=release
CANN_PATH=""
CONCURRENT_JOBS=16
BUILD_TEST_CASE=False
USE_LOCAL_FIRST=False
PYTHON_VERSION=""

HELP_DOC=$(cat << EOF
Usage: build.sh [OPTION]...\n
Build the C++ part of MsProbe.\n
\n
Arguments:\n
    -a, --arch                    Specify the schema, which generally does not need to be set up.\n
        --CANN-path               Specify the CANN path. When set, the build script will find the dependent files in\n
                                  the specified path.\n
    -j, --jobs                    Specify the number of compilation jobs(default 16).\n
    -f, --force-rebuild           Clean up the cache before building.\n
    -t, --test-cases              Build test cases.\n
        --local                   Prioritize the use of on-premises, third-party resources as dependencies.\n
        --release                 Build the release version(default).\n
        --debug                   Build the debug version.
    -v, --python-version          Specify version of python.
EOF
)

while true; do
    case "$1" in
        -h | --help)
            echo -e ${HELP_DOC}
            exit 0 ;;
        -a | --arch)
            ARCH_TYPE="$2" ; shift 2 ;;
        -v | --python-version)
            PYTHON_VERSION="$2" ; shift 2 ;;
        --release)
            BUILD_TYPE=release ; shift ;;
        --debug)
            BUILD_TYPE=debug ; shift ;;
        --CANN-path)
            CANN_PATH="$2" ; shift 2 ;;
        -j | --jobs)
            CONCURRENT_JOBS="$2"  ; shift 2 ;;
        --local)
            USE_LOCAL_FIRST=True ; shift ;;
        -f | --force-rebuild)
            rm -rf "${BUILD_PATH}/build_dependency" "${BUILD_PATH}/lib" "${BUILD_PATH}/output" "${BUILD_PATH}/third_party" \
                   "${BUILD_PATH}/python/msprobe/lib/_msprobe_c.so"
            shift ;;
        -t | --test-cases)
            BUILD_TEST_CASE=True ; shift ;;
        --)
            shift ; break ;;
        *)
            echo "Unknown argument $1"
            exit 1 ;;
    esac
done

BUILD_OUTPUT_PATH=${BUILD_PATH}/output/${BUILD_TYPE}

cmake -B ${BUILD_OUTPUT_PATH} -S . -DARCH_TYPE=${ARCH_TYPE} -DBUILD_TYPE=${BUILD_TYPE} -DCANN_PATH=${CANN_PATH} \
                                   -DUSE_LOCAL_FIRST=${USE_LOCAL_FIRST} -DBUILD_TEST_CASE=${BUILD_TEST_CASE} \
                                   -DPYTHON_VERSION=${PYTHON_VERSION}
cd ${BUILD_OUTPUT_PATH}
make -j${CONCURRENT_JOBS}

if [[ ! -e ${BUILD_OUTPUT_PATH}/ccsrc/adump/lib_msprobe_c.so ]]; then
    echo "Failed to build lib_msprobe_c.so."
    exit 1
fi

if [[ ! -e ${BUILD_PATH}/python/msprobe/lib ]]; then
    mkdir ${BUILD_PATH}/python/msprobe/lib
fi

cp ${BUILD_OUTPUT_PATH}/ccsrc/adump/lib_msprobe_c.so ${BUILD_PATH}/python/msprobe/lib/_msprobe_c.so
