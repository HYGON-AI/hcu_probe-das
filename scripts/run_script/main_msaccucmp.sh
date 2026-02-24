#!/bin/bash
# install.sh - 安装脚本
USERNAME=$(id -un)
USERGROUP=$(id -gn)

COMMON_SHELL_PATH="common.sh"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印彩色信息
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建安装目录
create_install_dir() {
    info "创建安装目录: $INSTALL_DIR"

    # 尝试创建目录
    if ! mkdir -p "$INSTALL_DIR"; then
        error "目录创建失败: $INSTALL_DIR"
        error "请检查是否有足够的权限，或手动创建该目录"
        exit 1
    fi
}

function convert_install_path() {
    local _install_path="$1"

    # delete last "/" "/."，并将中间的"/./"替换成"/"
    _install_path=`echo "${_install_path}" | sed -r "s/((\/)|(\/\.))*$//g" | sed -r "s|/\./|/|g"`
    if [ -z "${_install_path}" ]; then
        _install_path="/"
    fi
    # covert relative path to absolute path
    # 处理以 "./" 或 "." 开头的路径
    if [[ "${_install_path}" =~ ^\./.* ]] || [[ "${_install_path}" =~ ^\.$ ]]; then
        # 移除开头的 "./" 或 "."
        _install_path=`echo "${_install_path}" | sed -r "s|^\./||" | sed -r "s|^\.$||"`
        if [ -z "${_install_path}" ]; then
            _install_path="${run_path}"
        else
            _install_path="${run_path}/${_install_path}"
        fi
    else
        local _prefix=`echo "${_install_path}" | cut -d"/" -f1`
        if [ ! -z "${_prefix}" ] && [ "~" != "${_prefix}" ]; then
            _install_path="${run_path}/${_install_path}"
        fi
    fi
    # covert '~' to home path
    local _suffix_path=`echo "${_install_path}" | cut -d"~" -f2`
    if [ "${_suffix_path}" != "${_install_path}" ]; then
        local _home_path=`eval echo "~" | sed "s/\/*$//g"`
        _install_path="${_home_path}${_suffix_path}"
    fi
    # 规范化路径：移除双斜杠，处理相对路径组件
    _install_path=`echo "${_install_path}" | sed -r "s|//+|/|g"`
    # 转换为绝对路径（如果可能）
    if command -v realpath >/dev/null 2>&1; then
        _install_path=$(realpath -m "${_install_path}" 2>/dev/null || echo "${_install_path}")
    fi
    echo "${_install_path}"
}

# 主安装函数
main_install() {
    # 设置安装目录
    INSTALL_DIR=$(get_install_path)

    if [[ "$SILENT_MODE" != "true" ]]; then
        echo "========================================="
        echo "  Ascend-mindstudio-accucmp 安装程序"
        echo "========================================="
        echo "安装目录: $INSTALL_DIR"
        echo ""
    fi

    # 创建安装目录
    create_install_dir

    info "正在安装文件..."

    # 复制文件到安装目录
    if [[ "$SUDO_USED" == "true" ]]; then
        sudo cp -r "$PWD"/msaccucmp/* "$INSTALL_DIR"/
    else
        cp -r "$PWD"/msaccucmp/* "$INSTALL_DIR"/
    fi

    info "安装完成！"
    info "安装目录: $INSTALL_DIR"
}


function get_install_path() {
    if [ -z "${input_install_path}" ]; then
        local _install_path
        _install_path="/usr/local/Ascend/operator_cmp/compare"
    else
        _install_path="${input_install_path}/operator_cmp/compare"
    fi
    install_path=$(convert_install_path "${_install_path}")
    echo "${install_path}"
}

# 主函数
main() {
    # 执行安装
    main_install

    # 清理（makeself 会自动清理临时目录）
    exit 0
}

source ${COMMON_SHELL_PATH}
#参数初始化
install_file=""
input_install_path=""
install_for_all=n
install_flag=n
uninstall_flag=n
upgrade_flag=n
quiet_flag=n
version_flag=n
check_flag=n
pylocal=y
force_flag=n

# get run package path
run_path=`echo "$2" | cut -d"-" -f3-`
if [ -z "${run_path}" ]; then
    run_path=`pwd`
else
    # delete last "/" "/."
    run_path=`echo "${run_path}" | sed "s/((\/)|(\/\.))*$//g"`
    [ -z "${run_path}" ] && run_path="/"
    if [ ! -d "${run_path}" ]; then
        log_and_print $LEVEL_ERROR "Run package path is invalid: $run_path"
        exit 1
    fi
    # 确保 run_path 是绝对路径
    if [[ ! "${run_path}" =~ ^/ ]]; then
        run_path="$(cd "${run_path}" && pwd)"
    fi
fi

shift 2

start_log

for arg in "$@"; do
    case "$arg" in
    --force)
        force_flag=y
        ;;
    --check)
        check_flag=y
        ;;
    --quiet)
        quiet_flag=y
        ;;
    --install-path=*)
        input_install_path=`echo "$arg" | cut -d"=" -f2`
        ;;
    --install-for-all)
        install_for_all=y
        ;;
    --full)
        install_flag=y
        ;;
    --install)
        install_flag=y
        ;;
    --run)
        install_flag=y
        ;;
    --devel)
        install_flag=y
        ;;
    --uninstall)
        uninstall_flag=y
        ;;
    --upgrade)
        upgrade_flag=y
        ;;
    --version)
        version_flag=y
        ;;
    -*)
        echo "Unsupported parameters: $arg"
        ;;
    *)
        if [ ! -z "$arg" ]; then
            echo "Unsupported parameters: $arg"
        fi
        break
        ;;
    esac
done

# 执行主函数
main
