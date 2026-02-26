#!/bin/bash
# uninstall.sh - 卸载脚本

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

# 验证安装目录
validate_install_dir() {
    local base_dir="$1"
        
    # 检查目录是否存在
    if [[ ! -d "$base_dir" ]]; then
        error "Specified directory does not exist: $base_dir"
        exit 1
    fi
    
    # 构建完整的模块目录路径
    INSTALL_MODULE_DIR="$base_dir/tools/"
    
    info "Checking for operator_cmp at: $INSTALL_MODULE_DIR"
    
    # 检查模块目录是否存在
    if [[ ! -d "$INSTALL_MODULE_DIR" ]]; then
        error "operator_cmp not found at: $INSTALL_MODULE_DIR"
        error "Base directory: $base_dir"
        error "Expected path: $base_dir/tools/"
        exit 1
    fi
    
    return 0
}

check_write_permission() {
    if [[ ! -w "$INSTALL_MODULE_DIR" ]]; then
        error "No permission to delete $INSTALL_MODULE_DIR"
        error "Current user: $(whoami)"
        error "Please run uninstall with a user that has permissions"
        error "or manually delete: rm -rf $INSTALL_MODULE_DIR"
        return 1
    fi
    return 0
}

# 执行卸载
perform_uninstall() {
    info "Uninstalling mindstudio-accucmp..."
    info "Install path: $INSTALL_MODULE_DIR"
    
    # 删除目录
    if rm -rf "$INSTALL_MODULE_DIR"; then
        info "mindstudio-accucmp has been successfully removed"
        return 0
    else
        error "Deletion failed, please check permissions"
        return 1
    fi
}

# 主函数
main() {
    echo "========================================="
    echo "  mindstudio-accucmp Uninstaller"
    echo "========================================="
    # uninstall.sh 位于 ${INSTALL_DIR}/share/info/operator_cmp/uninstall.sh
    # 向上三层到达 INSTALL_DIR (operator_cmp -> info -> share -> compare)
    # 再向上三层到达 base_dir (compare -> operator_cmp -> tools -> test_cann)
    local _script_dir=$(cd "$(dirname "$0")" && pwd)
    local _install_dir=$(cd "${_script_dir}/../../.." && pwd)
    
    info "Script directory: ${_script_dir}"
    info "INSTALL_DIR: ${_install_dir}"
    
    local _cann_uninstall="${_install_dir}/cann_uninstall.sh"
    
    # 验证安装目录
    if ! validate_install_dir "${_install_dir}"; then
        exit 1
    fi

    if ! check_write_permission; then
        exit 1
    fi
    
    # 执行卸载
    if perform_uninstall; then
        info "Uninstall completed!"
        info "mindstudio-accucmp has been successfully removed"
    else
        error "Error occurred during uninstall"
        exit 1
    fi
    sed -i "/uninstall_package "share/info/operator_cmp"/d" "${_cann_uninstall}"
}

# 执行主函数
main "$@"
