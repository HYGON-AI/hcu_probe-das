"""hcuprobe 命令行入口(HCU 侧自维护,不属于上游 msprobe 同步范围)。

转调上游 CLI ``hcuprobe.hcuprobe:main``,仅在导入阶段屏蔽昇腾专属的误导性报错:
上游 CLI 启动时会导入 ``infer/offline/compare/msquickcmp``(注册 offline_dump 子命令),其中 ``import acl``
在 HCU 上必然失败并打印 "Please verify that the CANN environment is properly configured.",
导致每条 hcuprobe 命令(acc_check、compare 等)都带这条报错。上游代码保持不改。
"""

import contextlib

_ASCEND_ONLY_ERRORS = ("Please verify that the CANN environment is properly configured.",)


@contextlib.contextmanager
def _suppress_ascend_only_errors():
    from hcuprobe.core.common.log import logger

    original = logger.error

    def _error(msg, *args, **kwargs):
        if any(text in str(msg) for text in _ASCEND_ONLY_ERRORS):
            return None
        return original(msg, *args, **kwargs)

    logger.error = _error
    try:
        yield
    finally:
        logger.error = original


def main():
    with _suppress_ascend_only_errors():
        from hcuprobe import hcuprobe as upstream_cli
    upstream_cli.main()


if __name__ == "__main__":
    main()
