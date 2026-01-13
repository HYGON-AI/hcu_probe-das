# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------import json
from werkzeug.wrappers.request import Request
from werkzeug import Response
from ..utils.graph_utils import GraphUtils
from ..utils.constant import security_headers


def request_method(allowed_method):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                if len(args) <= 0:
                    raise RuntimeError('Illegal function call, at least 1 parameter is required but got 0')
                request = args[0]
                if not isinstance(request, Request):
                    raise RuntimeError('The request "parameter" is not in a format supported by werkzeug')
                actual_method = request.method.upper()
                expected_method = allowed_method.upper()
                if actual_method != expected_method:
                    raise RuntimeError(f"Method Not Allowed: expected '{expected_method}', got '{actual_method}'")
            except Exception as e:
                result = {'success': False, 'error': str(e)}
                return Response(json.dumps(result), content_type="application/json", headers=security_headers)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                result = {'success': False, 'error': GraphUtils.t('serverError')}
                return Response(json.dumps(result), content_type="application/json", headers=security_headers)
        return wrapper
    return decorator
