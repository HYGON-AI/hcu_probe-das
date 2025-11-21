# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import json
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
