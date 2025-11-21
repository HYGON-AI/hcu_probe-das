/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
export interface NodeInfoResult {
  success: boolean;
  data?: {
    npu?: {
      name?: string;
      inputData?: Record<string, unknown>;
      outputData?: Record<string, unknown>;
      stackData?: string;
      suggestions?: Record<string, unknown>;
    } | null;
    bench?: {
      name?: string;
      inputData?: Record<string, unknown>;
      outputData?: Record<string, unknown>;
      stackData?: string;
      suggestions?: Record<string, unknown>;
    } | null;
  };
  error?: string;
}

export type NodeInfoType = NodeInfoResult['data']['npu'];
