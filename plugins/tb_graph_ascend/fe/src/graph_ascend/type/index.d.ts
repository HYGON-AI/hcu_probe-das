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
export interface ProgressType {
    progress?: number;
    progressValue?: number;
    size?: number;
    read?: number;
    done?: boolean;
}

export interface SelectedItemType {
    value: number;
    label: string;
}

export interface SelectionType {
    run: string;
    tag: string;
    type: 'json' | 'db';
    lang: 'zh-CN' | 'en';
    microStep?: number;
    step?: number;
    rank?: number;
}

export interface GraphConfigType {
    tooltips: string;
    colors: Record<string, {
        value: number[];
        color: string;
    }>;
    overflowCheck: boolean;
    microSteps: number;
    isSingleGraph: boolean;
    matchedConfigFiles: string[];
    task: string;
    ranks: number[];
    steps: number[]
}

export interface GraphAllNodeType {
    npuNodeList: string[];
    benchNodeList: string[];
    npuUnMatchNodes: string[];
    benchUnMatchNodes: string[];
    npuMatchNodes: string[];
    benchMatchNodes: string[];
}

export interface NodeListType {
    npu: string[];
    bench: string[];
}

export interface UnmatchedNodeType {
    npuNodeList: string[];
    benchNodeList: string[];
}

export interface LoadGraphFileInfoListType {
    data: {
        [string]: string[];
    };
    error: [
        {
            run: string;
            tag: string;
            info: string;
        }
    ];
}