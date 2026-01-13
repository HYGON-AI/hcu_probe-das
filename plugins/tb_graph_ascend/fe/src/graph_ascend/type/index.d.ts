/*
 * -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
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