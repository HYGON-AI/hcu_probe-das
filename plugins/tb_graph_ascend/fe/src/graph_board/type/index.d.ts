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

import { SelectionType } from "../../graph_ascend/type";
export interface HierarchyNodeType {
    x: number;
    y: number;
    width: number;
    height: number;
    expand: boolean;
    parentNode: string;
    label: string;
    name: string;
    isRoot: boolean;
    children?: HierarchyNodeType[];
    nodeType: number;
    matchedNodeLink: string[];
    matchedDistributed: { communications_type: 'send' | 'receive'; nodes_info: { [string]: Array<string> } };
    precisionIndex: string;
    overflowLevel: string;
};

export interface ContextMenuItem {
    text?: string;
    rankId?: number;
    component?: any;
    nodeName?: string;
    type?: number;
    children?: ContextMenuItem[];
}

export interface PreProcessDataConfigType {
    colors: { string: { value: number[]; color: string } };
    isOverflowFilter: boolean;
    graphType: GraphType;
}

export type GraphType = 'NPU' | 'Bench' | 'Single';

export interface UseGraphType {
    bindInnerRect: (container: any, data: any) => void;
    bindOuterRect: (container: any, data: any) => void;
    bindText: (container: any, data: any) => void;
    preProcessData: (
        hierarchyObject: { [key: string]: HierarchyNodeType },
        data: any[],
        selectedNode: string,
        config: PreProcessDataConfigType,
        transform: { x: number; y: number; scale: number },
    ) => Array<any>;
    changeNodeExpandState: (nodeInfo: any, metaData: SelectionType) => Promise<any>;
    createComponent: (text, precision, colors: PreProcessDataConfigType['colors']) => any;
    updateHierarchyData: (graphType: string, metaData: SelectionType) => Promise<any>;
}

export interface TransformType {
    x: number;
    y: number;
    scale: number;
};
