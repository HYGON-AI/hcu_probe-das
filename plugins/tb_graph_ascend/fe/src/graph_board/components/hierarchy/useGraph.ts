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
import * as d3 from 'd3';
import { maybeTruncateString, darkenColor, safeJSONParse } from '../../../utils/index';
import request from '../../../utils/request';
import { isEmpty } from 'lodash';
import { HierarchyNodeType, PreProcessDataConfigType, GraphType } from '../../type';
import { SelectionType } from '../../../graph_ascend/type';
import { UseGraphType } from '../../type';
import {
    DURATION_TIME,
    NODE_TYPE_STYLES,
    SELECTED_STROKE_COLOR,
    NO_MATCHED_NODE_COLOR,
    BENCH_NODE_COLOR,
    BENCH_STROKE_COLOR,
    BASE_NODE_COLOR,
    NPU_PREFIX,
    BENCH_PREFIX,
    NODE_TYPE,
    OVERFLOW_COLOR,
    STROKE_WIDTH,
    SELECTED_STROKE_WIDTH,
} from '../../../common/constant';
const useGraph = (): UseGraphType => {
    const preProcessData: UseGraphType['preProcessData'] = (
        hierarchyObject: { [key: string]: HierarchyNodeType },
        data: Array<HierarchyNodeType>,
        selectedNode,
        config: PreProcessDataConfigType,
        transform: { x: number; y: number; scale: number },
    ) => {
        // 遍历数据并应用样式
        const { colors, isOverflowFilter, graphType } = config;
        // 优化性能，渲染3屏节点上中下各1000的高度
        let virtualNodes = data.filter(
            (d) =>
                d.y >= (-Number(transform.y) - 1000) / Number(transform.scale) &&
                d.y <= (-Number(transform.y) + 2000) / Number(transform.scale),
        );
        // virtualNodes的父节点
        const parentsVirtualNodes: Array<HierarchyNodeType> = [];
        virtualNodes.forEach((d) => {
            let node: HierarchyNodeType | undefined = d;
            while (node?.parentNode) {
                const parent = hierarchyObject[node.parentNode];
                if (parent && virtualNodes.indexOf(parent) === -1 && parentsVirtualNodes.indexOf(parent) === -1) {
                    parentsVirtualNodes.push(parent);
                }
                node = parent;
            }
        });
        virtualNodes = [...new Set([...parentsVirtualNodes.reverse(), ...virtualNodes])]; // 父节点放在前面，不然会覆盖子节点
        const renderData = virtualNodes.map((d) => {
            let precisionColor = isOverflowFilter ? getOverflowColor(d) : getPrecisionColor(d, colors, graphType);
            let strokeColor;
            if (d.name === selectedNode) {
                strokeColor = SELECTED_STROKE_COLOR;
            } else if (graphType === 'NPU' || graphType === 'Single') {
                strokeColor = darkenColor(precisionColor, 40);
            } else {
                strokeColor = BENCH_STROKE_COLOR;
            }
            if (d.nodeType === NODE_TYPE.API_LIST || d.nodeType === NODE_TYPE.MULTI_COLLECTION) {
                precisionColor = 'white';
            }
            return {
                ...d,
                ...NODE_TYPE_STYLES[d.nodeType],
                color: precisionColor,
                stroke: strokeColor,
                strokeWidth: d.name === selectedNode ? SELECTED_STROKE_WIDTH : STROKE_WIDTH,
            };
        });
        return renderData;
    };

    const getPrecisionColor = (
        node: HierarchyNodeType,
        colors: PreProcessDataConfigType['colors'],
        graphType: GraphType,
    ) => {
        if (!colors || !graphType) {
            return NO_MATCHED_NODE_COLOR;
        }
        if (isEmpty(node.matchedNodeLink)) {
            return Object.keys(colors).find((color) => colors[color].value === '无匹配节点') ?? NO_MATCHED_NODE_COLOR;
        }
        if (graphType === 'Bench') {
            return BENCH_NODE_COLOR;
        }

        const precisionValue = parseFloat(node.precisionIndex);
        return calcClolorByPrecision(precisionValue, colors);
    };

    const calcClolorByPrecision = (precisionValue: number, colors: PreProcessDataConfigType['colors']) => {

        if (isNaN(precisionValue)) {
            return BASE_NODE_COLOR; // 默认返回灰色
        }
        for (const [color, config] of Object.entries(colors)) {
            if (Array.isArray(config.value)) {
                const [min, max] = config.value;
                const isWithinRange = precisionValue >= min && precisionValue < max;
                const isMaxRange = max === 1 && precisionValue === 1;
                const isMinRange = min === 0 && precisionValue === 0;
                if (isWithinRange || isMaxRange || isMinRange) {
                    return color; // 返回对应的填充颜色
                }
            }
        }
        // 如果没有匹配到范围，返回默认颜色（灰色）
        return NO_MATCHED_NODE_COLOR;
    };

    const createComponent: UseGraphType['createComponent'] = (text, precision, colors: PreProcessDataConfigType['colors']) => {
        const component = document.createElement('vaadin-context-menu-item');
        component.appendChild(document.createTextNode(text));
        component.style.background = calcClolorByPrecision(precision, colors);
        component.style.margin = '3px 0';
        component.style.cursor = 'pointer';
        component.style.border = '1px solid #615f5f';

        return component;
    };

    const getOverflowColor = (node) => {
        const overflowLevel = node.overflowLevel;
        switch (overflowLevel) {
            case 'medium':
                return OVERFLOW_COLOR.medium;
            case 'high':
                return OVERFLOW_COLOR.high;
            case 'critical':
                return OVERFLOW_COLOR.critical;
            default:
                return OVERFLOW_COLOR.default;
        }
    };

    const bindInnerRect: UseGraphType['bindInnerRect'] = (container, data) => {
        // 绑定数据到 innnerReact 元素
        const innnerReact = container.selectAll('.inner-rect').data(data, (d: any) => d.name);
        innnerReact
            .transition()
            .duration(DURATION_TIME)
            .attr('opacity', 1)
            .attr('x', (d: any) => d.x)
            .attr('y', (d: any) => d.y)
            .attr('width', (d: any) => d.width)
            .attr('fill', (d: any) => d.color);

        innnerReact
            .enter()
            .append('rect')
            .attr('name', (d: any) => d.name)
            .attr('class', 'inner-rect')
            .attr('rx', (d: any) => d.rx)
            .attr('ry', (d: any) => d.ry)
            .attr('fill', (d: any) => d.color)
            .attr('x', (d: any) => d.x)
            .attr('y', (d: any) => d.y)
            .attr('width', (d: any) => d.width)
            .attr('height', 15)
            .attr('opacity', 0)
            .transition()
            .duration(DURATION_TIME + 60)
            .attr('opacity', 1);

        innnerReact
            .exit()
            .transition()
            .duration(DURATION_TIME - 60)
            .attr('opacity', 0)
            .remove();
        // 强制 DOM 排列顺序与数据一致
        innnerReact.order();
    };

    const bindOuterRect: UseGraphType['bindOuterRect'] = (container, data): void => {
        // 绑定数据到 outerReact 元素
        const outerReact = container.selectAll('.outer-rect').data(data, (d: any) => d.name);
        outerReact
            .transition()
            .duration(DURATION_TIME)
            .attr('opacity', 1)
            .attr('x', (d: any) => d.x)
            .attr('y', (d: any) => d.y)
            .attr('width', (d: any) => d.width)
            .attr('height', (d: any) => d.height)
            .attr('stroke', (d: any) => d.stroke)
            .attr('stroke-width', (d: any) => d.strokeWidth);

        outerReact
            .enter()
            .append('rect')
            .attr('name', (d: any) => d.name)
            .attr('class', 'outer-rect')
            .attr('rx', (d: any) => d.rx)
            .attr('ry', (d: any) => d.ry)
            .attr('fill', 'transparent')
            .attr('stroke', (d: any) => d.stroke)
            .attr('stroke-width', (d: any) => d.strokeWidth)
            .attr('stroke-dasharray', (d: any) => d.strokeDasharray)
            .attr('width', (d: any) => d.width)
            .attr('height', 15)
            .attr('x', (d: any) => d.x)
            .attr('y', (d: any) => d.y)
            .transition()
            .duration(DURATION_TIME + 60)
            .attr('height', (d: any) => d.height)
            .attr('opacity', 1);

        outerReact
            .exit()
            .transition()
            .duration(DURATION_TIME - 60)
            .attr('opacity', 0)
            .remove();

        // 强制 DOM 排列顺序与数据一致
        outerReact.order();
    };

    const bindText: UseGraphType['bindText'] = (container, data) => {
        // 绑定文本数据
        const texts = container.selectAll('text').data(data, (d: any) => d.name);
        // 更新现有文本
        texts
            .transition()
            .duration(DURATION_TIME)
            .attr('opacity', 1)
            .attr('x', (d: any) => d.x + (d.width / 2))
            .attr('y', (d: any) => d.y + 8);

        // 添加新文本
        texts
            .enter()
            .append('text')
            .attr('name', (d: any) => d.name)
            .attr('x', (d: any) => d.x + (d.width / 2))
            .attr('y', (d: any) => d.y + 8)
            .attr('dy', '0.35em')
            .attr('text-anchor', 'middle')
            .text((d: any) => maybeTruncateString(d.label, 9, d.width))
            .each(function (d) {
                // @ts-expect-error d3.select(this) this is a d3 selection
                d3.select(this).append('title').text(d.label);
            })
            .style('font-size', (d) => `${d.fontSize}px`)
            .attr('opacity', 0)
            .transition()
            .duration(DURATION_TIME + 60)
            .attr('opacity', 1);

        // 删除多余文本
        texts
            .exit()
            .transition()
            .duration(DURATION_TIME - 60)
            .attr('opacity', 0) // 动画过渡：透明度逐渐变为 0
            .remove();


        // 强制 DOM 排列顺序与数据一致
        texts.order();
    };

    const changeNodeExpandState: UseGraphType['changeNodeExpandState'] = async (nodeInfo: any, metaData: SelectionType): Promise<any> => {
        try {
            const metaDataSafe = safeJSONParse(JSON.stringify(metaData));
            const params = {
                nodeInfo,
                metaData: metaDataSafe,
            };
            const result = await request({
                url: 'changeNodeExpandState',
                method: 'POST',
                data: params,
                timeout: 10000,
            });
            return result;
        } catch (err) {
            return {
                suucess: false,
                error: '网络请求失败',
            };
        }
    };

    const updateHierarchyData = async (graphType: string, metaData: SelectionType): Promise<any> => {

        try {
            const params = {
                metaData,
                graphType
            };
            const result = await request({ url: 'updateHierarchyData', method: 'POST', data: params });
            return result;
        } catch (err) {
            return {
                suucess: false,
                error: '网络请求失败',
            };
        }
    };

    return {
        bindText,
        bindInnerRect,
        bindOuterRect,
        preProcessData,
        createComponent,
        updateHierarchyData,
        changeNodeExpandState,
    };
};

export default useGraph;
