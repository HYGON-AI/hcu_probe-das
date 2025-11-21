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

import { PolymerElement, html } from '@polymer/polymer';
import { customElement, property, observe } from '@polymer/decorators';
import * as d3 from 'd3';
import useGraph from './useGraph';
import { changeGraphPosition } from '../../../utils/index';
import { parseTransform } from '../../../utils/index';
import { isEmpty, throttle } from 'lodash';
import * as minimap from '../minimap/minimap';
import {
    NPU_PREFIX,
    BENCH_PREFIX,
    MOVE_STEP,
    SCALE_STEP,
    NODE_TYPE,
    MAX_SCALE,
    MIN_SCALE,
    PREFIX_MAP,
    DB_TYPE
} from '../../../common/constant';
import '../minimap/index';
import '@vaadin/context-menu';
import { Notification } from '@vaadin/notification';
import type { UseGraphType } from '../../type';
import type { HierarchyNodeType, ContextMenuItem, PreProcessDataConfigType, GraphType } from '../../type';
import type { ContextMenuItemSelectedEvent } from '@vaadin/context-menu';
import type { SelectionType } from '../../../graph_ascend/type';
import i18next from 'i18next';

const EXPAND_MATCHED_NODE = 1;
const DATA_COMMUNICATION = 2;
const DATA_COMMUNICATION_TYEPE = {
    send: '数据发送',
    receive: '数据接收',
    send_receive: '数据发送接收',
};
@customElement('graph-hierarchy')
class Hierarchy extends PolymerElement {
    static readonly template = html`
       <style>
        :host {
          display: block;
          width: 100%;
          height: 100%;
        }
        .wrapper{
           width: 100%;
           height: 100%;
           position: relative;
        }
        .loading-wrapper{
            position: absolute;
            width: 100%;
            height: 20%;
            color: rgba(37, 37, 37, 0.8);
            background-color: rgba(255, 255, 255, 0.76);
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 20px;
            font-weight: 600;
        }
        .outer-rect {
            cursor: default
        }
        .inner-rect{
            cursor: pointer
        }
   
        text {
          font-family: Arial, sans-serif;
          font-size: 8px;
          white-space: nowrap;
          cursor: pointer;
          user-select: none;
        }
        #graph {
          width: 100%;
          height: 100%;
        }
        #minimap {
            position: absolute;
            width: 150px;
            right: 10px;
            top: 10px;
        }
        #context-menu{
            height: 100%;
            width: 100%;
        }
      </style>
      <div class='wrapper'>
        <template is='dom-if' if='[[loading]]'>
            <div class='loading-wrapper'>
                Loading......
            </div>
        </template>
        <vaadin-context-menu items="[[contextMenuItems]]" id='context-menu' oncontextmenu="[[itemSelected]]"  on-item-selected=[[itemSelected]]>
            <svg id="graph">
                <g id="root" transform="translate(36,72) scale(1.8)"></g>
            </svg>  
        </vaadin-context-menu>
        <template is="dom-if" if="[[minimapVis]]">
            <graph-minimap id="minimap"></graph-minimap>
        <template>
    </div>
  `;
    @property({ type: Object })
    t: Function = (key) => i18next.t(key);

    @property({ type: String })
    graphType: GraphType = 'NPU';

    @property({ type: Boolean })
    loading: boolean = false;

    @property({ type: Object, notify: true })
    colors: PreProcessDataConfigType['colors'] = {} as PreProcessDataConfigType['colors'];

    @property({ type: Boolean, notify: true })
    isOverflowFilter: boolean = false;

    @property({ type: Object })
    selection: SelectionType = {} as SelectionType;

    @property({ type: Boolean, notify: true })
    selectedNode = '';

    @property({ type: Object })
    hierarchyData: Array<HierarchyNodeType> = [];

    @property({ type: Object })
    hierarchyObject: { [key: string]: HierarchyNodeType } = {};

    @property({ type: String })
    rootName = '';

    @property({ type: Boolean })
    needChangeNodeCenter: boolean = true;

    @property({ type: Object })
    _zoomStartCoords: { x: number; y: number } | null = null;

    @property({ type: Object })
    _zoomTransform: { x: number; y: number } | null = null;

    @property({ type: Boolean })
    minimapVis: boolean = true;

    @property({ type: Boolean })
    isSyncExpand: boolean = true;

    @property({ type: Object })
    contextMenuItems: Array<ContextMenuItem> = [];

    @property({ type: String })
    hightLightNodeName: string = '';

    @property({ type: Object })
    hightLightMatchedNode: ((matchedNodes, graphType) => void) | null = null;

    useGraph: UseGraphType = useGraph();
    container: HTMLElement | null | undefined;
    graph: HTMLElement | null | undefined;
    minimap: minimap.Minimap | null | undefined;
    cleanEventLisetener: ((...args: any[]) => void) | null = null;

    @observe('selectedNode')
    observeSelectNode() {
        this.changeSelectNode(this.selectedNode);
    }

    // 颜色变化
    @observe('colors', 'isOverflowFilter')
    reRenderGraph() {
        this.renderGraph(this.hierarchyData, this.hightLightNodeName);
    }

    override ready(): void {
        super.ready();
        this.graph = this.shadowRoot?.querySelector('#graph'); // 获取svg元素
        this.container = this.shadowRoot?.querySelector('#root'); // 获取g元素
    }

    async initHhierarchy(selectedNode) {
        if (isEmpty(this.selection) || !this.graphType) {
            return;
        }
        const nodeInfo = {
            nodeName: 'root', // 去掉前缀
            nodeType: this.graphType,
        };
        const { success, data } = await this.changeNodeExpandState(nodeInfo);
        if (success) {
            const hierarchyObject = data;
            const hierarchyData = Object.values(hierarchyObject) as Array<HierarchyNodeType>;
            // 清空container下面的所有子元素
            if (this.container) {
                this.container.innerHTML = '';
                d3.select(this.container as HTMLElement).attr('transform', 'translate(32,32) scale(1.8)');
            }
            if (this.cleanEventLisetener) {
                this.cleanEventLisetener();
            }
            this.cleanEventLisetener = this.bindEventLisetener();
            this.changeSelectNode(selectedNode); // 初始化选中节点,支持节点通信跳转
            this.set('rootName', Object.keys(hierarchyObject)[0]);
            this.set('hierarchyData', hierarchyData);
            this.set('hierarchyObject', hierarchyObject);
            setTimeout(this.initMinimap, 500); // container初始化后，初始化minimap
        }
    }

    initMinimap = () => {
        const minimapEle = this.shadowRoot?.querySelector('#minimap') as HTMLElement;
        if (!this.container || !minimapEle) {
            return;
        }
        const transformStr = this.container.getAttribute('transform') || '';
        const initialTransform = parseTransform(transformStr);
        const newTransform = d3.zoomIdentity
            .translate(initialTransform.x, initialTransform.y)
            .scale(initialTransform.scale);
        const mainZoom = d3.zoom().on('zoom', (event: d3.D3ZoomEvent<SVGElement, unknown>) => {
            this._zoomTransform = event.transform;
            if (!this._zoomStartCoords) {
                this._zoomStartCoords = this._zoomTransform;
            }
            if (this.container) {
                d3.select(this.container as HTMLElement).attr('transform', event.transform.toString());
            }
            this.renderGraph(this.hierarchyData, this.hightLightNodeName);
            this.minimap?.zoom(event.transform); // Notify the minimap.
        });

        this.minimap = (minimapEle as any)?.init(this.graph, this.container, mainZoom, 160, 10);
        this.minimap?.zoom(newTransform);
    };

    renderGraph(data, selectedNode, transform = this.getContainerTransform()) {
        if (!this.shadowRoot) {
            return;
        }
        const container = d3.select(this.container as HTMLElement);
        // 数据预处理
        const prefix = PREFIX_MAP[this.graphType];
        const selectedNodeName = selectedNode.startsWith(prefix) ? selectedNode : `${prefix}${selectedNode}`; // 加上前缀
        const config = { colors: this.colors, isOverflowFilter: this.isOverflowFilter, graphType: this.graphType };
        const renderData = this.useGraph.preProcessData(this.hierarchyObject, data, selectedNodeName, config, transform);
        this.useGraph.bindInnerRect(container, renderData);
        this.useGraph.bindOuterRect(container, renderData);
        this.useGraph.bindText(container, renderData);
        if (this.minimap) {
            setTimeout(() => this.minimap?.update(), 500);
        }
    }

    getContainerTransform() {
        let transform;
        if (this.container) {
            const transformStr = this.container.getAttribute('transform') || '';
            transform = parseTransform(transformStr);
        } else {
            transform = { x: 32, y: 32, scale: 1.8 };
        }
        return transform;
    }

    async changeSelectNode(selectedNode) {
        if (!selectedNode) {
            this.set('hightLightNodeName', '');
            this.renderGraph(this.hierarchyData, '');
            return;
        }
        const selectedNodeType = selectedNode.startsWith(NPU_PREFIX) ? 'NPU' : 'Bench';
        if (this.graphType !== 'Single' && selectedNodeType !== this.graphType) {
            return;
        } // 如果选中的节点类型和当前图类型不一致，则不处理
        const nodeName = selectedNode.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), ''); // 去掉前缀
        // 如果选中节点是当前图节点图中不存在，则展开其父节点，直到图中存在
        if (!this.hierarchyObject[nodeName]) {
            const nodeInfo = {
                nodeName,
                nodeType: this.graphType,
            };
            await this.changeNodeExpandState(nodeInfo);
        }
        // 是否需要居中
        let transform = this.getContainerTransform();
        if (this.needChangeNodeCenter) {
            transform = this.changeNodeCenter(nodeName);
        } else {
            this.set('needChangeNodeCenter', true);
        }
        // 高亮匹配的节点
        if (this.graphType !== 'Single') {
            const matchedNodes = this.hierarchyObject[nodeName]?.matchedNodeLink;
            this.hightLightMatchedNode?.(matchedNodes, this.graphType);
        }
        this.set('hightLightNodeName', selectedNode); // 设置高亮节点
        this.renderGraph(this.hierarchyData, selectedNode, transform);
    }

    // 父组件调用
    hightLightNode(nodeNames) {
        if (!Array.isArray(nodeNames) || isEmpty(nodeNames)) {
            this.renderGraph(this.hierarchyData, '');
            return;
        }
        const matchedNodeName = nodeNames[nodeNames.length - 1];
        this.set('hightLightNodeName', matchedNodeName);
        this.renderGraph(this.hierarchyData, matchedNodeName);
    }

    // 父组件调用
    fitScreen() {
        if (!this.container) {
            return;
        }
        changeGraphPosition(this.container, 0, 0, 1, 350);
        const newTransform = d3.zoomIdentity.translate(0, 0).scale(1);
        this.minimap?.zoom(newTransform);
        this.renderGraph(this.hierarchyData, this.hightLightNodeName);
    }

    // 总绑定事件方法，管理所有事件的绑定和解绑
    bindEventLisetener = () => {
        const cleanDragEvent = this.bindDragEvent(this.container);
        const cleanSelectedNodeEvent = this.bindSelectedNodeEvent(this.container);
        const cleanChangeNodeExpandStateEvent = this.bindChangeNodeExpandStateEvent(this.container);
        const cleanWheelEvent = this.bindWheelEvent();
        const cleanKeyboardEvent = this.bindKeyboardEvent(this.container);
        const cleanContextMenuEvent = this.bindContextMenuEvent();
        const cleanUpdateHierarchyDataEvent = this.bindUpdateHierarchyDataEvent();
        return () => {
            cleanDragEvent();
            cleanSelectedNodeEvent();
            cleanChangeNodeExpandStateEvent();
            cleanWheelEvent();
            cleanKeyboardEvent();
            cleanContextMenuEvent();
            cleanUpdateHierarchyDataEvent();
        };
    };

    bindUpdateHierarchyDataEvent() {
        const onUpdateHierarchyDataEvent = async () => {
            this.set('loading', true);
            const { success, data, error } = await this.useGraph.updateHierarchyData(this.graphType, this.selection);
            this.set('loading', false);
            if (success) {
                const hierarchyObject = data;
                const hierarchyData = Object.values(hierarchyObject);
                this.set('hierarchyData', hierarchyData);
                this.set('hierarchyObject', hierarchyObject);
                this.renderGraph(hierarchyData, this.hightLightNodeName);
                // // 选中节点
                const tempSelectedNode = this.selectedNode;
                this.set('selectedNode', '');
                this.set('selectedNode', tempSelectedNode);
            } else {
                Notification.show(`${this.t('upadte_map_data_fail')}：${error}`, {
                    position: 'middle',
                    duration: 3000,
                    theme: 'error',
                });
            }
        };
        document.addEventListener('updateHierarchyData', onUpdateHierarchyDataEvent);
        return () => {
            document.removeEventListener('updateHierarchyData', onUpdateHierarchyDataEvent);
        };
    }

    bindSelectedNodeEvent(container) {
        const onSelectNodeEvent = (event) => {
            event.preventDefault();
            const target: HTMLElement = event.target as HTMLElement;
            const selectedNode = target.getAttribute('name');
            if (selectedNode) {
                this.set('needChangeNodeCenter', false); // 点击不需要改变中心节点
                this.set('selectedNode', selectedNode);
            }
        };
        const throttleSelectNodeEvent = throttle(onSelectNodeEvent, 16);
        container.addEventListener('click', throttleSelectNodeEvent);
        return () => {
            container.removeEventListener('click', throttleSelectNodeEvent);
        };
    }

    bindContextMenuEvent() {
        const contextMenu = this.shadowRoot?.querySelector('#context-menu') as HTMLElement;
        const onContextMenuItemSelectedEvent = (event: ContextMenuItemSelectedEvent) => {
            event.preventDefault();
            const item = event.detail.value as ContextMenuItem;
            // 展开匹配节点
            if (item.type === EXPAND_MATCHED_NODE) {
                // 如果当前节点未匹配，则找到其相邻的匹配父节点
                const tempSelectedNode = this.selectedNode;
                this.findMatchedNodeName(tempSelectedNode);
                const { matchedNodeName, selectedNode } = this.findMatchedNodeName(tempSelectedNode);
                this.set('selectedNode', matchedNodeName); // 选中对应测节点就能触发展开和选中
                this.set('hightLightNodeName', selectedNode?.name)
                const transform = this.changeNodeCenter(selectedNode.name);
                this.renderGraph(this.hierarchyData, selectedNode.name, transform); // 更新selectedNode 会导致当前节点失去高亮显示
            }
            // 数据通信节点右键菜单
            if (item.type === DATA_COMMUNICATION) {
                const skipComunicaeRank = new CustomEvent('contextMenuTag-changed', {
                    detail: {
                        nodeName: item.nodeName,
                        rankId: item.rankId,
                    },
                    bubbles: true, // 允许事件冒泡
                    composed: true, // 允许跨 Shadow DOM 边界
                });
                this.dispatchEvent(skipComunicaeRank);
            }
        };
        const onContextmenuEvent = (event) => {
            event.preventDefault();
            const target = event.target as HTMLElement;
            // 图外点击，不显示右键菜单
            if (target.tagName.toLowerCase() !== 'rect' && target.tagName.toLowerCase() !== 'text') {
                event.stopPropagation();
            } else {
                const contextMenuItems: Array<ContextMenuItem> = [];
                if (this.graphType !== 'Single') {
                    contextMenuItems.push({
                        text: '展开对应侧节点',
                        type: EXPAND_MATCHED_NODE,
                    })
                }
                const selectedNode = target.getAttribute('name');
                const nodeName = selectedNode?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '') ?? '';
                const nodeData = this.hierarchyObject[nodeName];
                if (!isEmpty(nodeData?.matchedDistributed)) {
                    const matchedDistributed = nodeData?.matchedDistributed;
                    const communicationsType = matchedDistributed?.communications_type;
                    const nodeInfo = matchedDistributed?.nodes_info || {};
                    const rankIds = Object.keys(nodeInfo);
                    const children = rankIds.map((rankId) => {
                        const comunicateNode = nodeInfo[rankId];
                        const precision_index = comunicateNode?.[0];
                        const comunicateNodeName = comunicateNode?.[1];
                        const prefix = PREFIX_MAP[this.graphType];
                        return {
                            component: this.useGraph.createComponent(`rank${rankId}`, precision_index, this.colors),
                            nodeName: `${prefix}${comunicateNodeName}`,
                            rankId: Number(rankId),
                            type: DATA_COMMUNICATION,
                        };
                    });
                    const menuItem = {
                        text: DATA_COMMUNICATION_TYEPE[communicationsType],
                        children,
                    };
                    contextMenuItems.push(menuItem);
                }
                this.set('needChangeNodeCenter', false); // 点击不需要改变中心节点
                this.set('selectedNode', selectedNode);
                this.set('contextMenuItems', contextMenuItems);
            }
        };
        const throttleContextMenuItemSelectedEvent = throttle(onContextMenuItemSelectedEvent, 16);
        const throttleContextMenuEvent = throttle(onContextmenuEvent, 16);
        contextMenu.addEventListener('item-selected', throttleContextMenuItemSelectedEvent as any);
        this.graph?.addEventListener('contextmenu', throttleContextMenuEvent);
        return () => {
            contextMenu.removeEventListener('item-selected', throttleContextMenuItemSelectedEvent as any);
            this.graph?.removeEventListener('contextmenu', throttleContextMenuEvent);
        };
    }

    bindChangeNodeExpandStateEvent(container) {
        const onDoubleClickNodeEvent = async (event) => {
            event.preventDefault();
            let target;
            let selectedNode;
            //判断是点击展开，还是同步展开
            const isClickGraph = isEmpty(event.detail?.nodeName);
            if (isClickGraph) {
                target = event.target as HTMLElement;
                selectedNode = target.getAttribute('name');
            }
            else {
                selectedNode = event.detail.nodeName;
                selectedNode = selectedNode?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '')
                const graphType = event.detail.graphType;

                const orginNodeExpandState = event.detail.nodeExpandState;
                const targetNodeExpandState = this.hierarchyObject[selectedNode]?.expand;
                //也会触发当前侧图展开才操作，所以需要判断一下
                //保持展开状态同步,如果一侧展开，一侧为展开，则不触发对应测的展开或者收起的操作
                if (graphType === this.graphType || (orginNodeExpandState === targetNodeExpandState)) {
                    return;
                }
            }
            const nodeName = selectedNode?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), ''); // 去掉前缀
            if (nodeName === this.rootName) {
                return;
            }
            const nodeInfo = {
                nodeName,
                nodeType: this.graphType,
            };
            if (this.hierarchyObject[nodeInfo.nodeName || '']?.nodeType === NODE_TYPE.UNEXPAND_NODE) {
                return;
            }
            await this.changeNodeExpandState(nodeInfo);
            // 如果是点击展开，触发同步展开事件，通知展开对应节点
            if (isClickGraph && this.isSyncExpand && this.graphType !== "Single") {

                const findRes = this.findMatchedNodeName(nodeName);
                const changeMatchNodeExpandState = new CustomEvent('changeMatchNodeExpandState', {
                    detail: {
                        nodeName: findRes.matchedNodeName, // 通知通信图展开对应节点
                        nodeExpandState: findRes?.selectedNode?.expand,
                        graphType: this.graphType,
                    },
                    bubbles: true, // 允许事件冒泡
                    composed: true, // 允许跨 Shadow DOM 边界
                });
                this.dispatchEvent(changeMatchNodeExpandState);
            }
            const transform = this.changeNodeCenter(nodeName);
            this.set('needChangeNodeCenter', true);
            this.renderGraph(this.hierarchyData, this.hightLightNodeName, transform);
        };
        const onDoubleClickGraphEvent = (event) => {
            event.preventDefault();
        };
        const throttleDoubleClickNodeEvent = throttle(onDoubleClickNodeEvent, 16);
        document.addEventListener('changeMatchNodeExpandState', throttleDoubleClickNodeEvent);
        container.addEventListener('dblclick', throttleDoubleClickNodeEvent);
        this.graph?.addEventListener('dblclick', onDoubleClickGraphEvent); // 防止双击选中文本
        return () => {
            container.removeEventListener('dblclick', throttleDoubleClickNodeEvent);
            this.graph?.removeEventListener('dblclick', onDoubleClickGraphEvent);
            document.removeEventListener('changeMatchNodeExpandState', throttleDoubleClickNodeEvent);
        };
    }

    bindWheelEvent() {
        const onwheelEvent = (event) => {
            const transformStr = this.container?.getAttribute('transform') || '';
            const transform = parseTransform(transformStr);
            const delta = event.deltaY > 0 ? -MOVE_STEP : MOVE_STEP;
            transform.y = transform.y + delta;
            changeGraphPosition(this.container as HTMLElement, transform.x, transform.y, transform.scale);
            const newTransform = d3.zoomIdentity.translate(transform.x, transform.y).scale(transform.scale);
            this.minimap?.zoom(newTransform);
            this.renderGraph(this.hierarchyData, this.hightLightNodeName, {
                x: transform.x,
                y: transform.y,
                scale: transform.scale,
            });
        };
        const throttleWheelEvent = throttle(onwheelEvent, 16);
        this.graph?.addEventListener('wheel', throttleWheelEvent);
        return () => {
            this.graph?.removeEventListener('wheel', throttleWheelEvent);
        };
    }

    bindDragEvent(container) {
        let isDragging = false; // 是否正在拖拽
        let startX = 0; // 鼠标按下时的初始 X 坐标
        let startY = 0; // 鼠标按下时的初始 Y 坐标
        let initialTransform = { x: 0, y: 0, scale: 1.8 }; // 初始平移值
        const handleMouseDown = (event) => {
            event.preventDefault();
            isDragging = true;
            startX = event.clientX;
            startY = event.clientY;
            const transformStr = container.getAttribute('transform') || '';
            initialTransform = parseTransform(transformStr);
        };
        const handleMouseMove = (event) => {
            if (isDragging) {
                const dx = event.clientX - startX;
                const dy = event.clientY - startY;
                let newX = initialTransform.x + dx;
                let newY = initialTransform.y + dy;
                const scale = initialTransform.scale;
                changeGraphPosition(container, newX, newY, scale);
                const newTransform = d3.zoomIdentity.translate(newX, newY).scale(scale);
                this.minimap?.zoom(newTransform);
                this.renderGraph(this.hierarchyData, this.hightLightNodeName, { x: newX, y: newY, scale: scale });
            }
        };
        const handleMouseUp = () => {
            if (isDragging) {
                isDragging = false;
            }
        };
        const throttledMouseMove = throttle(handleMouseMove, 16);
        this.graph?.addEventListener('mousedown', handleMouseDown);
        document.addEventListener('mousemove', throttledMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        // 返回清理函数
        return () => {
            this.graph?.removeEventListener('mousedown', handleMouseDown);
            document.removeEventListener('mousemove', throttledMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }

    bindKeyboardEvent(container) {
        let isMouseInside = false;
        const handleMouseEnter = () => {
            isMouseInside = true;
        };

        const handleMouseLeave = () => {
            isMouseInside = false;
        };
        const handleKeyDown = (event) => {
            if (!isMouseInside) {
                return;
            }

            const transformStr = container.getAttribute('transform') || '';
            const transform = parseTransform(transformStr);

            switch (event.key) {
                case 'w':
                case 'W': // 放大
                    transform.scale += SCALE_STEP;
                    if (transform.scale > MAX_SCALE) {
                        return;
                    }
                    break;
                case 's':
                case 'S': // 缩小
                    transform.scale -= SCALE_STEP;
                    if (transform.scale < MIN_SCALE) {
                        return;
                    }
                    break;
                case 'a':
                case 'A': // 左移
                    transform.x -= MOVE_STEP;
                    break;
                case 'd':
                case 'D': // 右移
                    transform.x += MOVE_STEP;
                    break;
                default: {
                    return;
                } // 如果不是指定键，则退出
            }

            // 更新图形位置
            changeGraphPosition(this.container as HTMLElement, transform.x, transform.y, transform.scale);

            // 更新缩略图
            const newTransform = d3.zoomIdentity.translate(transform.x, transform.y).scale(transform.scale);
            this.minimap?.zoom(newTransform);
            this.renderGraph(this.hierarchyData, this.hightLightNodeName, {
                x: transform.x,
                y: transform.y,
                scale: transform.scale,
            });
        };

        // 使用 throttle 包装键盘事件处理函数
        const throttledHandleKeyDown = throttle(handleKeyDown, 16);

        this.graph?.addEventListener('mouseenter', handleMouseEnter);
        this.graph?.addEventListener('mouseleave', handleMouseLeave);
        document.addEventListener('keydown', throttledHandleKeyDown);

        // 返回清理函数
        return () => {
            this.graph?.removeEventListener('mouseenter', handleMouseEnter);
            this.graph?.removeEventListener('mouseleave', handleMouseLeave);
            document.removeEventListener('keydown', throttledHandleKeyDown);
        };
    }

    /**
     * 当前节点居中
     * @param nodeName 节点名称
     */
    changeNodeCenter(nodeName) {
        if (!nodeName) {
            return this.getContainerTransform();
        }
        const nodeNameReal = nodeName?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), ''); // 去掉前缀
        const selectedNode = this.hierarchyObject[nodeNameReal]; // 获取当前节点
        if (!selectedNode) {
            return this.getContainerTransform();
        }
        const transformStr = this.container?.getAttribute('transform') || '';
        const initialTransform = parseTransform(transformStr); // 保存初始位置
        const clientWidth = this.graph?.clientWidth || 0;
        const clientHeight = this.graph?.clientHeight || 0;
        const root = this.hierarchyObject[this.rootName];
        const newX = (clientWidth / 2) - ((root?.width * initialTransform.scale) / 2);
        const newY = (clientHeight / 2) - ((selectedNode?.y * initialTransform.scale) + 7.5) - 100;
        changeGraphPosition(this.container as HTMLElement, newX, newY, initialTransform.scale, 350);
        const newTransform = d3.zoomIdentity.translate(newX, newY).scale(initialTransform.scale);
        this.minimap?.zoom(newTransform);
        return { x: newX, y: newY, scale: initialTransform.scale };
    }

    /**
     * 展开节点树
     * @param nodeInfo 节点信息
     * @returns
     */
    async changeNodeExpandState(nodeInfo) {
        if (this.selection.type === DB_TYPE && (typeof (this.selection.rank) !== 'number' || typeof (this.selection.step) !== 'number')) {
            return {}
        }
        this.set('loading', true);
        const { success, data, error } = await this.useGraph.changeNodeExpandState(nodeInfo, this.selection);
        this.set('loading', false);
        if (success) {
            const hierarchyObject = data;
            const hierarchyData = Object.values(hierarchyObject);
            this.set('hierarchyData', hierarchyData);
            this.set('hierarchyObject', hierarchyObject);
        } else {
            Notification.show(`${this.t('expand_fail')}：${error}`, {
                position: 'middle',
                duration: 3000,
                theme: 'error',
            });
        }
        return { success, data, error };
    }
    /**
     * 寻找目标节点的匹配节点
     * @param tempSelectedNode 目标节点
     * @returns 
     * matchedNodeName: 匹配节点名称
     * selectedNode: 目标节点信息
     */

    findMatchedNodeName(tempSelectedNode) {
        let nodeName = tempSelectedNode?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '');
        let selectedNode = this.hierarchyObject[nodeName];
        while (isEmpty(selectedNode?.matchedNodeLink) && selectedNode?.parentNode) {
            nodeName = selectedNode.parentNode?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '');
            selectedNode = this.hierarchyObject[nodeName];
        }
        if (!isEmpty(selectedNode?.matchedNodeLink)) {
            let matchedNodeName = selectedNode.matchedNodeLink[selectedNode.matchedNodeLink.length - 1];
            const matchedPrefix = this.graphType === 'NPU' ? BENCH_PREFIX : NPU_PREFIX;
            matchedNodeName = matchedNodeName.startsWith(matchedPrefix)
                ? matchedNodeName
                : matchedPrefix + matchedNodeName; // 加上前缀
            return { matchedNodeName, selectedNode };
        } else {
            Notification.show(this.t('expand_fail_no_match_nodes'), {
                position: 'middle',
                duration: 3000,
                theme: 'error',
            });
            return { matchedNodeName: '', selectedNode: {} as HierarchyNodeType };
        }
    }
}
