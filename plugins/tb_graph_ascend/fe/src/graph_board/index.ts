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
import './components/legend/index';
import './components/hierarchy/index';
import { MIN_GRAPG_WIDTH } from '../common/constant';
import i18next from '../common/i18n';
@customElement('graph-board')
class MainGraph extends PolymerElement {
    static readonly template = html`
        <style>
            #container {
                width: 100%;
                height: 100%;
                background: white;
                display: flex;
            }
            #spliter {
                width: 6px;
                height: 100%;
                border-left: 2px dashed rgb(143, 143, 143);
                cursor: ew-resize;
            }
            #spliter:hover {
                border-left: 2px dashed hsl(214, 100%, 43%);
            }
            .graph-hierarchy {
                min-width: 320px;
            }
        </style>
        <scene-legend
            t="[[t]]"
        ></scene-legend>
        <div id="container">
            <graph-hierarchy
                id="NPU"
                class="graph-hierarchy"
                graph-type="[[mainGraphType]]"
                colors="{{colors}}"
                minimap-vis="[[minimapVis.npu]]"
                is-sync-expand="{{isSyncExpand}}"
                selection="[[selection]]"
                selected-node="{{selectedNode}}"
                is-overflow-filter="{{isOverflowFilter}}"
                hight-light-matched-node="[[hightLightMatchedNode]]"
            ></graph-hierarchy>
            <template is="dom-if" if="[[!isSingleGraph]]">
                <div id="spliter"></div>
                <graph-hierarchy
                    id="Bench"
                    class="graph-hierarchy"
                    graph-type="Bench"
                    colors="{{colors}}"
                    minimap-vis="[[minimapVis.bench]]"
                    is-sync-expand="{{isSyncExpand}}"
                    selection="[[selection]]"
                    selected-node="{{selectedNode}}"
                    is-overflow-filter="{{isOverflowFilter}}"
                    hight-light-matched-node="[[hightLightMatchedNode]]"
                ></graph-hierarchy>
            </template>
        </div>
    `;
    @property({ type: Object })
    t: Function = (key) => i18next.t(key);

    @property({ type: Boolean })
    isSingleGraph = false;

    @property({ type: Boolean, notify: true })
    selectedNode = '';

    @property({ type: String })
    mainGraphType = '';

    @observe('isSingleGraph')
    computeMainGraphType = () => {
        const mainGraphType = this.isSingleGraph ? 'Single' : 'NPU';
        this.set('mainGraphType', mainGraphType);
    };

    override ready() {
        super.ready();
        setTimeout(this.bindSpliterEvent, 16);
    }

    initGraphHierarchy = (selectedNode) => {
        const npuhGraph = this.shadowRoot?.querySelector('#NPU') as any;
        npuhGraph.initHhierarchy(selectedNode);
        if (!this.isSingleGraph) {
            const benchGraph = this.shadowRoot?.querySelector('#Bench') as any;
            benchGraph?.initHhierarchy(selectedNode);
        }
    };

    fitScreen = () => {
        const hierarchy = this.shadowRoot?.querySelectorAll('.graph-hierarchy') as any;
        if (!hierarchy) {
            return;
        }
        hierarchy.forEach((item) => {
            item.fitScreen();
        });
    };

    hightLightMatchedNode = (matchedNodes: Array<string>, graphType: string) => {
        if (graphType === 'NPU') {
            const benchGraph = this.shadowRoot?.querySelector('#Bench') as any;
            benchGraph?.hightLightNode?.(matchedNodes);
        } else if (graphType === 'Bench') {
            const npuhGraph = this.shadowRoot?.querySelector('#NPU') as any;
            npuhGraph?.hightLightNode?.(matchedNodes);
        }
    };

    bindSpliterEvent = () => {
        if (!this.shadowRoot || this.isSingleGraph) {
            return;
        }
        const spliter = this.shadowRoot.querySelector('#spliter');
        const container = this.shadowRoot.querySelector('#container');
        const NPU = this.shadowRoot.querySelector('#NPU') as HTMLElement;
        let isDragging = false;
        // 鼠标按下时开始拖动
        spliter?.addEventListener('mousedown', (e) => {
            e.preventDefault(); // 防止默认行为（如文本选中）
            isDragging = true;
            // 添加全局监听器以处理鼠标移动和释放
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });

        // 鼠标移动时调整宽度
        function onMouseMove(e) {
            if (!isDragging) {
                return;
            }
            // 获取鼠标相对于容器的位置
            const containerRect = container?.getBoundingClientRect();
            if (!containerRect) {
                return;
            }
            const newWidth = e.clientX - containerRect.left;
            // 限制最小宽度
            if (newWidth > MIN_GRAPG_WIDTH && newWidth < containerRect.width - MIN_GRAPG_WIDTH) {
                NPU.style.flex = `0 0 ${newWidth}px`; // 设置固定宽度
            }
        }
        // 鼠标松开时停止拖动
        function onMouseUp() {
            isDragging = false;
            // 移除全局监听器
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        }
    };
}
