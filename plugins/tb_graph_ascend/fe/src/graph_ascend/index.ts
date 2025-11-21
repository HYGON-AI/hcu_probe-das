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

import { customElement, observe, property } from '@polymer/decorators';
import { html, PolymerElement } from '@polymer/polymer';
import { LegacyElementMixin } from '../polymer/legacy_element_mixin';
import { LoadGraphFileInfoListType } from './type';
import useGraphAscend from './useGraphAscend';
import { formatBytes, safeJSONParse } from '../utils';
import { isEmpty } from 'lodash';
import '../graph_board/index';
import '../graph_info_board/index';
import '../graph_controls_board/index';
import '../common/graph-board-layout';
import '@vaadin/confirm-dialog'
import { Notification } from '@vaadin/notification';
import request from '../utils/request';
import type { SelectedItemType, SelectionType, ProgressType, GraphConfigType, GraphAllNodeType, NodeListType, UnmatchedNodeType } from './type';
import i18next from '../common/i18n';

@customElement('graph-ascend')
class TfGraphDashboard extends LegacyElementMixin(PolymerElement) {
    static readonly template = html`
        <template is="dom-if" if="[[!progressData.done]]">
            <div class="progress-bar">
                <label class="text-secondary">[[progressData.title]] [[progressData.progress]] %</label>
                <vaadin-progress-bar value="[[progressData.progressValue]]" theme="success"></vaadin-progress-bar>
                <span class="text-secondary text-xs">[[progressData.info]]</span>
            </div>
        </template>
        <graph-board-layout>
            <graph-controls-board
                id="controls"
                class="sidebar"
                slot="sidebar"
                meta-dir="[[metaDir]]"
                selection="{{selection}}"
                selected-node="{{selectedNode}}"
                colors="{{colors}}"
                colorset="[[colorset]]"
                overflowcheck="[[overflowcheck]]"
                steps="[[steps]]"
                ranks="[[ranks]]"
                microsteps="[[microsteps]]"
                npu-match-nodes="[[npuMatchNodes]]"
                bench-match-nodes="[[benchMatchNodes]]"
                matched-config-files="[[matchedConfigFiles]]"
                nodelist="[[nodelist]]"
                unmatched="[[unmatched]]"
                minimap-vis="{{minimapVis}}"
                is-sync-expand="{{isSyncExpand}}"
                is-single-graph="{{isSingleGraph}}"
                task="[[task]]"
                is-overflow-filter="{{isOverflowFilter}}"
                on-fit-tap="onFitTap"
                load-all-node-list="{{loadGraphAllNodeList}}"
                need-load-all-node-list="{{needLoadAllNodeList}}"
            ></graph-controls-board>
            <div class="center" slot="center">
                <div class="graph-board-wrapper">
                    <graph-board
                        t="[[t]]"
                        id="graph-board"
                        colors="{{colors}}"
                        selection="[[selection]]"
                        selected-node="{{selectedNode}}"
                        highlighted-node="{{_highlightedNode}}"
                        minimap-vis="[[minimapVis]]"
                        is-sync-expand="{{isSyncExpand}}"
                        is-single-graph="[[isSingleGraph]]"
                        is-overflow-filter="{{isOverflowFilter}}"
                    ></graph-board>
                </div>
                <graph-info-board
                    id="graph-info-board"
                    selected-node="{{selectedNode}}"
                    tooltips="[[tooltips]]"
                    selection="[[selection]]"
                    is-single-graph="[[isSingleGraph]]"
                >
                </graph-info-board>
            </div>
        </graph-board-layout>
        <vaadin-confirm-dialog
          id="safe-dialog"
          header="您尝试访问的文件或路径未通过系统的安全校验"
          opened="[[safeDialogOpened]]"
        >
        <div class='file-list-error'>
            <p> <vaadin-icon id="safe-warning" icon="vaadin:warning"></vaadin-icon>请知悉以下风险：</p>
            <div>非授权路径访问可能存在信息泄露和文件内容篡改。 文件过大或格式异常，可能导致性能问题或服务中断。路径中存在软链接或权限不当，可能存在越权访问和数据篡改风险。</div>
            <P>继续操作将由您自行承担相关后果。如非明确知晓风险，请取消操作并联系管理员处理。</p>
            <div class="error-info">
                <template is="dom-repeat" items="{{fileListError}}">
                    <div><span class='file-path'>[[item.run]]/[[item.tag]]：</span>[[item.info]]</div>
                </template>
            </div>
        </div>
        </vaadin-confirm-dialog>
        <style>
            :host /deep/ {
                font-family: 'Roboto', sans-serif;
                position: relative;
            }

            .sidebar {
                display: flex;
                height: 100%;
            }

            .center {
                height: 100%;
                display: flex;
                flex-direction: column;
            }

            vaadin-progress-bar::part(value) {
                background-color: var(--progress-background-color, rgb(21, 132, 67));
                transition: width 0.2s linear 0;
            }
            .progress-bar {
                width: 100%;
                height: 100%;
                z-index: 999;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background-color: rgba(170, 170, 170, 0.74);
                color: var(--progress-color, #757575);
                position: absolute;
            }
            .graph-board-wrapper {
                height: 80%;
                position: relative;
            }
            vaadin-progress-bar {
                width: 80%;
                height: 14px;
            }
        </style>
    `;
    @property({ type: Object })
    t: Function = (key) => i18next.t(key);

    @property({ type: Object })
    metaDir: Record<string, string> = {};

    @property({ type: Object, notify: true })
    selection: SelectionType | null = null;

    @property({ type: Object, notify: true })
    nodelist: NodeListType = {} as NodeListType;

    @property({ type: Object, notify: true })
    unmatched: UnmatchedNodeType = {} as UnmatchedNodeType;

    @property({ type: Object, notify: true })
    matchedlist: any;

    @property({ type: String, notify: true })
    selectedNode: string = '';

    @property({ type: String, notify: true })
    jumpToNode: string = '';

    @property({ type: Object, notify: true })
    colors: any;

    @property({ type: Boolean, notify: true })
    isOverflowFilter: boolean = false;

    @property({ type: Object })
    progressData: ProgressType = { progress: 0, progressValue: 0, done: false };

    @property({ type: Boolean })
    isSingleGraph: boolean = false;

    @property({ type: Array })
    microsteps: number[] = [];

    @property({ type: Array })
    steps: Array<SelectedItemType> = [];

    @property({ type: Array })
    ranks: Array<SelectedItemType> = [];

    @property({ type: Array })
    overflowcheck: boolean = false;

    @property({ type: Object })
    tooltips: object = {};

    @property({ type: Object })
    colorset: object = {};

    @property({ type: Object })
    npuMatchNodes: object = {};

    @property({ type: Object })
    benchMatchNodes: object = {};

    @property({ type: Object })
    matchedConfigFiles: string[] = [];

    @property({ type: Object })
    task: string = '';

    @property({ type: Boolean })
    safeDialogOpened: boolean = false;

    @property({ type: Array })
    fileListError: Array<LoadGraphFileInfoListType['error']> = [];

    @property({ type: Object })
    needLoadAllNodeList: boolean = true;

    private currentSelection: SelectionType | null = null;
    private useGraphAscend = useGraphAscend();
    private eventSource: EventSource | null = null;

    @observe('selection')
    updateGraphData = () => {
        if (!this.selection?.run || !this.selection?.tag) {
            return;
        }
        const isFileChange = this.currentSelection?.run !== this.selection?.run || this.currentSelection?.tag !== this.selection?.tag;
        const isDBChange = this.currentSelection?.rank !== this.selection?.rank || this.currentSelection?.step !== this.selection?.step;
        if (isFileChange) {
            switch (this.selection?.type) {
                case 'json':
                    this.loadJSONGraphData(this.selection);
                    break;
                case 'db':
                    this.loadDBGraphData(this.selection, true);
                    break;
                default:
                    break;
            }
        }
        else if (isDBChange) {
            this.loadDBGraphData(this.selection, false);
        }
        else if (this.currentSelection?.microStep !== this.selection?.microStep) {
            this.initGraphBoard(); // 只改变microsteps时，不重新加载图数据
        }
        else {
            return
        }
        this.set('needLoadAllNodeList', true);
        this.currentSelection = this.selection;
    };

    override async ready(): Promise<void> {
        super.ready();
        const { data, error } = await this.useGraphAscend.loadGraphFileInfoList();
        const safeDialog = this.shadowRoot?.querySelector('#safe-dialog') as HTMLElement;
        if (!isEmpty(error)) {
            this.set('safeDialogOpened', true);
            this.set('fileListError', error);
        }
        this.set('metaDir', data);
        document.addEventListener(
            'contextMenuTag-changed',
            (event: any) => this.set('jumpToNode', event.detail?.nodeName),
            { passive: true },
        );
    }

    loadDBGraphData = async (metaData: SelectionType, isInitDB: boolean = false) => {
        if (isInitDB) {
            this.progreesLoading(this.t('initializing_database'), this.t('please_wait'), { progress: 10, progressValue: 10, done: false });
            await request({ url: 'loadGraphData', method: 'GET', params: metaData });
            await this.loadGraphConfig(metaData)
        }
        this.progreesLoading(this.t('initializing_graph'), this.t('please_wait'), { progress: 90, progressValue: 90, done: false });
        this.initGraphBoard(); // 先读取配置，再加载图,顺序很重要
        this.progreesLoading(this.t('initialization_complete'), this.t('please_wait'), { progress: 100, progressValue: 100, done: true });
    }


    loadJSONGraphData = async (metaData: SelectionType) => {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        this.eventSource = new EventSource(`loadGraphData?run=${metaData.run}&tag=${metaData.tag}&type=${metaData.type}&lang=${metaData.lang}`);
        this.eventSource.onmessage = async (e) => {
            const data = safeJSONParse(e.data);
            if (data?.error) {
                this.progreesError(this.t('graph_initialization_fail'), data.error);
            }
            if (data?.status === 'reading') {
                this.progressReading(this.t('reading_file'), data);
            }
            if (data?.status === 'loading') {
                if (data.done) {
                    this.eventSource?.close();
                    this.eventSource = null;
                    try {
                        await this.loadGraphConfig(metaData)
                        this.initGraphBoard(); // 先读取配置，再加载图,顺序很重要
                        this.progreesLoading(this.t('initialization_complete'), this.t('please_wait'), data);
                    } catch (error) {
                        this.progreesError(this.t('graph_initialization_fail'), error);
                    }
                } else {
                    this.progreesLoading(this.t('parsing_file'), this.t('initializing_model'), data);
                }
            }
        };

        this.eventSource.onerror = (e) => {
            if (!this.progressData || !this.progressData.done) {
                this.progreesError(this.t('loadig_fail'), this.t('file_format_wrong'));
            }
            this.eventSource?.close();
        };
    }

    loadGraphConfig = async (metaData) => {
        const { success, data, error } = await this.useGraphAscend.loadGraphConfig(metaData);
        const config = data as GraphConfigType;
        const selection = { ...this.selection };
        if (success) {
            this.set('colors', config.colors);
            this.set('tooltips', safeJSONParse(config.tooltips));
            this.set('overflowcheck', config.overflowCheck);
            this.set('colorset', Object.entries(config.colors || {}));
            this.set('isSingleGraph', config.isSingleGraph);
            this.set('task', config.task);
            this.set('matchedConfigFiles', ['未选择', ...config.matchedConfigFiles]);
            const microstepsCount = Number(config.microSteps);
            const ranks = config.ranks || [0];
            const steps = config.steps || [0];
            if (microstepsCount) {
                const microstepsArray = Array.from({ length: microstepsCount + 1 }, (_, index) => ({
                    label: index === 0 ? 'ALL' : String(index - 1),
                    value: index - 1,
                }));
                this.set('microsteps', microstepsArray);
            }
            if (ranks.length > 0) {
                const ranksArray = ranks.map((rank) => ({
                    label: rank,
                    value: rank,
                }))
                this.set('ranks', ranksArray);
                selection.rank = ranks[0]
            }
            if (steps.length > 0) {
                const stepsArray = steps.map((step) => ({
                    label: step,
                    value: step,
                }))
                this.set('steps', stepsArray);
                selection.step = steps[0]
            }
            this.set('selection', selection);
        } else {
            Notification.show(`${this.t('load_image_config_fail')}:${error}`, {
                position: 'middle',
                duration: 2000,
                theme: 'error',
            });
        }
    };

    loadGraphAllNodeList = async (metaData: SelectionType) => {
        const { success, data, error } = await this.useGraphAscend.loadGraphAllNodeList(metaData);
        const allNodeList = data as GraphAllNodeType;
        if (success) {
            const nodelist = {} as NodeListType;
            const unmatched = {} as UnmatchedNodeType;
            if (this.isSingleGraph) {
                nodelist.npu = allNodeList?.npuNodeList;
            } else {
                nodelist.npu = allNodeList?.npuNodeList;
                nodelist.bench = allNodeList?.benchNodeList;
                unmatched.npuNodeList = allNodeList?.npuUnMatchNodes;
                unmatched.benchNodeList = allNodeList?.benchUnMatchNodes;
            }
            this.set('npuMatchNodes', allNodeList?.npuMatchNodes);
            this.set('benchMatchNodes', allNodeList?.benchMatchNodes);
            this.set('nodelist', nodelist);
            this.set('unmatched', unmatched);
        } else {
            Notification.show(`${this.t('load_graph_node_list_fail')}:${error}`, {
                position: 'middle',
                duration: 2000,
                theme: 'error',
            });
        }
    };

    initGraphBoard = () => {
        (this.shadowRoot?.querySelector('#graph-board') as any)?.initGraphHierarchy(this.jumpToNode);
        if (this.jumpToNode) {
            this.set('selectedNode', this.jumpToNode);
            this.set('jumpToNode', '');
        }
    };

    onFitTap(): void {
        (this.shadowRoot?.querySelector('#graph-board') as any).fitScreen();
    }

    progressReading = (title, data) => {
        data.progressValue = data.done ? 1 : data.progress / 100.0;
        data.size = formatBytes(data.size);
        data.read = formatBytes(data.read);
        data.title = title;
        data.info = `文件大小: ${data.size}, 已读取: ${data.read}`;
        this.set('progressData', data);
    };

    progreesLoading = (title, info, progressData) => {
        const data = {
            ...progressData,
            title,
            info,
        };
        data.progressValue = progressData.done ? 1 : progressData.progress / 100.0;
        this.set('progressData', data);
    };

    progreesError = (title, info) => {
        const data = {
            ...this.progressData,
            title,
            info,
        };
        this.updateStyles({
            '--progress-background-color': 'red',
            '--progress-color': 'red',
        });
        this.set('progressData', data);
    };
}
