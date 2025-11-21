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
import { customElement, property } from '@polymer/decorators';
import i18next from '../../../common/i18n';
@customElement('scene-legend')
class Legend extends PolymerElement {
  static readonly template = html`
      <style>
        :host {
          --legend-border-color:rgb(99, 99, 99);
          --legend-fill-color: rgb(230, 230, 230);
        }
        .legend {
          display: flex;
          justify-content: center;
          align-items: center;
          margin-top: 10px;
          background: #fff;
          height: 40px;
        }
        .legend-item {
          margin-right: 10px;
          display: flex;
          align-items: center;
        }
        .legend-item-value {
          margin-left: 5px;
          font-size: 12px;
        }
        .legend-clarifier {
          color: #266236;
          cursor: pointer;
          height: 16px;
          width: 16px;
          margin-left: 4px;
          display: inline-block;
          text-decoration: underline;
          background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDIwMDAgMjAwMCI+PHBhdGggZmlsbD0iIzc5Nzk3OSIgZD0iTTk3MSAxMzY4cTE4IDAgMzEgMTIgMTIgMTMgMTIgMzF2MTQzcTAgMTgtMTIgMzEtMTMgMTItMzEgMTJIODI3cS0xNyAwLTMwLTEyLTEzLTEzLTEzLTMxdi0xNDNxMC0xOCAxMy0zMSAxMy0xMiAzMC0xMmgxNDR6bTEyMi03NDJxODYgNDMgMTM4IDExNSA1MiA3MSA1MiAxNjEgMCA5Ny01NyAxNjUtMzYgNDAtMTE5IDg3LTQzIDI1LTU3IDM5LTI1IDE4LTI1IDQzdjU5SDc3NHYtNzBxMC04MiA1Ny0xNDAgMzItMzIgMTA4LTc1bDctNHE1NC0zMiA3NS01MCAyNS0yNSAyNS01NyAwLTQzLTQ0LTcyLTQ1LTI5LTEwMS0yOXQtOTUgMjVxLTMyIDIyLTc5IDgzLTExIDExLTI3IDE0LTE2IDQtMzAtN2wtMTAxLTc1cS0xNC0xMS0xNi0yOXQ1LTMycTY4LTk3IDE1NS0xNDYgODYtNDggMjA4LTQ4IDg2IDAgMTcyIDQzem01NzYgMjlxMTIxIDIwNCAxMjEgNDQ1IDAgMjQwLTEyMSA0NDUtMTIwIDIwNC0zMjUgMzI1LTIwNCAxMjAtNDQ1IDEyMC0yNDAgMC00NDUtMTIwLTIwNC0xMjEtMzI1LTMyNVE5IDEzNDAgOSAxMTAwcTAtMjQxIDEyMC00NDUgMTIxLTIwNSAzMjUtMzI1IDIwNS0xMjEgNDQ1LTEyMSAyNDEgMCA0NDUgMTIxIDIwNSAxMjAgMzI1IDMyNXptLTE0OSA4MDRxOTctMTY1IDk3LTM1OXQtOTctMzU5cS05Ny0xNjUtMjYyLTI2MnQtMzU5LTk3cS0xOTQgMC0zNTkgOTdUMjc4IDc0MXEtOTcgMTY1LTk3IDM1OXQ5NyAzNTlxOTcgMTY1IDI2MiAyNjJ0MzU5IDk3cTE5NCAwIDM1OS05N3QyNjItMjYyeiIvPjwvc3ZnPg==');
        }

        .legend-clarifier paper-tooltip {
          width: 150px;
          display: flex;
          align-items: center;
        }

        .custom-tooltip {
          font-size: 14px;
        }

        .module-rect {
          width: 46px;
          height: 16px;
          border-radius: 6px;
          border: 1px solid var(--legend-border-color);
          background: var(--legend-fill-color)
        }

        .unexpand-nodes{
          width: 46px;
          height: 16px;
          border-radius: 50%;
          border: 1px solid var(--legend-border-color);
          background: var(--legend-fill-color)
        }

        .api-list {
          width: 50px;
          height: 24px;
        }
        .api-list rect {
          fill:rgb(255, 255, 255); /* 内部无填充 */
          stroke: rgb(99, 99, 99); /* 边框颜色 */
          stroke-width: 1; /* 边框宽度 */
          stroke-dasharray: 10 1; /* 虚线样式 */
        }

        .fusion-node{
          width: 50px;
          height: 24px;
        }
        .fusion-node rect {
          fill: rgb(255, 255, 255); /* 内部无填充 */
          stroke: rgb(99, 99, 99); /* 边框颜色 */
          stroke-width: 1; /* 边框宽度 */
          stroke-dasharray: 2 1; /* 虚线样式 */
        }
      </style>
      <div class="legend">
        <div class="legend-item">
          <svg  class='module-rect'></svg>
          <span class="legend-item-value">[[t('module_or_operators')]]</span>
        </div>
        <div class="legend-item">
          <svg  class='unexpand-nodes'></svg>
          <span class="legend-item-value">[[t('unexpanded_nodes')]]</span>
          <div class="legend-item-value legend-clarifier">
            <paper-tooltip fit-to-visible-bounds animation-delay="0" position="right" offset="0">
              <div class="custom-tooltip">
                [[t('unexpanded_nodes_tooltip')]]
              </div>
            </paper-tooltip>
          </div>
        </div>
        <div class="legend-item">
          <svg class='api-list'>
              <rect width="46" height="18" rx='5' ry='5' x='2' y='4' />
          </svg>
          <span class="legend-item-value">[[t('api_list')]]</span>
          <div class="legend-item-value legend-clarifier">
            <paper-tooltip animation-delay="0" position="right" offset="0">
               <div class="custom-tooltip">[[t('api_list_tooltip')]]</div>
            </paper-tooltip>
          </div>
        </div>
        <div class="legend-item">
          <svg class='fusion-node'>
               <rect width="46" height="18" rx='5' ry='5' x='2' y='4' />
          </svg>
          <span class="legend-item-value">[[t('multi_collection')]]</span>
          <div class="legend-item-value legend-clarifier">
            <paper-tooltip animation-delay="0" position="right" offset="0">
                <div class="custom-tooltip">[[t('multi_collection_tooltip')]]</div>
            </paper-tooltip>
          </div>
        </div>
      </div>
    `;
    @property({ type: Object })
    t: Function = (key) => i18next.t(key);

    constructor() {
      super();
      this.setupLanguageListener();
    }

    setupLanguageListener() {
      i18next.on('languageChanged', () => {
        //更新语言后重新渲染
        const t = this.t;
        this.set('t', null);
        this.set('t', t);
      });
    }
}

