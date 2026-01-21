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

import { PolymerElement, html } from '@polymer/polymer';
import { customElement, property, observe } from '@polymer/decorators';

@customElement('tf-resize-height')
class ResizableTabsheet extends PolymerElement {
  static readonly template = html`
    <style>
      :host {
        display: block;
        font-family: Arial, sans-serif;
      }

      .tabsheet {
        width: 100%;
        height: var(--tabsheet-height, 300px); /* 默认高度 */
        background-color: rgb(255, 255, 255);
        will-change: transform;
      }

      .resize-handle {
        height: 2px;
        width: 100%;
        cursor: ns-resize;
        bottom: 2px;
        z-index: 999;
        position: relative;
        background-color: rgb(141, 141, 141);
      }

      .resize-handle:hover {
        background-color: hsl(214, 100%, 43%);
        height: 4px;
     
      .resize-handle.hidden {
        display: none;
        cursor: default;
      }
    </style>

    <div class$="[[_getResizeHandleClass(showHandle)]]" id="resizeHandle"></div>
    <div class="tabsheet" id="tabsheet">
      <slot></slot>
    </div>
  `;

  @property({
    type: Number,
    notify: true,
  })
  height: number = 300;

  @property({ type: Boolean })
  showHandle: boolean = true;

  _resize: (event: MouseEvent) => void = () => {};
  _stopResize: (this: Document, ev: MouseEvent) => any = () => {};

@observe('height')
_updateHeight(newHeight): void {
  // 从字符串中提取数字部分
  const heightMatch = String(newHeight).match(/\d+/);
  const heightValue = heightMatch ? heightMatch[0] : '300'; // 默认300px
  
  this.updateStyles({ '--tabsheet-height': `${heightValue}px` });
}

  @observe('showHandle')
  _handleShowHandleChange(): void {
    // 当 showHandle 变化时，重新初始化事件监听
    this._initResizeHandle();
  }

  override ready(): void {
    super.ready();
    this._initResizeHandle();
  }

  _initResizeHandle(): void {
    if (!this.showHandle) {
      return;
    }
    const tabsheet = this.$.tabsheet as HTMLElement;
    const resizeHandle = this.$.resizeHandle as HTMLElement;

    let isResizing = false;
    let startY = 0;
    let startHeight = 0;

    // 开始拖拽
    resizeHandle.addEventListener('mousedown', (event: MouseEvent) => {
      if (!this.showHandle) return;
      isResizing = true;
      startY = event.clientY;
      startHeight = tabsheet.offsetHeight;
      document.body.style.cursor = 'ns-resize';
      document.addEventListener('mousemove', this._resize);
      document.addEventListener('mouseup', this._stopResize);
    });

    // 拖拽过程
    this._resize = (event): void => {
      if (!isResizing || !this.showHandle) {
        return;
      }
      const deltaY = startY - event.clientY; // 向上拖拽为正
      this.set('height', Math.max(0, startHeight + deltaY)); // 更新高度
    };

    // 停止拖拽
    this._stopResize = (): void => {
      isResizing = false;
      document.body.style.cursor = '';
      document.removeEventListener('mousemove', this._resize);
      document.removeEventListener('mouseup', this._stopResize);
    };
  }

  _getResizeHandleClass(showHandle) {
    return showHandle ? 'resize-handle' : 'resize-handle hidden';
  }
}
