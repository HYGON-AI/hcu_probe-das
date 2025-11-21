/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import { customElement } from '@polymer/decorators';
import { html, PolymerElement } from '@polymer/polymer';
import { DarkModeMixin } from '../../polymer/dark_mode_mixin';
import './tensorboardColor';

@customElement('graph-board-layout')
class TfDashboardLayout extends DarkModeMixin(PolymerElement) {
  static readonly template = html`
  <div class='sidebar-container'>
    <div id="sidebar">
      <slot name="sidebar"></slot>
    </div>
    <div class='sidebar-toggle-container'>
      <div id='sidebar-toggle' class='sidebar-toggle-expand sidebar-toggle-fold' on-click="_toggleSidebar"></div>
    </div>
  </div>
  </div>
    <div id="center">
      <slot name="center" class="scollbar"></slot>
    </div>
    <style include="scrollbar-style"></style>
    <style>
      :host {
        background-color: #f5f5f5;
        display: flex;
        flex-direction: row;
        height: 100%;
      }

      :host(.dark-mode) {
        background-color: var(--secondary-background-color);
      }

      .sidebar-container{
        display: flex;
        justify-content: flex-start;
      }
      .scrollbar::-webkit-scrollbar-track {
      visibility: hidden;
    }

      .scrollbar::-webkit-scrollbar {
        width: 10px;
      }

      .scrollbar::-webkit-scrollbar-thumb {
        border-radius: 10px;
        -webkit-box-shadow: inset 0 0 2px rgba(0, 0, 0, 0.3);
        background-color: var(--paper-grey-500);
        color: var(--paper-grey-900);
      }
      .scrollbar {
        box-sizing: border-box;
      }

      #sidebar {
        height: 100%;
        width: 350px;
        max-width: var(--tf-dashboard-layout-sidebar-max-width, 350px);
        min-width: var(--tf-dashboard-layout-sidebar-min-width, 270px);
        overflow-y: auto;
        text-overflow: ellipsis;
      }
      .sider-hidden{
        display: none;
      }

      .sidebar-toggle-container{
        width: 16px;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: rgb(246, 246, 246);
        border-right: 1px solid #ddd;
      }
      #sidebar-toggle{
        width: 16px;  
        height: 100px;
        background-color: rgb(233, 232, 232);
      
        cursor: pointer;
        display: flex;
        justify-content: center;
        align-items: center;
        transition: background-color 0.3s ease;
      }
      /* 切换按钮的箭头 */
      #sidebar-toggle::before {
        content: '';
        display: block;
        width: 0;
        height: 0;
        margin:0 12px;
        border-style: solid;
        transition: transform 0.3s ease;
      }
      /* 按钮悬停效果 */
      #sidebar-toggle:hover {
        background-color:rgb(219, 218, 218); /* 悬停变深灰 */
      }
      .sidebar-toggle-expand{
        clip-path: polygon(0% 100%, 100% 90%, 100% 10%, 0% 0%);/* 切换按钮的形状 */
      }
      .sidebar-toggle-fold{
        clip-path: polygon(0% 90%, 100% 100%, 100% 0%, 0% 10%);/* 切换按钮的形状 */

      }
      /* 收起状态箭头（向右） */
      .sidebar-toggle-expand::before {
        border-width: 6px 0 6px 8px;
        border-color: transparent transparent transparent #333;
      }
       /* 展开状态箭头（向左） */
      .sidebar-toggle-fold::before {
        border-width: 6px 8px 6px 0;
        border-color: transparent #333 transparent transparent;
      }

      #center {
        flex-grow: 1;
        flex-shrink: 1;
        height: 100%;
        overflow: hidden;
        background:white;
      }

      ::slotted([slot='center']) {
        contain: strict;
        height: 100%;
        overflow-x: hidden;
        overflow-y: auto;
        width: 100%;
        will-change: transform;
      }

      .tf-graph-dashboard #center {
        background: #fff;
      }
    </style>
  `;

  _toggleSidebar(): void {
    // 通过 ID 获取元素并隐藏
    const sidebar = this.shadowRoot?.querySelector('#sidebar');
    const sidebarToggle = this.shadowRoot?.querySelector('#sidebar-toggle');
    // 检查并切换 display 样式
    if (sidebar) {
      sidebar?.classList.toggle('sider-hidden'); // 改为显示
      sidebarToggle?.classList.toggle('sidebar-toggle-fold'); // 改变箭头方向
    }
  }
}
