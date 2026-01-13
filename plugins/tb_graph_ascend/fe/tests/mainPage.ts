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
import { test as baseTest, Locator, Page } from '@playwright/test';

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

class MainPage {
  readonly page: Page;
  readonly mainArea: Locator;
  readonly dirSelector: Locator;
  readonly npuGraph: Locator;
  readonly benchGraph: Locator;
  readonly splitter: Locator;
  readonly syncCheckBox: Locator;
  readonly npuMinimap: Locator;
  readonly benchMinimap: Locator;

  constructor(page: Page) {
    this.page = page;
    this.mainArea = page.locator('graph-ascend');
    this.dirSelector = page.getByRole('combobox', { name: '目录' });
    this.npuGraph = page.locator('#NPU');
    this.benchGraph = page.locator('#Bench');
    this.splitter = page.locator('#spliter');
    this.syncCheckBox = page.getByRole('checkbox', { name: '是否同步展开对应侧节点' });
    this.npuMinimap = this.npuGraph.locator('#minimap');
    this.benchMinimap = this.benchGraph.locator('#minimap');
  }

  async getBoundingBoxes(): Promise<{ npuArea: BoundingBox, benchArea: BoundingBox }> {
    const npuArea = await this.npuGraph.boundingBox();
    const benchArea = await this.benchGraph.boundingBox();
    if (!npuArea || !benchArea) {
      throw new Error('Test failed because the graph area was not rendered correctly.');
    }
    return { npuArea, benchArea };
  }
}

export const test = baseTest.extend<{ mainPage: MainPage }>({
  mainPage: async ({ page }, use) => {
    const mainPage = new MainPage(page);
    await use(mainPage);
  }
});