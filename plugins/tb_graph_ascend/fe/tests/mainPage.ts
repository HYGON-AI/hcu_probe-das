/* -------------------------------------------------------------------------
 Copyright (c) 2025, Huawei Technologies.
 All rights reserved.

 Licensed under the Apache License, Version 2.0  (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
--------------------------------------------------------------------------------------------*/
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