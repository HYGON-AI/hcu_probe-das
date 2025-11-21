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
import { expect, Page } from '@playwright/test';
import { test } from './mainPage';

const MAX_DIFF_PIXELS = 100;
// 按数次键，设置间隔以提高测试稳定性
const pressKeyMultipleTimes = async (page: Page, key: string, times: number) => {
  for (let i = 0; i < times; i++) {
    await page.keyboard.press(key);
    await page.waitForTimeout(100);
  }
};
// 滚轮操作，拆解动作并设置间隔以提高测试稳定性
const mouseWheelMultipleTimes = async (page: Page, offsetX: number, offsetY: number, times: number) => {
  for (let i = 0; i < times; i++) {
    await page.mouse.wheel(offsetX, offsetY);
    await page.waitForTimeout(100);
  }
};

// 测试图中节点操作用例集
test.describe('MainGraphTest', () => {
  test.beforeEach(async ({ page, mainPage }) => {
    const allParsedPromise = page.waitForResponse(response =>
      response.url().includes('/loadGraphConfigInfo') && response.status() === 200
    );
    await page.goto('/');
    await allParsedPromise;
    await mainPage.dirSelector.click();
    await page.getByRole('option', { name: 'dbCompare' }).click();
  });

  // 测试信息栏相关功能
  test('test_select_node', async ({ page, mainPage }) => {
    const { mainArea, npuGraph } = mainPage;
    await npuGraph.getByText('Module.c…Module.conv1.Conv2d.forward.0').click();
    await expect(page.locator('.node-info-item.selected-node')).toHaveText('标杆节点：Module.conv1.Conv2d.forward.0');
    await expect(mainArea).toHaveScreenshot('nodeSelected.png', { maxDiffPixels: MAX_DIFF_PIXELS });
    await page.getByRole('tab', { name: '节点信息' }).click();
    await page.getByRole('textbox').first().hover();
    const copyBtn = page.getByRole('button', { name: 'copy' });
    await expect(copyBtn).toBeVisible();
    await copyBtn.click();
    await page.waitForTimeout(500);
    await expect(mainArea).toHaveScreenshot('copySuccess.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });

  // 测试节点展开/收起以及对应侧关联节点行为
  test('test_node_expand_and_collapse', async ({ mainPage }) => {
    const { mainArea, npuGraph, benchGraph, syncCheckBox } = mainPage;
    const npuNode = npuGraph.getByText('Module.layer1.Sequential.forward.0Module.layer1.Sequential.forward.0');
    const benchNode = benchGraph.getByText('Module.layer1.Sequential.forward.0Module.layer1.Sequential.forward.0');
    const npuSubNode = npuGraph.getByText('layer1.0.BasicBlock.forward.0layer1.0.BasicBlock.forward.0');
    const benchSubNode = benchGraph.getByText('layer1.0.BasicBlock.forward.0layer1.0.BasicBlock.forward.0');
    // 默认勾选同步展开对应侧节点，任意侧节点双击关联节点都会展开/收起
    await npuNode.dblclick();
    await expect(npuSubNode).toBeVisible();
    await expect(benchSubNode).toBeVisible();
    await expect(mainArea).toHaveScreenshot('nodeExpandedNpuAndBench.png', { maxDiffPixels: MAX_DIFF_PIXELS });
    await benchNode.dblclick();
    await expect(npuSubNode).toBeHidden();
    // 取消勾选同步展开对应侧节点，只能控制自身
    await syncCheckBox.click();
    await expect(syncCheckBox).not.toBeChecked();
    await npuNode.dblclick();
    await expect(npuSubNode).toBeVisible();
    await expect(benchSubNode).toBeHidden();
    await expect(mainArea).toHaveScreenshot('nodeExpandedOnlyNpu.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });

  // 测试模型图中鼠标事件，包含拖拽，图放缩
  test('test_mouse_drag_and_wheel_graph', async ({ page, mainPage }) => {
    const { mainArea } = mainPage;
    const { npuArea, benchArea } = await mainPage.getBoundingBoxes();
    const { x: startXNpu, y: startYNpu } = npuArea;
    const { x: startXBench, y: startYBench } = benchArea;
    const DRAG_START_OFFSET = 200;
    const DRAG_END_OFFSET = 400;
    // 测试鼠标拖拽移动
    await page.mouse.move(startXNpu + DRAG_START_OFFSET, startYNpu + DRAG_START_OFFSET);
    await page.mouse.down();
    await page.mouse.move(startXNpu + DRAG_END_OFFSET, startYNpu + DRAG_END_OFFSET);
    await page.mouse.up();
    await page.mouse.move(startXBench + DRAG_START_OFFSET, startYBench + DRAG_START_OFFSET);
    await page.mouse.down();
    await page.mouse.move(startXBench + DRAG_END_OFFSET, startYBench + DRAG_END_OFFSET);
    await page.mouse.up();
    await expect(mainArea).toHaveScreenshot('mouseDragDisplay.png', { maxDiffPixels: MAX_DIFF_PIXELS });
    // 测试鼠标滚轮移动
    await page.mouse.move(startXNpu + DRAG_START_OFFSET, startYNpu + DRAG_START_OFFSET);
    await mouseWheelMultipleTimes(page, 0, 100, 4);
    await page.mouse.move(startXBench + DRAG_START_OFFSET, startYBench + DRAG_START_OFFSET);
    await mouseWheelMultipleTimes(page, 0, -100, 4);
    await expect(mainArea).toHaveScreenshot('mouseWheelDisplay.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });

  // 测试鼠标拖拽分割线
  test('test_mouse_drag_splitter', async ({ page, mainPage }) => {
    const { mainArea, splitter } = mainPage;
    const splitterBox = await splitter.boundingBox();
    if (!splitterBox) {
      throw new Error('Test failed because the splitter was not rendered correctly.');
    }
    await splitter.hover();
    await page.mouse.down();
    await page.mouse.move(splitterBox.x - 400, splitterBox.y);
    await page.mouse.up();
    await expect(mainArea).toHaveScreenshot('mouseDragSplitter.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });

  // 测试键盘控制左右移动和放缩
  test('test_move_and_scale_with_keyboard', async ({ page, mainPage }) => {
    const { mainArea } = mainPage;
    const { npuArea, benchArea } = await mainPage.getBoundingBoxes();
    const { x: startXNpu, y: startYNpu } = npuArea;
    const { x: startXBench, y: startYBench } = benchArea;
    await page.mouse.move(startXNpu, startYNpu);
    await pressKeyMultipleTimes(page, 'D', 4);  // 左移
    await pressKeyMultipleTimes(page, 'W', 10);  // 放大
    await page.mouse.move(startXBench, startYBench);
    await pressKeyMultipleTimes(page, 'A', 3);  // 右移
    await pressKeyMultipleTimes(page, 'S', 3);  // 缩小
    await expect(mainArea).toHaveScreenshot('keyboardOperations.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });

  // 测试缩略图相关
  test('test_minimap_display_and_controls', async ({ page, mainPage }) => {
    const { mainArea, npuMinimap } = mainPage;
    const { npuArea } = await mainPage.getBoundingBoxes();
    await page.mouse.move(npuArea.x, npuArea.y);
    await pressKeyMultipleTimes(page, 'D', 4);
    await pressKeyMultipleTimes(page, 'W', 10);
    const npuMinimapArea = await npuMinimap.boundingBox();
    if (!npuMinimapArea) {
      throw new Error('Test failed because the minimap was not rendered correctly.');
    }
    const { x: startX, y: startY } = npuMinimapArea;
    const DRAG_START_OFFSET = 50;
    const DRAG_END_OFFSET = 100;
    await page.mouse.move(startX + DRAG_START_OFFSET, startY + DRAG_START_OFFSET);
    await page.mouse.down();
    await page.mouse.move(startX + DRAG_END_OFFSET, startY + DRAG_END_OFFSET);
    await page.mouse.up();
    // 设置延迟等待动画完成
    await page.waitForTimeout(200);
    await expect(mainArea).toHaveScreenshot('mouseDragMinimap.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });
});
