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

import { test, expect } from '@playwright/test';

test.describe('BaseTextTest', () => {
  test.beforeEach(async ({ page }) => {
    const allParsedPromise = page.waitForResponse(response =>
      response.url().includes('/loadGraphConfigInfo') && response.status() === 200
    );
    page.goto('/');
    await allParsedPromise;
    await page.getByRole('combobox', { name: '目录' }).click();
    await page.getByRole('option', { name: 'textCompare' }).click();
  });

  test('switch_language', async ({ page }) => {
    const switchBtn = page.getByRole('button', { name: '中|en' });
    await switchBtn.click();
    await expect(page.locator('graph-ascend')).toHaveScreenshot('textInEnglish.png', { maxDiffPixels: 300 });
  });
});

 