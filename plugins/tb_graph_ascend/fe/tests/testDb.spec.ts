import { test, expect } from '@playwright/test';

test.describe('BaseDbTest', () => {
  test.beforeEach(async ({ page }) => {
    const allParsedPromise = page.waitForResponse(response =>
      response.url().includes('/loadGraphConfigInfo') && response.status() === 200
    );
    page.goto('/');
    await allParsedPromise;
    await page.getByRole('combobox', { name: '目录' }).click();
    await page.getByRole('option', { name: 'dbCompare' }).click();
  });
    
  test('switch_language', async ({ page }) => {
    const switchBtn = page.getByRole('button', { name: '中|en' });
    await switchBtn.click();
    await expect(page.locator('graph-ascend')).toHaveScreenshot('dbInEnglish.png', { maxDiffPixels: 300 });
  });
});

 