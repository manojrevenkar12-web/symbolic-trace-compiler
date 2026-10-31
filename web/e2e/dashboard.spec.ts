import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('every section renders from the generated results, without console errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('every sequence length');
  for (const name of [
    'Programs compiled per workload',
    'Are the guards sound?',
    'Programs written by someone who never saw the compiler',
    'Thunder on the same calls',
    'Release gate',
  ]) {
    await expect(page.getByRole('heading', { level: 2, name })).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test('switching the workload changes the listing', async ({ page }) => {
  await page.goto('/');
  const listing = page.getByLabel('Generated computation');
  const before = await listing.textContent();
  await page.getByRole('button', { name: 'Chat prefill (Llama)' }).click();
  await expect(listing).not.toHaveText(before ?? '');
});

test('nothing scrolls sideways on a phone', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 2, name: 'Release gate' })).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

test('no serious accessibility violations', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 2, name: 'Release gate' })).toBeVisible();
  const a = await new AxeBuilder({ page } as unknown as ConstructorParameters<typeof AxeBuilder>[0]).withTags(['wcag2a', 'wcag2aa']).analyze();
  expect(a.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical')).toEqual([]);
});
