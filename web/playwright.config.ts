import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  timeout: 30_000,
  reporter: [['list']],
  // CI installs the matching browser; elsewhere a pre-installed Chromium can be named in PW_CHROMIUM.
  use: { baseURL: 'http://localhost:4178', launchOptions: process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {} },
  webServer: { command: 'npm run preview', port: 4178, reuseExistingServer: true },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1360, height: 900 } } },
    { name: 'phone', use: { ...devices['Pixel 7'] } },
  ],
});
