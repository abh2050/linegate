import { defineConfig, devices } from '@playwright/test';

const root = new URL('..', import.meta.url).pathname;

export default defineConfig({
  testDir: '../tests/e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  use: { baseURL: 'http://127.0.0.1:8765', trace: 'retain-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'uv run python -m linegate.console.api --port 8765 --test',
    cwd: root,
    url: 'http://127.0.0.1:8765/api/queue',
    timeout: 300_000,
    reuseExistingServer: false,
    env: {
      LINEGATE_CONFIRMED_PATH: `${root}data/artifacts/e2e/confirmed_labels.jsonl`,
      LINEGATE_DISPOSITION_DIR: `${root}data/artifacts/e2e/dispositions`,
    },
  },
});
