import { test, expect } from '@playwright/test';

test.describe.configure({ mode: 'serial' });

async function firstQueuedPart(page) {
  await page.goto('/');
  const first = page.getByTestId('queue-item').first();
  await expect(first).toBeVisible();
  const id = await first.getAttribute('data-part-id');
  await expect(page.getByTestId('detail')).toHaveAttribute('data-part-id', id);
  return Number(id);
}

test('engineer clears a part from the keyboard and the label lands in the training set', async ({ page, request }) => {
  const id = await firstQueuedPart(page);
  await page.keyboard.press('h');
  await expect(page.getByTestId('last-action')).toContainText(`Part ${id}: ship`);
  await expect(page.locator(`[data-testid="queue-item"][data-part-id="${id}"]`)).toHaveCount(0);
  const confirmed = await (await request.get('/api/training-set/confirmed')).json();
  expect(confirmed).toContainEqual(expect.objectContaining({ part_id: id, action: 'ship', label: 0 }));
});

test('scrap from the keyboard records a failure label', async ({ page, request }) => {
  await firstQueuedPart(page);
  await page.keyboard.press('j');
  const second = await page.locator('[data-testid="queue-item"].selected').getAttribute('data-part-id');
  await expect(page.getByTestId('detail')).toHaveAttribute('data-part-id', second);
  await page.keyboard.press('s');
  await expect(page.getByTestId('last-action')).toContainText(`Part ${second}: scrap`);
  const confirmed = await (await request.get('/api/training-set/confirmed')).json();
  expect(confirmed).toContainEqual(expect.objectContaining({ part_id: Number(second), action: 'scrap', label: 1 }));
});

test('a fabricated column name in an agent response is rejected before render', async ({ page, request }) => {
  const queue = await (await request.get('/api/queue')).json();
  const id = queue.items[0].part_id;
  const { part } = await (await request.get(`/api/parts/${id}`)).json();
  const real = part.columns[0];
  const payload = {
    part_id: id, allowed_columns: part.columns, allowed_part_ids: [id],
    disposition: {
      route: part.route, recommendation: 'scrap', out_of_range: [], neighbor_summary: 'No similar parts reviewed.',
      reason: `${real} looks normal but L9_S99_F99999 confirms a hidden defect on this part.`,
      citations: [{ kind: 'column', value: real }, { kind: 'column', value: 'L9_S99_F99999' }],
    },
  };
  expect((await request.post('/api/test/dispositions', { data: payload })).ok()).toBeTruthy();
  await firstQueuedPart(page);
  await expect(page.getByTestId('disposition-rejected')).toBeVisible();
  await expect(page.getByTestId('disposition-rejected')).toContainText('L9_S99_F99999');
  await expect(page.getByText('confirms a hidden defect')).toHaveCount(0);
  await expect(page.getByTestId('disposition-valid')).toHaveCount(0);
});
