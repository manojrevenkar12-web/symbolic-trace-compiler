// Copies the generated results into the app. The dashboard never computes a result itself.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dir = resolve(here, process.env.RESULTS_DIR ?? '../../reports');
mkdirSync(resolve(here, '../public'), { recursive: true });
for (const name of ['results.json']) {
  const src = resolve(dir, name);
  if (existsSync(src)) {
    copyFileSync(src, resolve(here, '../public', name));
    console.log(`synced ${src}`);
  } else {
    console.warn(`${src} not found; the dashboard will show its empty state`);
  }
}
