// Runs before `npm run dev` and `npm run build`: generate content/ with
// tools/build_docs.py when it is missing, so a fresh checkout needs no
// separate step. Regenerate after editing source/ or docstrings with
// `npm run content` (dev picks the new files up live).
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';

const contentDir = new URL('../content/docs/', import.meta.url);
if (existsSync(contentDir)) process.exit(0);

const python = process.env.PYTHON ?? 'python';
console.log(`content/ is missing; running ${python} ../tools/build_docs.py`);
const result = spawnSync(python, ['../tools/build_docs.py'], { stdio: 'inherit' });
if (result.error) {
  console.error(
    `Could not run ${python} (${result.error.message}). Install the package with ` +
      '`pip install -e ".[docs,viz]"` from the repository root, or set PYTHON.',
  );
  process.exit(1);
}
process.exit(result.status ?? 1);
