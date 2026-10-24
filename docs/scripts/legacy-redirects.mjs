// The Quarto site served pages as x.html. After `next build`, write a small
// redirect at each x.html pointing to the page's new x/ address, so old links
// and search-engine entries keep working.
import { existsSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';

const out = new URL('../out/', import.meta.url).pathname;
const base = process.env.DOCS_BASE_PATH ?? '';
let written = 0;

function walk(dir) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (!statSync(full).isDirectory() || name.startsWith('_')) continue;
    if (existsSync(join(full, 'index.html')) && !existsSync(`${full}.html`)) {
      const target = `${base}/${relative(out, full)}/`;
      writeFileSync(
        `${full}.html`,
        `<!doctype html><meta charset="utf-8"><title>Moved</title>` +
          `<link rel="canonical" href="${target}">` +
          `<meta http-equiv="refresh" content="0; url=${target}">` +
          `<a href="${target}">${target}</a>\n`,
      );
      written += 1;
    }
    walk(full);
  }
}

walk(out);
console.log(`legacy-redirects: ${written} redirect pages`);
