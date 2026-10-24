import { createGetUrl } from 'fumadocs-core/source';

export const appName = 'peyesim';
export const appDescription =
  'Fixation densities, template similarity and scanpath comparison for eye-movement data.';
// The docs are the whole site, so they live at the root.
export const docsRoute = '/';
export const docsContentRoute = '/llms.mdx';
// Set by the GitHub Pages build (DOCS_BASE_PATH=/peyesim); empty locally.
export const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export const gitConfig = {
  user: 'bbuchsbaum',
  repo: 'peyesim',
  branch: 'main',
};

export const repoUrl = `https://github.com/${gitConfig.user}/${gitConfig.repo}`;

const getContentUrl = createGetUrl(docsContentRoute);

export function getPageMarkdownUrl(page: { slugs: string[]; locale?: string }) {
  const segments = [...page.slugs, 'content.md'];

  return { segments, url: getContentUrl(segments, page.locale) };
}
