import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

// GitHub Pages serves the site from /peyesim; the docs workflow sets this.
const basePath = process.env.DOCS_BASE_PATH ?? '';

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  basePath,
  // Every page is x/index.html, which GitHub Pages serves unambiguously;
  // scripts/legacy-redirects.mjs keeps the old Quarto x.html URLs working.
  trailingSlash: true,
  images: { unoptimized: true },
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  reactStrictMode: true,
};

export default withMDX(config);
