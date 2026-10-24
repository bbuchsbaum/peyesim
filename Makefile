.PHONY: docs docs-check docs-content docs-lint docs-preview docs-clean

# Editorial lint is a hard prerequisite — prose drift fails the build,
# not review.
docs-lint:
	python tools/docs_lint.py

# Execute docs/source pages and generate the API reference into docs/content.
docs-content: docs-lint
	python tools/build_docs.py

docs/node_modules: docs/package-lock.json
	cd docs && npm ci

# Static site in docs/out (set DOCS_BASE_PATH=/peyesim for GitHub Pages).
docs: docs-content docs/node_modules
	cd docs && npm run build

# Lint and type-check the site, build it as GitHub Pages serves it, and check
# every internal link and anchor.
docs-check: docs-content docs/node_modules
	cd docs && npm run lint && npm run types:check && DOCS_BASE_PATH=/peyesim npm run build
	python tools/check_docs_links.py docs/out --base /peyesim

docs-preview: docs-content docs/node_modules
	cd docs && npm run dev

docs-clean:
	rm -rf docs/content docs/public/reference docs/.source docs/.next docs/out
