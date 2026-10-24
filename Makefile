.PHONY: docs docs-content docs-lint docs-preview docs-clean

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

docs-preview: docs-content docs/node_modules
	cd docs && npm run dev

docs-clean:
	rm -rf docs/content docs/.source docs/.next docs/out
