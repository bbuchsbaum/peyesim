.PHONY: docs docs-preview docs-clean

docs:
	cd docs && quartodoc build --config _quartodoc.yml && QUARTO_PYTHON=$${QUARTO_PYTHON:-python3} quarto render

docs-preview:
	cd docs && quartodoc build --config _quartodoc.yml && QUARTO_PYTHON=$${QUARTO_PYTHON:-python3} quarto preview

docs-clean:
	rm -rf docs/_site docs/.quarto docs/reference/*.qmd docs/reference/_sidebar.yml
	@touch docs/reference/index.qmd
