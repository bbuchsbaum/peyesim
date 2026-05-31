.PHONY: docs docs-kernel docs-lint docs-preview docs-clean

QUARTO_PYTHON ?= $(shell command -v python)
export QUARTO_PYTHON
PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))
export PYTHONPATH
QUARTO_KERNEL_PREFIX ?= $(CURDIR)/.quarto-kernels
JUPYTER_PATH := $(QUARTO_KERNEL_PREFIX)/share/jupyter$(if $(JUPYTER_PATH),:$(JUPYTER_PATH))
export JUPYTER_PATH

# Editorial lint is a hard prerequisite — prose drift fails the build,
# not review.
docs-lint:
	python tools/docs_lint.py

docs-kernel:
	python -m ipykernel install --prefix "$(QUARTO_KERNEL_PREFIX)" --name python3 --display-name "Python 3 (peyesim docs)"

docs: docs-lint docs-kernel
	cd docs && quartodoc build --config _quartodoc.yml && quarto render

docs-preview: docs-lint docs-kernel
	cd docs && quartodoc build --config _quartodoc.yml && quarto preview

docs-clean:
	rm -rf docs/_site docs/.quarto .quarto-kernels docs/reference/*.qmd docs/reference/_sidebar.yml
	@touch docs/reference/index.qmd
