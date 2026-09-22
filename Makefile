.PHONY: help install corpus train eval ablation test lint clean
PY := PYTHONPATH=src PYTHONIOENCODING=utf-8 python

help:
	@echo "make install   editable install + dev deps"
	@echo "make eval      every number in the README (nested CV, ~minutes)"
	@echo "make ablation  operating points + shortcut check (~1 min)"
	@echo "make train     fit the shipped model on the full corpus"
	@echo "make test      pytest (includes evaluation-integrity guards)"
	@echo 'make check MSG="..."   score one message'

install: ; python -m pip install -e ".[dev]"
corpus:  ; $(PY) -m sajag.cli corpus
train:   ; $(PY) -m sajag.cli train
eval:    ; $(PY) -m sajag.cli eval
ablation:; $(PY) tools/ablation.py
check:   ; $(PY) -m sajag.cli check "$(MSG)"
test:    ; $(PY) -m pytest -q
lint:    ; python -m ruff check src tests tools

clean:
	rm -rf data/*.joblib data/results.json __pycache__ .pytest_cache
