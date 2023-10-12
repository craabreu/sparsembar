#!/usr/bin/env bash
set -e -v
isort ./sparsembar
black --line-length 100 ./sparsembar
flake8 --ignore=E203,W503 ./sparsembar
pylint --rcfile=devtools/linters/pylintrc ./sparsembar
