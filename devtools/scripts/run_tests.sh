#!/usr/bin/env bash

pytest -v -s --cov=sparsembar --cov-report=term-missing --cov-report=html --pyargs --doctest-modules "$@" sparsembar
