---
title: "The license is declared as a plain \"MIT\" string, so building needs setuptools 77 or newer"
author: Amateur
date: 2026-11-19
branch: main
cost: { minutes: 21, tokens: 129035 }
id: l-4abd661e
scope: repo
status: active
src: agent
touches:
  - LICENSE
  - README.md
  - pyproject.toml
---

## Decided
- Pandalyzer is MIT-licensed. The copyright holder and package author is 4m4teur, the name the author chose. Commits show the git user name "Amateur", so don't take credit names from git config.
- The license is declared the PEP 639 way, as the SPDX string `license = "MIT"`. setuptools only reads that form from version 77, so the build requirement went from 68 to 77. The build picks up LICENSE by itself and records `License-Expression: MIT` and `License-File: LICENSE` in the wheel metadata.

## Tried and ruled out
- Keeping setuptools>=68: setuptools 76.1.0 rejects the string with "configuration error: `project.license` must be valid exactly by one definition (2 matches found)".
- The table form `license = { text = "MIT" }`, which older setuptools accepts: setuptools 84 still builds it, but warns that the table form is deprecated and has to be gone by 2027-Feb-18.

## Watch out
- Don't add the "License :: OSI Approved :: MIT License" classifier that templates and PyPI guides often include. Next to an SPDX `license` string, setuptools 77+ refuses to build: "License classifiers have been superseded by license expressions".

In the code:
- LICENSE: the standard MIT text, "Copyright (c) 2026 4m4teur"
- pyproject.toml: `requires = ["setuptools>=77"]`, `license = "MIT"`, `authors = [{ name = "4m4teur" }]`
- To check the metadata: `pip wheel . --no-deps -w <dir>`, then `unzip -p <dir>/*.whl '*/METADATA' | grep -E '^(Author|License)'`. CI's `pip install -e ".[dev]"` uses the same build requirement and passed on Python 3.10–3.14.
