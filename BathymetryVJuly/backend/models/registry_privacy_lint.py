"""TRAIN-R8 — cheap privacy lint: flag any model_card.json line that looks
like a raw decimal-degree coordinate (5+ decimal places — point_fingerprint's
own ".6f" formatting, distinct from rounded metrics which use 2-4 decimals
throughout this repo, e.g. `round(rmse, 3)`) outside a "bbox" key.
Run against a git diff or a file directly.
"""
from __future__ import annotations

import re
import sys

_DECIMAL_RE = re.compile(r"-?\d{1,3}\.\d{5,}")
_BBOX_LINE_RE = re.compile(r'"bbox"\s*:')


def lint_lines(lines: list[str]) -> list[str]:
    violations = []
    in_bbox_array = False
    for i, line in enumerate(lines):
        if _BBOX_LINE_RE.search(line):
            in_bbox_array = True
            continue
        if in_bbox_array:
            if "]" in line:
                in_bbox_array = False
            continue
        if _DECIMAL_RE.search(line):
            violations.append(f"line {i + 1}: {line.strip()}")
    return violations


def lint_file(path: str) -> list[str]:
    with open(path) as fh:
        return lint_lines(fh.readlines())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: registry_privacy_lint.py <model_card.json>")
        sys.exit(1)
    v = lint_file(sys.argv[1])
    if v:
        print(f"PRIVACY LINT FAILED ({len(v)} violation(s)):")
        for line in v:
            print(" ", line)
        sys.exit(1)
    print("PRIVACY LINT OK — no raw-coordinate-shaped lines outside bbox")
