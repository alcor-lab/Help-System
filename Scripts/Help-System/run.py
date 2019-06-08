#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
"""
Run anything: scripts, tests, examples, applications.
"""
import __init__
from project_tools import run


def main():
    run(file=__file__, doc=__doc__)


if __name__ == '__main__':
    import sys
    sys.exit(main())
