#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
"""
Command actions:
  get     - get requirements
  prepare - prepare the project (e.g. generate source code)
  build   - build the project
  run     - run programs
  install - install the project

These actions can be overridden by passing a function to the 'setup' as
a keyword argument.

As an example, for `get` command action, you can redefine:
  get(regular_script)              - basic command action
  get_docs(regular_script)         - optional extension for Documentation
  get_examples(regular_script)     - optional extension for Examples
  get_tests(regular_script)        - optional extension for Tests
"""
import __init__  # noqa
from project_tools import setup


def main():
    setup(file=__file__, doc=__doc__, build_docs=build_docs)


def build_docs(regular_script):
    """
    This is an example of how to override the action.
    The first parameter is an instance of RegularScript.
    It has methods for each action with a default behaviour.

    Invoke it with:
      ./setup.py build --docs
    or:
      hey project setup build --docs
    """
    regular_script.build_docs()


if __name__ == '__main__':
    import sys
    sys.exit(main())
