# -*- coding: utf-8 -*-
#
# Usage:
# import __init__
#
# Purpose:
# Import this file in python executables to add Libraries and
# Dependencies' Libraries to python's sys.path.
#
from os.path import abspath, dirname, isdir, join, realpath
import os
import sys

THIS_DIR = dirname(realpath(__file__))
PROJECT_DIR = abspath(join(THIS_DIR, '..', '..'))
LIBRARIES_DIR = join(PROJECT_DIR, 'Libraries')
DEPENDENCIES_DIR = join(PROJECT_DIR, 'Dependencies')
PROJECT_TOOLS = 'ProjectTools'


def add_directories_to_path():
    directory_list = find_dependency_libraries()

    add_to_path(LIBRARIES_DIR)

    for directory in directory_list:
        add_to_path(directory)

    if THIS_DIR in sys.path:
        sys.path.remove(THIS_DIR)


def add_to_path(directory):
    if directory not in sys.path:
        sys.path.append(directory)


def find_dependency_libraries():
    dependencies = list_directory(DEPENDENCIES_DIR)

    # prioritise Dependencies/ProjectTools to the ProjectTools
    # in other dependencies outputs folder
    moveToFirst(dependencies, PROJECT_TOOLS)

    other_libraries = [choose_library_path(directory)
                       for directory in dependencies]

    return list(filter(None, other_libraries))


def moveToFirst(some_list, value):
    if value in some_list:
        value_index = some_list.index(value)
        some_list.pop(value_index)
        some_list.insert(0, value)


def list_directory(directory):
    try:
        return os.listdir(directory)
    except FileNotFoundError:
        return []


def choose_library_path(directory):
    if isdir(get_outputs_path(directory)):
        return get_outputs_path(directory)
    elif isdir(get_library_path(directory)):
        return get_library_path(directory)


def get_outputs_path(directory):
    return join(DEPENDENCIES_DIR, directory, 'Outputs', 'Libraries')


def get_library_path(directory):
    return join(DEPENDENCIES_DIR, directory, 'Libraries')


add_directories_to_path()
