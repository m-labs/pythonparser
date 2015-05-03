"""
The :mod:`ast` module contains the shims which implement AST classes
missing in versions of Python earlier than the latest one pyparser targets.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from ast import *
import sys

if sys.version_info <= (2, 6):
    class DictComp(expr):
        _fields = ('key', 'value', 'generators')

    class Set(expr):
        _fields = ('elts',)

    class SetComp(expr):
        _fields = ('elt', 'generators')

if sys.version_info >= (3,):
    class Repr(expr):
        _fields = ('value',)

    class Exec(expr):
        _fields = ('body', 'globals', 'locals')

    class Print(expr):
        _fields = ('dest', 'values', 'nl')

    class TryExcept(stmt):
        _fields = ('body', 'handlers', 'orelse')
