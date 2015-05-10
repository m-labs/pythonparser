"""
The :mod:`Diagnostic` module provides several commonly useful
algorithms that operate on abstract syntax trees.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from . import ast

class Visitor:
    """
    A node visitor base class that does a pre-order traversal
    of the abstract syntax tree.

    This class is meant to be subclassed, with the subclass adding visitor
    methods.

    The visitor functions for the nodes are ``'visit_'`` +
    class name of the node.  So a `Try` node visit function would
    be `visit_Try`.
    """

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        pass

    def visit(self, node):
        """Visit a node."""
        visit_attr = 'visit_' + type(node).__name__
        if hasattr(self, visit_attr):
            getattr(self, visit_attr)(node)
        else:
            self.generic_visit(node)

        for field_name in node._fields:
            field_val = getattr(node, field_name)
            if isinstance(field_val, ast.AST):
                self.visit(field_val)
            elif isinstance(field_val, list):
                for field_val_elt in field_val:
                    self.visit(field_val_elt)

def compare(left, right, compare_locs=False):
    """
    An AST comparison function. Returns ``True`` if all fields in
    ``left`` are equal to fields in ``right``; if ``compare_locs`` is
    true, all locations should match as well.
    """
    if type(left) != type(right):
        return False

    if isinstance(left, ast.AST):
        for field in left._fields:
            if not compare(getattr(left, field), getattr(right, field)):
                return False

        if compare_locs:
            for loc in left._locs:
                if getattr(left, loc) != getattr(right, loc):
                    return False

        return True
    elif isinstance(left, list):
        if len(left) != len(right):
            return False

        for left_elt, right_elt in zip(left, right):
            if not compare(left_elt, right_elt):
                return False

        return True
    else:
        return left == right
