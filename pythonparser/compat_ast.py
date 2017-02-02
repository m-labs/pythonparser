from __future__ import absolute_import, division, print_function, unicode_literals


from .algorithm import Visitor as NodeVisitor
from .ast import AST, FunctionDef, ClassDef, Module

# Compat
from .ast import *
from . import *


# The CPython one is a singleton, but this will do for now...
class Load(AST):
    pass


def copy_location(new_node, old_node):
    raise NotImplementedError()


def dump(node, annotate_fields=True, include_attributes=False):
    """
    From: https://github.com/python/cpython/blob/2.7/Lib/ast.py

    Return a formatted dump of the tree in *node*.  This is mainly useful for
    debugging purposes.  The returned string will show the names and the values
    for fields.  This makes the code impossible to evaluate, so if evaluation is
    wanted *annotate_fields* must be set to False.  Attributes such as line
    numbers and column offsets are not dumped by default.  If this is wanted,
    *include_attributes* can be set to True.
    """
    def _format(node):
        if isinstance(node, AST):
            fields = [(a, _format(b)) for a, b in iter_fields(node)]
            rv = '%s(%s' % (node.__class__.__name__, ', '.join(
                ('%s=%s' % field for field in fields)
                if annotate_fields else
                (b for a, b in fields)
            ))
            if include_attributes and node._attributes:
                rv += fields and ', ' or ' '
                rv += ', '.join('%s=%s' % (a, _format(getattr(node, a)))
                                for a in node._attributes)
            return rv + ')'
        elif isinstance(node, list):
            return '[%s]' % ', '.join(_format(x) for x in node)
        return repr(node)
    if not isinstance(node, AST):
        raise TypeError('expected AST, got %r' % node.__class__.__name__)
    return _format(node)


def load():
    raise NotImplementedError()

def get_docstring(node, clean=True):
    """
    From: https://github.com/python/cpython/blob/2.7/Lib/ast.py

    Return the docstring for the given node or None if no docstring can
    be found.  If the node provided does not have docstrings a TypeError
    will be raised.
    """
    if not isinstance(node, (FunctionDef, ClassDef, Module)):
        raise TypeError("%r can't have docstrings" % node.__class__.__name__)
    if node.body and isinstance(node.body[0], Expr) and \
       isinstance(node.body[0].value, Str):
        if clean:
            import inspect
            return inspect.cleandoc(node.body[0].value.s)
        return node.body[0].value.s

def _walk(node):
    """
    From: https://github.com/python/cpython/blob/2.7/Lib/ast.py

    Recursively yield all descendant nodes in the tree starting at *node*
    (including *node* itself), in no specified order.  This is useful if you
    only want to modify nodes in place and don't care about the context.
    """
    from collections import deque
    todo = deque([node])
    while todo:
        node = todo.popleft()
        todo.extend(iter_child_nodes(node))
        yield node

def increment_lineno(node, n=1):
    """
    From: https://github.com/python/cpython/blob/2.7/Lib/ast.py

    Increment the line number of each node in the tree starting at *node* by *n*.
    This is useful to "move code" to a different location in a file.
    """
    for child in _walk(node):
        if 'lineno' in child._attributes:
            child.lineno = getattr(child, 'lineno', 0) + n
    return node


def iter_child_nodes(node):
    """
    From: https://github.com/python/cpython/blob/2.7/Lib/ast.py

    Yield all direct child nodes of *node*, that is, all fields that are nodes
    and all items of fields that are lists of nodes.
    """
    for name, field in iter_fields(node):
        if isinstance(field, AST):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, AST):
                    yield item


def iter_fields(node):
    """
    From: https://github.com/python/cpython/blob/2.7/Lib/ast.py

    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


def literal_eval(node_or_string):
    """
    From: https://github.com/python/cpython/blob/2.7/Lib/ast.py

    Safely evaluate an expression node or a string containing a Python
    expression.  The string or node provided may only consist of the following
    Python literal structures: strings, numbers, tuples, lists, dicts, booleans,
    and None.
    """
    _safe_names = {'None': None, 'True': True, 'False': False}
    if isinstance(node_or_string, basestring):
        node_or_string = parse(node_or_string, mode='eval')
    if isinstance(node_or_string, Expression):
        node_or_string = node_or_string.body

    def _convert(node):
        if isinstance(node, Str):
            return node.s
        elif isinstance(node, Num):
            return node.n
        elif isinstance(node, Tuple):
            return tuple(map(_convert, node.elts))
        elif isinstance(node, List):
            return list(map(_convert, node.elts))
        elif isinstance(node, Dict):
            return dict((_convert(k), _convert(v)) for k, v
                        in zip(node.keys, node.values))
        elif isinstance(node, Name):
            if node.id in _safe_names:
                return _safe_names[node.id]
        elif isinstance(node, BinOp) and \
             isinstance(node.op, (Add, Sub)) and \
             isinstance(node.right, Num) and \
             isinstance(node.right.n, complex) and \
             isinstance(node.left, Num) and \
             isinstance(node.left.n, (int, long, float)):
            left = node.left.n
            right = node.right.n
            if isinstance(node.op, Add):
                return left + right
            else:
                return left - right
        raise ValueError('malformed string: node "%s" is a "%s"', node.name, type(node))
    return _convert(node_or_string)


def fix_missing_locations(node):
    """
    From: https://github.com/python/cpython/blob/2.7/Lib/ast.py

    When you compile a node tree with compile(), the compiler expects lineno and
    col_offset attributes for every node that supports them.  This is rather
    tedious to fill in for generated nodes, so this helper adds these attributes
    recursively where not already set, by setting them to the values of the
    parent node.  It works recursively starting at *node*.
    """
    def _fix(node, lineno, col_offset):
        if 'lineno' in node._attributes:
            if not hasattr(node, 'lineno'):
                node.lineno = lineno
            else:
                lineno = node.lineno
        if 'col_offset' in node._attributes:
            if not hasattr(node, 'col_offset'):
                node.col_offset = col_offset
            else:
                col_offset = node.col_offset
        for child in iter_child_nodes(node):
            _fix(child, lineno, col_offset)
    _fix(node, 1, 0)
    return node
