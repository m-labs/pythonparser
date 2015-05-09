# encoding: utf-8

"""
The :mod:`ast` module contains the classes comprising the Python abstract syntax tree.

All attributes ending with ``loc`` contain instances of :class:`.source.Range`
or None. All attributes ending with ``_locs`` contain lists of instances of
:class:`.source.Range` or [].

The attribute ``loc``, present in every class except those inheriting :class:`boolop`,
has a special meaning: it encompasses the entire AST node, so that it is possible
to cut the range contained inside ``loc`` of a parsetree fragment and paste it
somewhere else without altering said parsetree fragment that.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Location mixins

class commonloc(object):
    """
    A mixin common for all nodes.

    :cvar _locs: (tuple of strings)
        names of all attributes with location values

    :ivar loc: range encompassing all locations defined for this node
        or its children
    """

    _locs = ('loc',)

    def __repr__(self):
        def value(name):
            try:
                loc = self.__dict__[name]
                if isinstance(loc, list):
                    return "[%s]" % (', '.join(map(repr, loc)))
                else:
                    return repr(loc)
            except:
                return "(!!!MISSING!!!)"
        fields = ', '.join(map(lambda name: "%s=%s" % (name, value(name)),
                           self._fields + self._locs))
        return "%s(%s)" % (self.__class__.__name__, fields)

class keywordloc(commonloc):
    """
    A mixin common for all keyword statements, e.g. ``pass`` and ``yield expr``.

    :ivar keyword_loc: location of the keyword, e.g. ``yield``.
    """
    _locs = commonloc._locs + ('keyword_loc',)

class beginendloc(commonloc):
    """
    A mixin common for nodes with a opening and closing delimiters, e.g. tuples and lists.

    :ivar begin_loc: location of the opening delimiter, e.g. ``(``.
    :ivar end_loc: location of the closing delimiter, e.g. ``)``.
    """
    _locs = commonloc._locs + ('begin_loc', 'end_loc')

# AST nodes

class AST:
    """
    An ancestor of all nodes.

    :cvar _fields: (tuple of strings)
        names of all attributes with semantic values
    """
    _fields = ()

    def __init__(self, **fields):
        for field in fields:
            setattr(self, field, fields[field])

class alias(AST, commonloc):
    """
    An import alias, e.g. ``x as y``.

    :ivar name: (string) value to import
    :ivar asname: (string) name to add to the environment
    :ivar name_loc: location of name
    :ivar as_loc: location of ``as``
    :ivar asname_loc: location of asname
    """
    _fields = ('name', 'asname')
    _locs = commonloc._locs + ('name_loc', 'as_loc', 'asname_loc')

class arguments(AST, beginendloc):
    """
    Function definition arguments, e.g. in ``def f(x, y=1, *z, **t)``.

    :ivar args: (list of assignable node) regular formal arguments
    :ivar vararg: (string) splat formal argument (if any), e.g. in ``*x``
    :ivar kwarg: (string) keyword splat formal argument (if any), e.g. in ``**x``
    :ivar defaults: (list of node) values of default arguments
    :ivar star_loc: location of ``*``, if any
    :ivar vararg_loc: location of splat formal argument, if any
    :ivar dstar_loc: location of ``**``, if any
    :ivar kwarg_loc: location of keyword splat formal argument, if any
    :ivar equals_locs: locations of ``=``
    """
    _fields = ('args', 'vararg', 'kwarg', 'defaults')
    _locs = beginendloc._locs + ('star_loc', 'vararg_loc', 'dstar_loc',
                                 'vararg_loc', 'kwarg_loc', 'equals_locs')

class boolop(AST, commonloc):
    """
    Base class for binary boolean operators.

    This class is unlike others in that it does not have the ``loc`` field.
    It serves only as an indicator of operation and corresponds to no source
    itself; locations are recorded in :class:`BoolOp`.
    """
    _locs = ()
class And(boolop):
    """The ``and`` operator."""
class Or(boolop):
    """The ``or`` operator."""

class cmpop(AST, commonloc):
    """Base class for comparison operators."""
class Eq(cmpop):
    """The ``==`` operator."""
class Gt(cmpop):
    """The ``>`` operator."""
class GtE(cmpop):
    """The ``>=`` operator."""
class In(cmpop):
    """The ``in`` operator."""
class Is(cmpop):
    """The ``is`` operator."""
class IsNot(cmpop):
    """The ``is not`` operator."""
class Lt(cmpop):
    """The ``<`` operator."""
class LtE(cmpop):
    """The ``<=`` operator."""
class NotEq(cmpop):
    """The ``!=`` (or deprecated ``<>``) operator."""
class NotIn(cmpop):
    """The ``not in`` operator."""

class comprehension(AST, commonloc):
    """
    A single ``for`` list comprehension clause.

    :ivar target: (node) the variable(s) bound in comprehension body
    :ivar iter: (node) the expression being iterated
    :ivar ifs: (list of node) the ``if`` clauses
    :ivar for_loc: location of the ``for`` keyword
    :ivar in_loc: location of the ``in`` keyword
    :ivar if_locs: locations of ``if`` keywords
    """
    _fields = ('target', 'iter', 'ifs')
    _locs = commonloc._locs + ('for_loc', 'in_loc', 'if_locs')

class excepthandler(AST, commonloc):
    """Base class for the exception handler."""
class ExceptHandler(excepthandler):
    """
    An exception handler, e.g. ``except x as y:·  z``.

    :ivar type: (node) type of handled exception, if any
    :ivar name: (assignable node) variable bound to exception, if any
    :ivar body: (list of node) code to execute when exception is caught
    :ivar except_loc: location of ``except``
    :ivar as_loc: location of ``as``, if any
    :ivar colon_loc: location of ``:``
    """
    _fields = ('type', 'name', 'body')
    _locs = excepthandler._locs + ('except_loc', 'as_loc', 'colon_loc')

class expr(AST, commonloc):
    """Base class for expression nodes."""
class Attribute(expr):
    """
    An attribute access, e.g. ``x.y``.

    :ivar value: (node) left-hand side
    :ivar attr: (string) attribute name
    """
    _fields = ('value', 'attr', 'ctx')
    _locs = expr._locs + ('dot_loc', 'attr_loc')
class BinOp(expr):
    """
    A binary operation, e.g. ``x + y``.

    :ivar left: (node) left-hand side
    :ivar op: (:class:`operator`) operator
    :ivar right: (node) right-hand side
    """
    _fields = ('left', 'op', 'right')
class BoolOp(expr):
    """
    A boolean operation, e.g. ``x and y``.

    :ivar op: (:class:`boolop`) operator
    :ivar values: (list of node) operands
    :ivar op_locs: locations of operators
    """
    _fields = ('op', 'values')
    _locs = expr._locs + ('op_locs',)
class Call(expr, beginendloc):
    """
    A function call, e.g. ``f(x, y=1, *z, **t)``.

    :ivar func: (node) function to call
    :ivar args: (list of node) regular arguments
    :ivar keywords: (list of :class:`keyword`) keyword arguments
    :ivar starargs: (node) splat argument (if any), e.g. in ``*x``
    :ivar kwargs: (node) keyword splat argument (if any), e.g. in ``**x``
    :ivar star_loc: location of ``*``, if any
    :ivar dstar_loc: location of ``**``, if any
    """
    _fields = ('func', 'args', 'keywords', 'starargs', 'kwargs')
    _locs = beginendloc._locs + ('star_loc', 'dstar_loc')
class Compare(expr):
    """
    A comparison operation, e.g. ``x < y`` or ``x < y > z``.

    :ivar left: (node) left-hand
    :ivar ops: (list of :class:`cmpop`) compare operators
    :ivar comparators: (list of node) compare values
    """
    _fields = ('left', 'ops', 'comparators')
class Dict(expr, beginendloc):
    """
    A dictionary, e.g. ``{x: y}``.

    :ivar keys: (list of node) keys
    :ivar values: (list of node) values
    :ivar colon_locs: ``:`` locations
    """
    _fields = ('keys', 'values')
    _locs = beginendloc._locs + ('colon_locs',)
class DictComp(expr, beginendloc):
    """
    A list comprehension, e.g. ``{x: y for x,y in z}``.

    **Emitted since 2.7.**

    :ivar key: (node) key part of comprehension body
    :ivar value: (node) value part of comprehension body
    :ivar generators: (list of :class:`comprehension`) ``for`` clauses
    """
    _fields = ('key', 'value', 'generators')
class GeneratorExp(expr, beginendloc):
    """
    A generator expression, e.g. ``(x for x in y)``.

    :ivar elt: (node) expression body
    :ivar generators: (list of :class:`comprehension`) ``for`` clauses
    """
    _fields = ('elt', 'generators')
class IfExp(expr):
    """
    A conditional expression, e.g. ``x if y else z``.

    :ivar test: (node) condition
    :ivar body: (node) value if true
    :ivar orelse: (node) value if false
    :ivar if_loc: location of ``if``
    :ivar else_loc: location of ``else``
    """
    _fields = ('test', 'body', 'orelse')
    _locs = expr._locs + ('if_loc', 'else_loc')
class Lambda(expr):
    """
    A lambda expression, e.g. ``lambda x: x*x``.

    :ivar args: (:class:`arguments`) arguments
    :ivar body: (node) body
    :ivar lambda_loc: location of ``lambda``
    :ivar colon_loc: location of ``:``
    """
    _fields = ('args', 'body')
    _locs = expr._locs + ('lambda_loc', 'colon_loc')
class List(expr, beginendloc):
    """
    A list, e.g. ``[x, y]``.

    :ivar elts: (list of node) elements
    """
    _fields = ('elts', 'ctx')
class ListComp(expr, beginendloc):
    """
    A list comprehension, e.g. ``[x for x in y]``.

    :ivar elt: (node) comprehension body
    :ivar generators: (list of :class:`comprehension`) ``for`` clauses
    """
    _fields = ('elt', 'generators')
class Name(expr):
    """
    An identifier, e.g. ``x``.

    :ivar id: (string) name
    """
    _fields = ('id', 'ctx')
class Num(expr):
    """
    An integer, floating point or complex number, e.g. ``1``, ``1.0`` or ``1.0j``.

    :ivar n: (int, float or complex) value
    """
    _fields = ('n',)
class Repr(expr, beginendloc):
    """
    A repr operation, e.g. ``\`x\```

    **Emitted until 3.0.**

    :ivar value: (node) value
    """
    _fields = ('value',)
class Set(expr, beginendloc):
    """
    A set, e.g. ``{x, y}``.

    **Emitted since 2.7.**

    :ivar elts: (list of node) elements
    """
    _fields = ('elts',)
class SetComp(expr, beginendloc):
    """
    A set comprehension, e.g. ``{x for x in y}``.

    **Emitted since 2.7.**

    :ivar elt: (node) comprehension body
    :ivar generators: (list of :class:`comprehension`) ``for`` clauses
    """
    _fields = ('elt', 'generators')
class Str(expr, beginendloc):
    """
    A string, e.g. ``"x"``.

    :ivar s: (string) value
    """
    _fields = ('s',)
class Subscript(expr, beginendloc):
    """
    A subscript operation, e.g. ``x[1]``.

    :ivar value: (node) object being sliced
    :ivar slice: (:class:`slice`) slice
    """
    _fields = ('value', 'slice', 'ctx')
class Tuple(expr, beginendloc):
    """
    A tuple, e.g. ``(x,)`` or ``x,y``.

    :ivar elts: (list of nodes) elements
    """
    _fields = ('elts', 'ctx')
class UnaryOp(expr):
    """
    An unary operation, e.g. ``+x``.

    :ivar op: (:class:`unaryop`) operator
    :ivar operand: (node) operand
    """
    _fields = ('op', 'operand')
class Yield(expr):
    """
    A yield expression, e.g. ``(yield x)``.

    :ivar value: (node) yielded value
    :ivar yield_loc: location of ``yield``
    """
    _fields = ('value',)
    _locs = expr._locs + ('yield_loc',)

# expr_context
#     AugLoad
#     AugStore
#     Del
#     Load
#     Param
#     Store

class keyword(AST, commonloc):
    """
    A keyword actual argument, e.g. in ``f(x=1)``.

    :ivar arg: (string) name
    :ivar value: (node) value
    :ivar equals_loc: location of ``=``
    """
    _fields = ('arg', 'value')
    _locs = commonloc._locs + ('arg_loc', 'equals_loc')

class mod(AST, commonloc):
    """Base class for modules (groups of statements)."""
    _fields = ('body',)
class Expression(mod):
    """A group of statements parsed as if for :func:`eval`."""
class Interactive(mod):
    """A group of statements parsed as if it was REPL input."""
class Module(mod):
    """A group of statements parsed as if it was a file."""

class operator(AST, commonloc):
    """Base class for numeric binary operators."""
class Add(operator):
    """The ``+`` operator."""
class BitAnd(operator):
    """The ``&`` operator."""
class BitOr(operator):
    """The ``|`` operator."""
class BitXor(operator):
    """The ``^`` operator."""
class Div(operator):
    """The ``\\`` operator."""
class FloorDiv(operator):
    """The ``\\\\`` operator."""
class LShift(operator):
    """The ``<<`` operator."""
class Mod(operator):
    """The ``%`` operator."""
class Mult(operator):
    """The ``*`` operator."""
class Pow(operator):
    """The ``**`` operator."""
class RShift(operator):
    """The ``>>`` operator."""
class Sub(operator):
    """The ``-`` operator."""

class slice(AST, commonloc):
    """Base class for slice operations."""
class Ellipsis(slice):
    """The ellipsis, e.g. in ``x[...]``."""
class ExtSlice(slice):
    """
    The multiple slice, e.g. in ``x[0:1, 2:3]``.

    :ivar dims: (:class:`slice`) sub-slices
    """
    _fields = ('dims',)
class Index(slice):
    """
    The index, e.g. in ``x[1]``.

    :ivar value: (node) index
    """
    _fields = ('value',)
class Slice(slice):
    """
    The slice, e.g. in ``x[0:1]`` or ``x[0:1:2]``.

    :ivar lower: (node or None) lower bound
    :ivar upper: (node or None) upper bound
    :ivar step: (node or None) iteration step
    :ivar bound_colon_loc: location of first semicolon
    :ivar step_colon_loc: location of second semicolon
    """
    _fields = ('lower', 'upper', 'step')
    _locs = slice._locs + ('bound_colon_loc', 'step_colon_loc')

class stmt(AST, commonloc):
    """Base class for statement nodes."""
class Assert(stmt, keywordloc):
    """
    The ``assert x, msg`` statement.

    :ivar test: (node) condition
    :ivar msg: (node) message, if any
    """
    _fields = ('test', 'msg')
class Assign(stmt):
    """
    The ``=`` statement.

    :ivar targets: (list of assignable node) left-hand sides
    :ivar value: (node) right-hand side
    :ivar op_locs: location of equality signs corresponding to ``targets``
    """
    _fields = ('targets', 'value')
    _locs = stmt._locs + ('op_locs',)
class AugAssign(stmt):
    """
    The operator-assignment statement, e.g. ``+=``.

    :ivar target: (assignable node) left-hand side
    :ivar op: (:class`) operator
    :ivar value: (node) right-hand side
    """
    _fields = ('target', 'op', 'value')
class Break(stmt, keywordloc):
    """The ``break`` statement."""
class ClassDef(stmt, keywordloc):
    """
    The ``class x(z, y):·  t`` statement.

    :ivar name: (string) name
    :ivar bases: (list of node) base classes
    :ivar body: (list of node) body
    :ivar decorator_list: (list of node) decorators
    :ivar keyword_loc: location of ``class``
    :ivar name_loc: location of name
    :ivar lparen_loc: location of ``(``, if any
    :ivar rparen_loc: location of ``)``, if any
    :ivar colon_loc: location of ``:``
    :ivar at_locs: locations of decorator ``@``
    """
    _fields = ('name', 'bases', 'body', 'decorator_list')
    _locs = keywordloc._locs + ('name_loc', 'lparen_loc', 'rparen_loc', 'colon_loc', 'at_locs')
class Continue(stmt, keywordloc):
    """The ``continue`` statement."""
class Delete(stmt, keywordloc):
    """
    The ``del x, y`` statement.

    :ivar targets: (list of :class:`Name`)
    """
    _fields = ('targets',)
class Exec(stmt, keywordloc):
    """
    The ``exec code in locals, globals`` statement.

    **Emitted until 3.0.**

    :ivar body: (node) code
    :ivar locals: (node) locals
    :ivar globals: (node) globals
    :ivar keyword_loc: location of ``exec``
    :ivar in_loc: location of ``in``
    """
    _fields = ('body', 'locals', 'globals')
    _locs = keywordloc._locs + ('in_loc',)
class Expr(stmt):
    """
    An expression in statement context. The value of expression is discarded.

    :ivar value: (:class:`expr`) value
    """
    _fields = ('value',)
class For(stmt, keywordloc):
    """
    The ``for x in y:·  z·else:·  t`` statement.

    :ivar target: (assignable node) loop variable
    :ivar iter: (node) loop collection
    :ivar body: (list of node) code for every iteration
    :ivar orelse: (list of node) code if empty
    :ivar keyword_loc: location of ``for``
    :ivar in_loc: location of ``in``
    :ivar for_colon_loc: location of colon after ``for``
    :ivar else_loc: location of ``else``, if any
    :ivar else_colon_loc: location of colon after ``else``, if any
    """
    _fields = ('target', 'iter', 'body', 'orelse')
    _locs = keywordloc._locs + ('in_loc', 'for_colon_loc', 'else_loc', 'else_colon_loc')
class FunctionDef(stmt, keywordloc):
    """
    The ``def f(x):·  y`` statement.

    :ivar name: (string) name
    :ivar args: (:class:`arguments`) formal arguments
    :ivar body: (list of node) body
    :ivar decorator_list: (list of node) decorators
    :ivar keyword_loc: location of ``def``
    :ivar name_loc: location of name
    :ivar colon_loc: location of ``:``, if any
    :ivar at_locs: locations of decorator ``@``
    """
    _fields = ('name', 'args', 'body', 'decorator_list')
    _locs = keywordloc._locs + ('name_loc', 'colon_loc', 'at_locs')
class Global(stmt, keywordloc):
    """
    The ``global x, y`` statement.

    :ivar names: (list of string) names
    :ivar name_locs: locations of names
    """
    _fields = ('names',)
    _locs = keywordloc._locs + ('name_locs',)
class If(stmt, keywordloc):
    """
    The ``if x:·  y·else:·  z`` or ``if x:·  y·elif: z·  t`` statement.

    :ivar test: (node) condition
    :ivar body: (list of node) code if true
    :ivar orelse: (list of node) code if false
    :ivar if_colon_loc: location of colon after ``if`` or ``elif``
    :ivar else_loc: location of ``else``, if any
    :ivar else_colon_loc: location of colon after ``else``, if any
    """
    _fields = ('test', 'body', 'orelse')
    _locs = keywordloc._locs + ('if_colon_loc', 'else_loc', 'else_colon_loc')
class Import(stmt, keywordloc):
    """
    The ``import x, y`` statement.

    :ivar names: (list of :class:`alias`) names
    """
    _fields = ('names',)
class ImportFrom(stmt, keywordloc):
    """
    The ``from ...x import y, z`` or ``from x import (y, z)`` or
    ``from x import *`` statement.

    :ivar names: (list of :class:`alias`) names
    :ivar module: (string) module name, if any
    :ivar level: (integer) amount of dots before module name
    :ivar keyword_loc: location of ``from``
    :ivar dots_loc: location of dots, if any
    :ivar module_loc: location of module name, if any
    :ivar import_loc: location of ``import``
    :ivar lparen_loc: location of ``(``, if any
    :ivar rparen_loc: location of ``)``, if any
    """
    _fields = ('names', 'module', 'level')
    _locs = keywordloc._locs + ('dots_loc', 'module_loc', 'import_loc', 'lparen_loc', 'rparen_loc')
class Pass(stmt, keywordloc):
    """The ``pass`` statement."""
class Print(stmt, keywordloc):
    """
    The ``print >>x, y, z,`` statement.

    **Emitted until 3.0 or ``print_function`` future flag.**

    :ivar dest: (node) destination stream, if any
    :ivar values: (list of node) values to print
    :ivar nl: (boolean) whether to print newline after values
    :ivar dest_loc: location of ``>>``
    """
    _fields = ('dest', 'values', 'nl')
    _locs = keywordloc._locs + ('dest_loc',)
class Raise(stmt, keywordloc):
    """
    The ``raise exn, arg, traceback`` statement.

    :ivar type: (node) exception type or instance
    :ivar inst: (node) exception instance or argument list, if any
    :ivar tback: (node) traceback, if any
    """
    _fields = ('type', 'inst', 'tback')
class Return(stmt, keywordloc):
    """
    The ``return x`` statement.

    :ivar value: (node) return value, if any
    """
    _fields = ('value',)
class TryExcept(stmt, keywordloc):
    """
    The ``try:·  x·except y:·  z·else:·  t`` statement.

    **Emitted until 3.0.**

    :ivar body: (list of node) code to try
    :ivar handlers: (list of :class:`ExceptHandler`) exception handlers
    :ivar orelse: (list of node) code if no exception
    :ivar keyword_loc: location of ``try``
    :ivar try_colon_loc: location of ``:`` after ``try``
    :ivar else_loc: location of ``else``
    :ivar else_colon_loc: location of ``:`` after ``else``
    """
    _fields = ('body', 'handlers', 'orelse')
    _locs = keywordloc._locs + ('try_colon_loc', 'else_loc', 'else_colon_loc',)
class TryFinally(stmt, keywordloc):
    """
    The ``try:·  x·finally:·  y`` statement.

    **Emitted until 3.0.**

    :ivar body: (list of node) code to try
    :ivar finalbody: (list of node) code to finalize
    :ivar keyword_loc: location of ``try``
    :ivar try_colon_loc: location of ``:`` after ``try``
    :ivar finally_loc: location of ``finally``
    :ivar finally_colon_loc: location of ``:`` after ``finally``
    """
    _fields = ('body', 'finalbody')
    _locs = keywordloc._locs + ('try_colon_loc', 'finally_loc', 'finally_colon_loc',)
class While(stmt, keywordloc):
    """
    The ``while x:·  y·else:·  z`` statement.

    :ivar test: (node) condition
    :ivar body: (list of node) code for every iteration
    :ivar orelse: (list of node) code if empty
    :ivar keyword_loc: location of ``while``
    :ivar while_colon_loc: location of colon after ``while``
    :ivar else_loc: location of ``else``, if any
    :ivar else_colon_loc: location of colon after ``else``, if any
    """
    _fields = ('test', 'body', 'orelse')
    _locs = keywordloc._locs + ('while_colon_loc', 'else_loc', 'else_colon_loc')
class With(stmt, keywordloc):
    """
    The ``with x as y:·  z`` statement.

    :ivar context_expr: (node) context
    :ivar optional_vars: (assignable node) context binding
    :ivar body: (node) body
    :ivar keyword_loc: location of ``with``
    :ivar as_loc: location of ``as``, if any
    :ivar colon_loc: location of ``:``
    """
    _fields = ('context_expr', 'optional_vars', 'body')
    _locs = keywordloc._locs + ('as_loc', 'colon_loc')

class unaryop(AST, commonloc):
    """Base class for unary numeric and boolean operators."""
class Invert(unaryop):
    """The ``~`` operator."""
class Not(unaryop):
    """The ``not`` operator."""
class UAdd(unaryop):
    """The unary ``+`` operator."""
class USub(unaryop):
    """The unary ``-`` operator."""
