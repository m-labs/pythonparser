# encoding: utf-8

"""
The :mod:`ast` module contains the classes comprising the Python abstract syntax tree.

Every node class is inherited from the corresponding Python :mod:`..ast` class,
if one exists in the version of Python that is running.

All attributes ending with ``loc`` contain instances of :class:`.source.Range`
or None. All attributes ending with ``_locs`` contain lists of instances of
:class:`.source.Range` or [].
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from .shim import ast
from .shim.ast import AST

# Location mixins

class commonloc(object):
    """
    A mixin common for all nodes.

    :cvar _fields: (tuple of strings) (defined in Python)
        names of all attributes with semantic values
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

class alias(commonloc, ast.alias):
    """
    An import alias, e.g. ``x as y``.

    :ivar name: (string) value to import
    :ivar asname: (string) name to add to the environment
    :ivar name_loc: location of name
    :ivar as_loc: location of ``as``
    :ivar asname_loc: location of asname
    """
    _locs = commonloc._locs + ('name_loc', 'as_loc', 'asname_loc')

class arguments(beginendloc, ast.arguments):
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
    _locs = beginendloc._locs + ('star_loc', 'vararg_loc', 'dstar_loc',
                                 'vararg_loc', 'kwarg_loc', 'equals_locs')

class boolop:
    """
    Base class for binary boolean operators.

    This class is unlike others in that it does not have the ``loc`` field.
    It serves only as an indicator of operation and corresponds to no source
    itself; locations are recorded in :class:`BoolOp`.
    """
    _locs = ()
class And(boolop, ast.And):
    """The ``and`` operator."""
class Or(boolop, ast.Or):
    """The ``or`` operator."""

class cmpop(commonloc):
    """Base class for comparison operators."""
class Eq(cmpop, ast.Eq):
    """The ``==`` operator."""
class Gt(cmpop, ast.Gt):
    """The ``>`` operator."""
class GtE(cmpop, ast.GtE):
    """The ``>=`` operator."""
class In(cmpop, ast.In):
    """The ``in`` operator."""
class Is(cmpop, ast.Is):
    """The ``is`` operator."""
class IsNot(cmpop, ast.IsNot):
    """The ``is not`` operator."""
class Lt(cmpop, ast.Lt):
    """The ``<`` operator."""
class LtE(cmpop, ast.LtE):
    """The ``<=`` operator."""
class NotEq(cmpop, ast.NotEq):
    """The ``!=`` (or deprecated ``<>``) operator."""
class NotIn(cmpop, ast.NotIn):
    """The ``not in`` operator."""

class comprehension(commonloc, ast.comprehension):
    """
    A single ``for`` list comprehension clause.

    :ivar target: (node) the variable(s) bound in comprehension body
    :ivar iter: (node) the expression being iterated
    :ivar ifs: (list of node) the ``if`` clauses
    :ivar for_loc: location of the ``for`` keyword
    :ivar in_loc: location of the ``in`` keyword
    :ivar if_locs: locations of ``if`` keywords
    """
    _locs = commonloc._locs + ('for_loc', 'in_loc', 'if_locs')

class excepthandler(commonloc):
    """Base class for the exception handler."""
class ExceptHandler(excepthandler, ast.ExceptHandler):
    """
    An exception handler, e.g. ``except x as y:·  z``.

    :ivar type: (node) type of handled exception, if any
    :ivar name: (assignable node) variable bound to exception, if any
    :ivar body: (list of node) code to execute when exception is caught
    :ivar except_loc: location of ``except``
    :ivar as_loc: location of ``as``, if any
    :ivar colon_loc: location of ``:``
    """
    _locs = excepthandler._locs + ('except_loc', 'as_loc', 'colon_loc')

class expr(commonloc):
    """Base class for expression nodes."""
class Attribute(expr, ast.Attribute):
    """
    An attribute access, e.g. ``x.y``.

    :ivar value: (node) left-hand side
    :ivar attr: (string) attribute name
    """
    _locs = expr._locs + ('dot_loc', 'attr_loc')
class BinOp(expr, ast.BinOp):
    """
    A binary operation, e.g. ``x + y``.

    :ivar left: (node) left-hand side
    :ivar op: (:class:`operator`) operator
    :ivar right: (node) right-hand side
    """
class BoolOp(expr, ast.BoolOp):
    """
    A boolean operation, e.g. ``x and y``.

    :ivar left: (node) left-hand side
    :ivar op: (:class:`boolop`) operator
    :ivar right: (node) right-hand side
    :ivar op_locs: locations of operators
    """
    _locs = expr._locs + ('op_locs',)
class Call(beginendloc, expr, ast.Call):
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
    _locs = beginendloc._locs + ('star_loc', 'dstar_loc')
class Compare(expr, ast.Compare):
    """
    A comparison operation, e.g. ``x < y`` or ``x < y > z``.

    :ivar left: (node) left-hand
    :ivar ops: (list of :class:`cmpop`) compare operators
    :ivar comparators: (list of node) compare values
    """
class Dict(beginendloc, expr, ast.Dict):
    """
    A dictionary, e.g. ``{x: y}``.

    :ivar keys: (list of node) keys
    :ivar values: (list of node) values
    :ivar colon_locs: ``:`` locations
    """
    _locs = beginendloc._locs + ('colon_locs',)
class DictComp(beginendloc, expr, ast.DictComp):
    """
    A list comprehension, e.g. ``{x: y for x,y in z}``.

    :ivar key: (node) key part of comprehension body
    :ivar value: (node) value part of comprehension body
    :ivar generators: (list of :class:`comprehension`) ``for`` clauses
    """
class GeneratorExp(beginendloc, expr, ast.GeneratorExp):
    """
    A generator expression, e.g. ``(x for x in y)``.

    :ivar elt: (node) expression body
    :ivar generators: (list of :class:`comprehension`) ``for`` clauses
    """
class IfExp(expr, ast.IfExp):
    """
    A conditional expression, e.g. ``x if y else z``.

    :ivar test: (node) condition
    :ivar body: (node) value if true
    :ivar orelse: (node) value if false
    :ivar if_loc: location of ``if``
    :ivar else_loc: location of ``else``
    """
    _locs = expr._locs + ('if_loc', 'else_loc')
class Lambda(expr, ast.Lambda):
    """
    A lambda expression, e.g. ``lambda x: x*x``.

    :ivar args: (:class:`arguments`) arguments
    :ivar body: (node) body
    :ivar lambda_loc: location of ``lambda``
    :ivar colon_loc: location of ``:``
    """
    _locs = expr._locs + ('lambda_loc', 'colon_loc')
class List(beginendloc, expr, ast.List):
    """
    A list, e.g. ``[x, y]``.

    :ivar elts: (list of node) elements
    """
class ListComp(beginendloc, expr, ast.ListComp):
    """
    A list comprehension, e.g. ``[x for x in y]``.

    :ivar elt: (node) comprehension body
    :ivar generators: (list of :class:`comprehension`) ``for`` clauses
    """
class Name(expr, ast.Name):
    """
    An identifier, e.g. ``x``.

    :ivar id: (string) name
    """
class Num(expr, ast.Num):
    """
    An integer, floating point or complex number, e.g. ``1``, ``1.0`` or ``1.0j``.

    :ivar n: (int, float or complex) value
    """
class Repr(beginendloc, expr, ast.Repr):
    """
    A repr operation, e.g. ``\`x\```

    :ivar value: (node) value
    """
class Set(beginendloc, expr, ast.Set):
    """
    A set, e.g. ``{x, y}``.

    :ivar elts: (list of node) elements
    """
class SetComp(beginendloc, expr, ast.ListComp):
    """
    A set comprehension, e.g. ``{x for x in y}``.

    :ivar elt: (node) comprehension body
    :ivar generators: (list of :class:`comprehension`) ``for`` clauses
    """
class Str(beginendloc, expr, ast.Str):
    """
    A string, e.g. ``"x"``.

    :ivar s: (string) value
    """
class Subscript(beginendloc, expr, ast.Subscript):
    """
    A subscript operation, e.g. ``x[1]``.

    :ivar value: (node) object being sliced
    :ivar slice: (:class:`slice`) slice
    """
class Tuple(beginendloc, expr, ast.Tuple):
    """
    A tuple, e.g. ``(x,)`` or ``x,y``.

    :ivar elts: (list of nodes) elements
    """
class UnaryOp(expr, ast.UnaryOp):
    """
    An unary operation, e.g. ``+x``.

    :ivar op: (:class:`unaryop`) operator
    :ivar operand: (node) operand
    """
class Yield(expr, ast.Yield):
    """
    A yield expression, e.g. ``(yield x)``.

    :ivar value: (node) yielded value
    :ivar yield_loc: location of ``yield``
    """
    _locs = expr._locs + ('yield_loc',)

# expr_context
#     AugLoad
#     AugStore
#     Del
#     Load
#     Param
#     Store

class keyword(commonloc, ast.keyword):
    """
    A keyword actual argument, e.g. in ``f(x=1)``.

    :ivar arg: (string) name
    :ivar value: (node) value
    :ivar equals_loc: location of ``=``
    """
    _locs = commonloc._locs + ('arg_loc', 'equals_loc')

class mod(commonloc):
    """Base class for modules (groups of statements)."""
class Expression(mod, ast.Expression):
    """A group of statements parsed as if for :func:`eval`."""
class Interactive(mod, ast.Interactive):
    """A group of statements parsed as if it was REPL input."""
class Module(mod, ast.Module):
    """A group of statements parsed as if it was a file."""
class Suite(mod, ast.Suite):
    """
    Doesn't appear to be used by Python; included for compatibility
    with :mod:`ast`.
    """

class operator(commonloc):
    """Base class for numeric binary operators."""
class Add(operator, ast.Add):
    """The ``+`` operator."""
class BitAnd(operator, ast.BitAnd):
    """The ``&`` operator."""
class BitOr(operator, ast.BitOr):
    """The ``|`` operator."""
class BitXor(operator, ast.BitXor):
    """The ``^`` operator."""
class Div(operator, ast.Div):
    """The ``\\`` operator."""
class FloorDiv(operator, ast.FloorDiv):
    """The ``\\\\`` operator."""
class LShift(operator, ast.LShift):
    """The ``<<`` operator."""
class Mod(operator, ast.Mod):
    """The ``%`` operator."""
class Mult(operator, ast.Mult):
    """The ``*`` operator."""
class Pow(operator, ast.Pow):
    """The ``**`` operator."""
class RShift(operator, ast.RShift):
    """The ``>>`` operator."""
class Sub(operator, ast.Sub):
    """The ``-`` operator."""

class slice(commonloc):
    """Base class for slice operations."""
class Ellipsis(slice, ast.Ellipsis):
    """The ellipsis, e.g. in ``x[...]``."""
class ExtSlice(slice, ast.ExtSlice):
    """
    The multiple slice, e.g. in ``x[0:1, 2:3]``.

    :ivar dims: (:class:`slice`) sub-slices
    """
class Index(slice, ast.Index):
    """
    The index, e.g. in ``x[1]``.

    :ivar value: (node) index
    """
class Slice(slice, ast.Slice):
    """
    The slice, e.g. in ``x[0:1]`` or ``x[0:1:2]``.

    :ivar lower: (node or None) lower bound
    :ivar upper: (node or None) upper bound
    :ivar step: (node or None) iteration step
    :ivar bound_colon_loc: location of first semicolon
    :ivar step_colon_loc: location of second semicolon
    """
    _locs = slice._locs + ('bound_colon_loc', 'step_colon_loc')

class stmt(commonloc):
    """Base class for statement nodes."""
class Assert(keywordloc, stmt, ast.Assert):
    """
    The ``assert x, msg`` statement.

    :ivar test: (node) condition
    :ivar msg: (node) message, if any
    """
class Assign(stmt, ast.Assign):
    """
    The ``=`` statement.

    :ivar targets: (list of assignable node) left-hand sides
    :ivar value: (node) right-hand side
    :ivar op_locs: location of equality signs corresponding to ``targets``
    """
    _locs = stmt._locs + ('op_locs',)
class AugAssign(stmt, ast.AugAssign):
    """
    The operator-assignment statement, e.g. ``+=``.

    :ivar target: (assignable node) left-hand side
    :ivar op: (:class:`ast.operator`) operator
    :ivar value: (node) right-hand side
    """
class Break(keywordloc, stmt, ast.Break):
    """The ``break`` statement."""
class ClassDef(keywordloc, stmt, ast.ClassDef):
    """
    The ``class x(y, z):·  t`` statement.

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
    _locs = keywordloc._locs + ('name_loc', 'lparen_loc', 'rparen_loc', 'colon_loc', 'at_locs')
class Continue(keywordloc, stmt, ast.Continue):
    """The ``continue`` statement."""
class Delete(keywordloc, stmt, ast.Delete):
    """
    The ``del x, y`` statement.

    :ivar targets: (list of :class:`Name`)
    """
class Exec(keywordloc, stmt, ast.Exec):
    """
    The ``exec code in locals, globals`` statement.

    :ivar body: (node) code
    :ivar locals: (node) locals
    :ivar globals: (node) globals
    :ivar keyword_loc: location of ``exec``
    :ivar in_loc: location of ``in``
    """
    _locs = keywordloc._locs + ('in_loc',)
class Expr(stmt, ast.Expr):
    """
    An expression in statement context. The value of expression is discarded.

    :ivar value: (:class:`expr`) value
    """
class For(keywordloc, stmt, ast.For):
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
    _locs = keywordloc._locs + ('in_loc', 'for_colon_loc', 'else_loc', 'else_colon_loc')
class FunctionDef(keywordloc, stmt, ast.FunctionDef):
    """
    The ``def f(x):·  y`` statement.

    :ivar name: (string) name
    :ivar args: (:class:`arguments`) formal arguments
    :ivar body: (list of node) body
    :ivar keyword_loc: location of ``def``
    :ivar decorator_list: (list of node) decorators
    :ivar name_loc: location of name
    :ivar colon_loc: location of ``:``, if any
    :ivar at_locs: locations of decorator ``@``
    """
    _locs = keywordloc._locs + ('name_loc', 'colon_loc', 'at_locs')
class Global(keywordloc, stmt, ast.Global):
    """
    The ``global x, y`` statement.

    :ivar names: (list of string) names
    :ivar name_locs: locations of names
    """
    _locs = keywordloc._locs + ('name_locs',)
class If(keywordloc, stmt, ast.If):
    """
    The ``if x:·  y·else:·  z`` or ``if x:·  y·elif: z·  t`` statement.

    :ivar test: (node) condition
    :ivar body: (list of node) code if true
    :ivar orelse: (list of node) code if false
    :ivar if_colon_loc: location of colon after ``if`` or ``elif``
    :ivar else_loc: location of ``else``, if any
    :ivar else_colon_loc: location of colon after ``else``, if any
    """
    _locs = keywordloc._locs + ('if_colon_loc', 'else_loc', 'else_colon_loc')
class Import(keywordloc, stmt, ast.Import):
    """
    The ``import x, y`` statement.

    :ivar names: (list of :class:`alias`) names
    """
class ImportFrom(keywordloc, stmt, ast.ImportFrom):
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
    _locs = keywordloc._locs + ('dots_loc', 'module_loc', 'import_loc', 'lparen_loc', 'rparen_loc')
class Pass(keywordloc, stmt, ast.Pass):
    """The ``pass`` statement."""
class Print(keywordloc, stmt, ast.Print):
    """
    The ``print >>x, y, z,`` statement.

    :ivar dest: (node) destination stream, if any
    :ivar values: (list of node) values to print
    :ivar nl: (boolean) whether to print newline after values
    :ivar dest_loc: location of ``>>``
    """
    _locs = keywordloc._locs + ('dest_loc',)
class Raise(keywordloc, stmt, ast.Raise):
    """
    The ``raise exn, arg, traceback`` statement.

    :ivar type: (node) exception type or instance
    :ivar inst: (node) exception instance or argument list, if any
    :ivar tback: (node) traceback, if any
    """
class Return(keywordloc, stmt, ast.Return):
    """The ``return x`` statement."""
class TryExcept(keywordloc, stmt, ast.TryExcept):
    """
    The ``try:·  x·except y:·  z·else:·  t`` statement.

    :ivar body: (list of node) code to try
    :ivar handlers: (list of :class:`ExceptHandler`) exception handlers
    :ivar orelse: (list of node) code if no exception
    :ivar keyword_loc: location of ``try``
    :ivar try_colon_loc: location of ``:`` after ``try``
    :ivar else_loc: location of ``else``
    :ivar else_colon_loc: location of ``:`` after ``else``
    """
    _locs = keywordloc._locs + ('try_colon_loc', 'else_loc', 'else_colon_loc',)
class TryFinally(keywordloc, stmt, ast.TryFinally):
    """
    The ``try:·  x·finally:·  y`` statement.

    :ivar body: (list of node) code to try
    :ivar finalbody: (list of node) code to finalize
    :ivar keyword_loc: location of ``try``
    :ivar try_colon_loc: location of ``:`` after ``try``
    :ivar finally_loc: location of ``finally``
    :ivar finally_colon_loc: location of ``:`` after ``finally``
    """
    _locs = keywordloc._locs + ('try_colon_loc', 'finally_loc', 'finally_colon_loc',)
class While(keywordloc, stmt, ast.While):
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
    _locs = keywordloc._locs + ('while_colon_loc', 'else_loc', 'else_colon_loc')
class With(keywordloc, stmt, ast.With):
    """
    The ``with x as y:·  z`` statement.

    :ivar context_expr: (node) context
    :ivar optional_vars: (assignable node) context binding
    :ivar body: (node) body
    :ivar keyword_loc: location of ``with``
    :ivar as_loc: location of ``as``, if any
    :ivar colon_loc: location of ``:``
    """
    _locs = keywordloc._locs + ('as_loc', 'colon_loc')

class unaryop(commonloc):
    """Base class for unary numeric and boolean operators."""
class Invert(unaryop, ast.Invert):
    """The ``~`` operator."""
class Not(unaryop, ast.Not):
    """The ``not`` operator."""
class UAdd(unaryop, ast.UAdd):
    """The unary ``+`` operator."""
class USub(unaryop, ast.USub):
    """The unary ``-`` operator."""
