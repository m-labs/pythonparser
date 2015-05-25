PythonParser documentation
==========================

PythonParser is a Python parser written specifically for use in tooling.
It parses source code into an AST that is a superset of Python's
built-in :mod:`ast` module, but returns precise location information
for every token.

The most useful APIs of PyParser are :meth:`pythonparser.parse`,
which parses the source, :mod:`pythonparser.ast`, which provides access
to the semantic and location information in the AST, and
:mod:`pythonparser.diagnostic.Engine`, which provides a unified way
to report errors and warnings both to PyParser itself and to any
code that consumes the ASTs. The :class:`pythonparser.source.Range` class,
instances of which are contained in the AST, allows to extract
location information in various convenient formats.

The :mod:`pythonparser.algorithm` module contains some useful algorithms
to manipulate ASTs.

If a consumer of ASTs wishes to modify the original source without
losing formatting, it can use :class:`pythonparser.source.Rewriter`
to insert code fragments around or instead of a known
:class:`pythonparser.source.Range`. If the AST is not expected to
change after the modification, it is recommended to re-parse
the result and compare it to the original AST using
:meth:`pythonparser.algorithm.compare`.

For some applications, e.g. syntax highlighting,
:class:`pythonparser.lexer.Lexer` will be able to provide a raw
stream of tokens.

PyParser is licensed under 3-clause BSD.

:mod:`pythonparser` Module
--------------------------

.. automodule:: pythonparser
    :members:

:mod:`pythonparser.source` Module
---------------------------------

.. automodule:: pythonparser.source
    :members:
    :show-inheritance:

:mod:`pythonparser.diagnostic` Module
-------------------------------------

.. automodule:: pythonparser.diagnostic
    :members:
    :show-inheritance:

:mod:`pythonparser.lexer` Module
--------------------------------

.. automodule:: pythonparser.lexer
    :members:
    :show-inheritance:

:mod:`pythonparser.ast` Module
------------------------------

.. automodule:: pythonparser.ast
    :members: commonloc, beginendloc, keywordloc,
      AST,
      alias,
      arg,
      arguments,
      boolop, And, Or,
      cmpop, Eq, Gt, GtE, In, Is, IsNot, Lt, LtE, NotEq, NotIn,
      comprehension,
      excepthandler, ExceptHandler,
      expr, BinOp, BoolOp, Call, Compare, Dict, DictComp, Ellipsis, GeneratorExp, IfExp, Lambda,
      List, ListComp, Name, Num, Repr, Set, SetComp, Str, Subscript, Tuple, UnaryOp,
      Yield, YieldFrom,
      keyword,
      mod, Expression, Interactive, Module,
      operator, Add, BitAnd, BitOr, BitXor, Div, FloorDiv, LShift, MatMult, Mod, Mult,
      Pow, RShift, Sub,
      slice, ExtSlice, Index, Slice,
      stmt, Assert, Assign, AugAssign, Break, ClassDef, Continue, Delete, Exec, Expr, For,
      FunctionDef, Global, If, Import, ImportFrom, Nonlocal, Pass, Print, Raise, Return,
      Try, While, With,
      unaryop, Invert, Not, UAdd, USub,
      withitem
    :show-inheritance:

:mod:`pythonparser.algorithm` Module
------------------------------------

.. automodule:: pythonparser.algorithm
    :members:
    :show-inheritance:
