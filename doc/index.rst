PyParser documentation
======================

PyParser is a Python parser written specifically for use in tooling.
It parses source code into an AST that is a superset of Python's
built-in :mod:`ast` module, but returns precise location information
for every token.

:mod:`source` Module
--------------------

.. automodule:: pyparser.source
    :members:
    :show-inheritance:

:mod:`diagnostic` Module
------------------------

.. automodule:: pyparser.diagnostic
    :members:
    :show-inheritance:

:mod:`lexer` Module
-------------------

.. automodule:: pyparser.lexer
    :members:
    :show-inheritance:

:mod:`ast` Module
-----------------

.. automodule:: pyparser.ast
    :members: commonloc, beginendloc, keywordloc,
      alias,
      arguments,
      boolop, And, Or,
      cmpop, Eq, Gt, GtE, In, Is, IsNot, Lt, LtE, NotEq, NotIn,
      comprehension,
      expr, BinOp, BoolOp, Call, Compare, Dict, DictComp, GeneratorExp, IfExp, Lambda,
      List, ListComp, Name, Num, Repr, Set, SetComp, Str, Subscript, Tuple, UnaryOp, Yield,
      keyword,
      mod, Expression, Interactive, Module, Suite,
      operator, Add, BitAnd, BitOr, BitXor, Div, FloorDiv, LShift, Mod, Mult, Pow, RShift, Sub,
      slice, Ellipsis, ExtSlice, Index, Slice,
      stmt, Assert, Assign, AugAssign, Break, ClassDef, Continue, Delete, Exec, Expr, For,
      FunctionDef, Global, If, Import, ImportFrom, Pass, Print, Raise, Return, TryExcept,
      TryFinally, While, With,
      unaryop, Invert, Not, UAdd, USub
    :show-inheritance:
