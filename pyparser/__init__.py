from __future__ import absolute_import, division, print_function, unicode_literals
import sys, pyparser.source, pyparser.lexer, pyparser.parser

def parse(source, filename='<unknown>', mode='exec',
          flags=[], version=sys.version_info[0:2]):
    """
    Parse a string into an abstract syntax tree.
    This is the replacement for the built-in :meth:`..ast.parse`.

    :param str source: Source code in the correct encoding
    :param filename: Filename of the source (used in diagnostics)
    :param mode: Execution mode. Pass ``"exec"`` to parse a module,
        ``"single"`` to parse a single (interactive) statement,
        and ``"eval"`` to parse an expression. In the last two cases,
        ``source`` must be terminated with an empty line
        (i.e. end with ``"\\n\\n"``).
    :param flags: Future flags. Equivalent to ``from __future__ import <flags>``.
    :param int,int version: A tuple of the major and minor version
        of Python syntax to recognize.
    :return ast.AST: abstract syntax tree
    :raise diagnostic.DiagnosticException:
        if the source code is not well-formed
    """
    buffer = pyparser.source.Buffer(source, filename)
    lexer  = pyparser.lexer.Lexer(buffer, version)
    parser = pyparser.parser.Parser(lexer)

    parser.add_flags(flags)

    if mode == 'exec':
        return parser.file_input()
    elif mode == 'single':
        return parser.single_input()
    elif mode == 'eval':
        return parser.eval_input()
