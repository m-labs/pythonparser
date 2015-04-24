# coding:utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source, lexer, diagnostic, ast, coverage
from ..coverage import parser
import unittest

def tearDownModule():
    coverage.report(parser)

class ParserTestCase(unittest.TestCase):

    def parser_for(self, code, version=(2, 6)):
        code = code.replace("·", "\n")

        self.source_buffer = source.Buffer(code)
        self.lexer = lexer.Lexer(self.source_buffer, version)
        self.parser = parser.Parser(self.lexer)

        old_next = self.lexer.next
        def lexer_next(**args):
            token = old_next(**args)
            # print(repr(token))
            return token
        self.lexer.next = lexer_next

        return self.parser

    def flatten_ast(self, node):
        # Validate locs
        for attr in node.__dict__:
            if attr.endswith('_loc') or attr.endswith('_locs'):
                self.assertTrue(attr in node._locs)
        for loc in node._locs:
            self.assertTrue(loc in node.__dict__)

        flat_node = { 'ty': unicode(type(node).__name__) }
        for field in node._fields:
            value = getattr(node, field)
            if isinstance(value, ast.AST):
                value = self.flatten_ast(value)
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], ast.AST):
                value = map(self.flatten_ast, value)
            flat_node[unicode(field)] = value
        return flat_node

    def assertParses(self, expected_flat_ast, code, loc_matchers=""):
        ast = self.parser_for(code + "\n").file_input()
        flat_ast = self.flatten_ast(ast)
        self.assertEqual({'ty': 'Module', 'body': expected_flat_ast},
                         flat_ast)

    def assertParsesExpr(self, expected_flat_ast, code, loc_matchers=""):
        self.assertParses([{'ty': 'Expr', 'value': expected_flat_ast}],
                          code, loc_matchers)

    def assertDiagnoses(self, code, diag):
        try:
            self.parser_for(code).file_input()
            self.fail("Expected a diagnostic")
        except diagnostic.DiagnosticException as e:
            level, reason, args, loc = diag
            self.assertEqual(level, e.diagnostic.level)
            self.assertEqual(reason, e.diagnostic.reason)
            for key in args:
                self.assertEqual(args[key], e.diagnostic.arguments[key],
                                 "{{%s}}: \"%s\" != \"%s\"" %
                                    (key, args[key], e.diagnostic.arguments[key]))
            self.assertEqual(source.Range(self.source_buffer, *loc),
                             e.diagnostic.location)

    def assertDiagnosesUnexpected(self, code, err_token, loc):
        self.assertDiagnoses(code,
            ("error", "unexpected {actual}: expected {expected}", {'actual': err_token}, loc))

    def test_int(self):
        self.assertParsesExpr(
            {'ty': 'Num', 'n': 1},
            "1",
            "^ loc")

