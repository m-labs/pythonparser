# coding:utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source, lexer, diagnostic, parser
import unittest

class ParserTestCase(unittest.TestCase):

    def parser_for(self, code):
        self.source_buffer = source.Buffer(code)
        self.lexer = lexer.Lexer(self.source_buffer, (2,7))
        self.parser = parser.Parser(self.lexer)

        old_next = self.lexer.next
        def lexer_next(**args):
            token = old_next(**args)
            # print(repr(token))
            return token
        self.lexer.next = lexer_next

        return self.parser

    def assertParses(self, ast, code):
        self.assertEqual(ast, self.parser_for(code).expr())

    def assertDiagnoses(self, code, diag):
        try:
            self.parser_for(code).expr()
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
        self.assertParses(1, "1")
        self.assertParses(1.0, "1.0")
        self.assertParses(1.0, "(1.0)")
        self.assertParses(1.0, "((1.0))")
        self.assertDiagnosesUnexpected("()", ")", (1, 2))

