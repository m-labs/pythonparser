# coding:utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source, lexer, diagnostic, coverage
from ..coverage import parser
import unittest

def tearDownModule():
    coverage.report(parser)

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
        self.assertEqual(ast, self.parser_for(code).file_input())

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

    def test_pass(self):
        self.assertParses(None, "pass\n")

