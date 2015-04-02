from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source, lexer, diagnostic
import unittest

class LexerTestCase(unittest.TestCase):

    def assertLexesVersions(self, input, versions, *expected_tokens):
        for version in versions:
            tokens = expected_tokens
            self.buffer = source.Buffer(input)
            self.lexer = lexer.Lexer(self.buffer, version)
            for (range, token, data) in self.lexer:
                if len(tokens) < 2:
                    raise Exception("stray tokens: %s" % ((token,data),))
                expected_token, expected_data = tokens[:2]
                tokens = tokens[2:]
                self.assertEqual(expected_token, token)
                self.assertEqual(expected_data, data)
            self.assertEqual((), tokens)

    def assertDiagnosesVersions(self, input, versions, diag, *tokens):
        for version in versions:
            try:
                self.assertLexesVersions(input, [version], *tokens)
                self.fail("Expected a diagnostic")
            except diagnostic.DiagnosticException as e:
                level, message, loc = diag[0]
                self.assertEqual(level, e.diagnostic.level)
                self.assertEqual(message, e.diagnostic.message())
                self.assertEqual(source.Range(self.buffer, *loc),
                                 e.diagnostic.location)

    VERSIONS = [(2,6), (3,0), (3,1)]

    def assertLexes(self, input, *tokens):
        self.assertLexesVersions(input, self.VERSIONS, *tokens)

    def assertDiagnoses(self, input, diag, *tokens):
        self.assertDiagnosesVersions(input, self.VERSIONS, diag, *tokens)

    def test_empty(self):
        self.assertLexes("")

    def test_newline(self):
        self.assertLexes("\n",
                         "newline", None)
        self.assertLexes("\r\n",
                         "newline", None)
        self.assertLexes("\r",
                         "newline", None)
        self.assertLexes("\\\n")

    def test_comment(self):
        self.assertLexes("# foo")
        self.assertEqual([(source.Range(self.buffer, 0, 5), "# foo")],
                         self.lexer.comments)

    def test_float(self):
        self.assertLexes("0.0",
                         "float", 0.0)
        self.assertLexes(".0",
                         "float", 0.0)
        self.assertLexes("0.",
                         "float", 0.0)
        self.assertLexes("0.0e0",
                         "float", 0.0)
        self.assertLexes(".0e0",
                         "float", 0.0)
        self.assertLexes("0.e0",
                         "float", 0.0)
        self.assertLexes("0e0",
                         "float", 0.0)
        self.assertLexes("0e00",
                         "float", 0.0)
        self.assertLexes("0e+0",
                         "float", 0.0)
        self.assertLexes("0e-0",
                         "float", 0.0)

    def test_complex(self):
        self.assertLexes("1e+1j",
                         "complex", 10j)
        self.assertLexes("10j",
                         "complex", 10j)

    def test_integer(self):
        self.assertLexes("0",
                         "int", 0)
        self.assertLexes("123",
                         "int", 123)
        self.assertLexes("0o123",
                         "int", 0o123)
        self.assertLexes("0x123af",
                         "int", 0x123af)
        self.assertLexes("0b0101",
                         "int", 0b0101)

    def test_integer_py3(self):
        self.assertLexesVersions(
                         "0123", [(2,6)],
                         "int", 83)
        self.assertLexesVersions(
                         "123L", [(2,6)],
                         "int", 123)
        self.assertLexesVersions(
                         "123l", [(2,6)],
                         "int", 123)

        self.assertDiagnosesVersions(
                         "0123", [(3,0)],
                         [("error", "in Python 3, decimal literals must not start with a zero", (0, 1))],
                         "int", 83)
        self.assertDiagnosesVersions(
                         "123L", [(3,0)],
                         [("error", "in Python 3, long integer literals were removed", (3, 4))],
                         "int", 123)
        self.assertDiagnosesVersions(
                         "123l", [(3,0)],
                         [("error", "in Python 3, long integer literals were removed", (3, 4))],
                         "int", 123)

    def test_string_literal(self):
        self.assertLexes("\"",
                         "\"", "")
        self.assertLexes("u\"",
                         "\"", "u")
        self.assertLexes("ur\"",
                         "\"", "ur")
        self.assertLexes("UR\"",
                         "\"", "ur")

        self.assertLexes("'''",
                         "'''", "")
        self.assertLexes("\"\"\"",
                         "\"\"\"", "")

    def test_identifier(self):
        self.assertLexes("a",
                         "ident", "a")
        self.assertLexes("andi",
                         "ident", "andi")

    def test_keywords(self):
        self.assertLexes("/",
                         "/", None)
        self.assertLexes("//",
                         "//", None)
        self.assertLexes("//=",
                         "//=", None)
        self.assertLexes("and",
                         "and", None)

        self.assertLexesVersions(
                         "<>", [(2,6),(3,1)],
                         "<>", None)
        self.assertLexesVersions(
                         "<>", [(3,0)],
                         "<", None,
                         ">", None)

    def test_implicit_joining(self):
        self.assertLexes("[1,\n2]",
                         "[", None,
                         "int", 1,
                         ",", None,
                         "int", 2,
                         "]", None)

    def test_diag_unrecognized(self):
        self.assertDiagnoses(
                         "$",
                         [("fatal", "unexpected '$'", (0, 1))])

    def test_diag_delim_mismatch(self):
        self.assertDiagnoses(
                         "[)",
                         [("fatal", "mismatched ')'", (1, 2))],
                         "[", None)

"""
    def test_(self):
        self.assertLexes("",
                         )
"""
