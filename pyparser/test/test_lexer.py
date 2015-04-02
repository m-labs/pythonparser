from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source, lexer, diagnostic
import unittest

class LexerTestCase(unittest.TestCase):

    def assertLexesVersion(self, input, version, *tokens):
        self.buffer = source.Buffer(input)
        self.lexer = lexer.Lexer(self.buffer, version)
        for (range, token, data) in self.lexer:
            if len(tokens) < 2:
                raise Exception(u"stray tokens: %s" % (token,data))
            expected_token, expected_data = tokens[:2]
            tokens = tokens[2:]
            self.assertEqual(expected_token, token)
            self.assertEqual(expected_data, data)
        self.assertEqual((), tokens)

    def assertDiagnosesVersion(self, input, version, diag, *tokens):
        try:
            self.assertLexesVersion(input, version, *tokens)
        except diagnostic.DiagnosticException as e:
            reason, args, loc = diag
            self.assertEqual(reason, e.diagnostic.reason)
            self.assertEqual(args, e.diagnostic.arguments)
            self.assertEqual(source.Range(self.buffer, *loc),
                             e.diagnostic.location)
            return
        self.assert_("Expected a diagnostic")

    VERSIONS = [(2,6), (3,0), (3,1)]

    def assertLexes(self, input, *tokens):
        for version in self.VERSIONS:
            self.assertLexesVersion(input, version, *tokens)

    def assertDiagnoses(self, input, diag, *tokens):
        for version in self.VERSIONS:
            self.assertDiagnosesVersion(input, version, diag, *tokens)

    def test_empty(self):
        self.assertLexes(u"")

    def test_newline(self):
        self.assertLexes(u"\n",
                         u"newline", None)
        self.assertLexes(u"\r\n",
                         u"newline", None)
        self.assertLexes(u"\r",
                         u"newline", None)
        self.assertLexes(u"\\\n")

    def test_comment(self):
        self.assertLexes(u"# foo")
        self.assertEqual([(source.Range(self.buffer, 0, 5), "# foo")],
                         self.lexer.comments)

    def test_float(self):
        self.assertLexes(u"0.0",
                         u"float", 0.0)
        self.assertLexes(u".0",
                         u"float", 0.0)
        self.assertLexes(u"0.",
                         u"float", 0.0)
        self.assertLexes(u"0.0e0",
                         u"float", 0.0)
        self.assertLexes(u".0e0",
                         u"float", 0.0)
        self.assertLexes(u"0.e0",
                         u"float", 0.0)
        self.assertLexes(u"0e0",
                         u"float", 0.0)
        self.assertLexes(u"0e00",
                         u"float", 0.0)
        self.assertLexes(u"0e+0",
                         u"float", 0.0)
        self.assertLexes(u"0e-0",
                         u"float", 0.0)

    def test_complex(self):
        self.assertLexes(u"1e+1j",
                         u"complex", 10j)
        self.assertLexes(u"10j",
                         u"complex", 10j)

    def test_integer(self):
        self.assertLexes(u"123",
                         u"int", 123)
        self.assertLexes(u"0123",
                         u"int", 83)
        self.assertLexes(u"0o123",
                         u"int", 0o123)
        self.assertLexes(u"0x123af",
                         u"int", 0x123af)
        self.assertLexes(u"0b0101",
                         u"int", 0b0101)
        self.assertLexes(u"123L",
                         u"int", 123)
        self.assertLexes(u"123l",
                         u"int", 123)

    def test_string_literal(self):
        self.assertLexes(u"\"",
                         u"\"", "")
        self.assertLexes(u"u\"",
                         u"\"", "u")
        self.assertLexes(u"ur\"",
                         u"\"", "ur")
        self.assertLexes(u"UR\"",
                         u"\"", "ur")

        self.assertLexes(u"'''",
                         u"'''", "")
        self.assertLexes(u"\"\"\"",
                         u"\"\"\"", "")

    def test_identifier(self):
        self.assertLexes(u"a",
                         u"ident", "a")
        self.assertLexes(u"andi",
                         u"ident", "andi")

    def test_keywords(self):
        self.assertLexes(u"/",
                         u"/", None)
        self.assertLexes(u"//",
                         u"//", None)
        self.assertLexes(u"//=",
                         u"//=", None)
        self.assertLexes(u"and",
                         u"and", None)

        self.assertLexesVersion(u"<>", (2,6),
                                u"<>", None)
        self.assertLexesVersion(u"<>", (3,0),
                                u"<", None,
                                u">", None)
        self.assertLexesVersion(u"<>", (3,1),
                                u"<>", None)

    def test_implicit_joining(self):
        self.assertLexes(u"[1,\n2]",
                         u"[", None,
                         u"int", 1,
                         u",", None,
                         u"int", 2,
                         u"]", None)

    def test_diag_unrecognized(self):
        self.assertDiagnoses(u"$",
                             (u"unexpected {character}", {"character": "'$'"}, (0, 1)))

    def test_diag_delim_mismatch(self):
        self.assertDiagnoses(u"[)",
                             (u"mismatched '{delimiter}'", {"delimiter": u")"}, (1, 2)),
                             u"[", None)

"""
    def test_(self):
        self.assertLexes(u"",
                         )
"""
