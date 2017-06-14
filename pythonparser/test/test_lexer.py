# coding:utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
from . import test_utils
from .. import source, lexer, diagnostic
import unittest

BytesOnly = test_utils.BytesOnly
UnicodeOnly = test_utils.UnicodeOnly

class LexerTestCase(unittest.TestCase):

    def assertLexesVersions(self, input, versions, *expected_tokens, **kwargs):
        expect_trailing_nl = True
        if "expect_trailing_nl" in kwargs:
            expect_trailing_nl = kwargs.pop("expect_trailing_nl")
        for version in versions:
            tokens = expected_tokens
            if expect_trailing_nl:
                tokens += ("newline", None)
            self.buffer = source.Buffer(input)
            self.engine = diagnostic.Engine(all_errors_are_fatal=True)
            self.engine.render_diagnostic = lambda diag: None
            self.lexer = lexer.Lexer(self.buffer, version, self.engine, **kwargs)
            for token in self.lexer:
                if len(tokens) < 2:
                    raise Exception("stray tokens: %s" % repr(token))
                expected_kind, expected_value = tokens[:2]
                tokens = tokens[2:]
                self.assertEqual(expected_kind, token.kind)
                self.assertEqual(expected_value, token.value)
            self.assertEqual((), tokens)

    def assertDiagnosesVersions(self, input, versions, diag, *tokens):
        for version in versions:
            try:
                self.assertLexesVersions(input, [version], *tokens)
                self.fail("Expected a diagnostic")
            except diagnostic.Error as e:
                level, message, loc = diag[0]
                self.assertEqual(level, e.diagnostic.level)
                self.assertEqual(message, e.diagnostic.message())
                self.assertEqual(source.Range(self.buffer, *loc),
                                 e.diagnostic.location)

    VERSIONS = [(2,6), (3,0), (3,1)]

    def assertLexes(self, input, *tokens, **kwargs):
        self.assertLexesVersions(input, self.VERSIONS, *tokens, **kwargs)

    def assertDiagnoses(self, input, diag, *tokens):
        self.assertDiagnosesVersions(input, self.VERSIONS, diag, *tokens)

    def test_empty(self):
        self.assertLexes("", expect_trailing_nl=False)

    def test_newline(self):
        self.assertLexes("x\n",
                         "ident",   "x")
        self.assertLexes("x\r\n",
                         "ident",   "x")
        self.assertLexes("x\r",
                         "ident",   "x")
        self.assertLexes("x\\\n",
                         "ident",   "x")

        self.assertLexes("x\n\n",
                         "ident",   "x")

    def test_comment(self):
        self.assertLexes("# foo", expect_trailing_nl=False)
        self.assertEqual(source.Range(self.buffer, 0, 5),
                         self.lexer.comments[0].loc)
        self.assertEqual("# foo",
                         self.lexer.comments[0].text)

        self.assertLexes("class x:\n  # foo\n  pass",
                         "class",   None,
                         "ident",   "x",
                         ":",       None,
                         "newline", None,
                         "indent",  None,
                         "pass",    None,
                         "newline", None,
                         "dedent",  None,
                         expect_trailing_nl=False)

        self.assertLexes("class x:\n    # foo\n  pass",
                         "class",   None,
                         "ident",   "x",
                         ":",       None,
                         "newline", None,
                         "indent",  None,
                         "pass",    None,
                         "newline", None,
                         "dedent",  None,
                         expect_trailing_nl=False)

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
        for version in self.VERSIONS:
            if version < (3,):
                str_type = BytesOnly
            else:
                str_type = UnicodeOnly
            self.assertLexesVersions("''", [version],
                                     "strbegin", "",
                                     "strdata",  str_type(""),
                                     "strend",   None)
            self.assertLexesVersions("''''''", [version],
                                     "strbegin", "",
                                     "strdata",  str_type(""),
                                     "strend",   None)
            self.assertLexesVersions("\"\"", [version],
                                     "strbegin", "",
                                     "strdata",  str_type(""),
                                     "strend",   None)
            self.assertLexesVersions("\"\"\"\"\"\"", [version],
                                     "strbegin", "",
                                     "strdata",  str_type(""),
                                     "strend",   None)

            self.assertLexesVersions("'x'", [version],
                                     "strbegin", "",
                                     "strdata",  str_type("x"),
                                     "strend",   None)

            self.assertLexesVersions("'''\n'''", [version],
                                     "strbegin", "",
                                     "strdata",  str_type("\n"),
                                     "strend",   None)

            self.assertLexesVersions("'''\n'''", [version],
                                     "strbegin", "",
                                     "strdata",  str_type("\n"),
                                     "strend",   None)

            self.assertLexesVersions(r"'\0 \10 \010'", [version],
                                     "strbegin", "",
                                     "strdata",  str_type("\x00 \x08 \x08"),
                                     "strend",   None)

        self.assertLexesVersions(r"b'\xc3\xa7'", [(2,7), (3,0), (3,1)],
                                 "strbegin", "b",
                                 "strdata",  BytesOnly(b"\xc3\xa7"),
                                 "strend",   None)

        self.assertLexesVersions(b"# coding: koi8-r\nb'\xc3\xa7'", [(2,7), (3,0), (3,1)],
                                 "strbegin", "b",
                                 "strdata",  BytesOnly(b"\xc3\xa7"),
                                 "strend",   None)

        self.assertLexesVersions(b"# coding: koi8-r\n'\xc3\xa7'", [(3,0), (3,1)],
                                 "strbegin", "",
                                 "strdata",  UnicodeOnly("\u0446\u2556"),
                                 "strend",   None)

        self.assertLexesVersions(b"# coding: koi8-r\nu'\xc3\xa7'", [(2,7)],
                                 "strbegin", "u",
                                 "strdata",  UnicodeOnly("\u0446\u2556"),
                                 "strend",   None)

        self.assertDiagnoses(
                         "'",
                         [("fatal", "unterminated string", (0, 1))])

    def test_string_literal_kinds(self):
        self.assertDiagnosesVersions(
                         "u''", [(3,0)],
                         [("error", "string prefix 'u' is not available in Python 3.0", (0, 2))])

    def assertLexesEscape(self, mode, src, val):
        self.assertLexesVersions(
                         mode + "'" + src + "'", [(3,4)],
                         "strbegin", mode,
                         "strdata",  val,
                         "strend",   None)

    def test_escape_clike(self):
        for chr, val in [ ("\\\n", ""),
                (r"\\", "\\"), (r"\'", "'"),  (r"\"", "\""),
                (r"\a", "\a"), (r"\b", "\b"), (r"\f", "\f"), (r"\n", "\n"),
                (r"\r", "\r"), (r"\t", "\t"), (r"\v", "\v"),
                (r"\x53", "S"), (r"\123", "S")]:
            self.assertLexesEscape("b", chr, BytesOnly(val))
            self.assertLexesEscape("u", chr, UnicodeOnly(val))
            self.assertLexesEscape("", chr, UnicodeOnly(val))
            self.assertLexesEscape("r", chr, UnicodeOnly(chr))
            self.assertLexesEscape("br", chr, BytesOnly(chr))

        self.assertLexesEscape("r", "\\\"", UnicodeOnly("\\\""))

    def test_escape_unicode(self):
        self.assertLexesEscape("u", "\\u044b", "ы")
        self.assertLexesEscape("u", "\\U0000044b", "ы")
        self.assertLexesEscape("u", "\\N{LATIN CAPITAL LETTER A}", "A")

        self.assertDiagnosesVersions(
                         "u'\\U11111111'", [(3,4)],
                         [("error", "unicode character out of range", (2, 12))])
        self.assertDiagnosesVersions(
                         "u'\\N{foobar}'", [(3,4)],
                         [("error", "unknown unicode character name", (2, 12))])

    def test_identifier(self):
        self.assertLexes("a",
                         "ident", "a")
        self.assertLexes("andi",
                         "ident", "andi")
        self.assertLexesVersions(
                         "ышка", [(3,0)],
                         "ident", "ышка")
        self.assertLexesVersions(
                         "ышкаs", [(3,0)],
                         "ident", "ышкаs")
        self.assertLexesVersions(
                         "sышка", [(3,0)],
                         "ident", "sышка")

        self.assertDiagnosesVersions(
                         "ышка", [(2,7)],
                         [("error", "in Python 2, Unicode identifiers are not allowed", (0, 4))])

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

    def test_indent(self):
        self.assertLexes("  x\n  x\n    x\nx\n",
                         "indent",  None,
                         "ident",   "x",
                         "newline", None,
                         "ident",   "x",
                         "newline", None,
                         "indent",  None,
                         "ident",   "x",
                         "newline", None,
                         "dedent",  None,
                         "dedent",  None,
                         "ident",   "x")

        self.assertDiagnoses(
                         "  x\n    x\n x\n",
                         [("fatal", "inconsistent indentation", (11, 11))],
                         "indent",  None,
                         "ident",   "x",
                         "newline", None,
                         "indent",  None,
                         "ident",   "x")

        self.assertLexesVersions(
                         "    \tx\n\tx\n        x\n", [(2,7)],
                         "indent",  None,
                         "ident",   "x",
                         "newline", None,
                         "ident",   "x",
                         "newline", None,
                         "ident",   "x",
                         "newline", None,
                         "dedent",  None,
                         expect_trailing_nl=False)

        self.assertDiagnosesVersions(
                         "    \tx\n\tx", [(3,0)],
                         [("error", "inconsistent use of tabs and spaces in indentation", (8, 8))],
                         "indent",  None,
                         "ident",   "x")

    def test_eof(self):
        self.assertLexes("\t",
                         "indent",  None,
                         "newline", None,
                         "dedent",  None,
                         expect_trailing_nl=False)

    def test_stmt_at_eof(self):
        self.assertLexes("x",
                         "ident",   "x")

    def test_interactive(self):
        self.assertLexes("x\n\n",
                         "ident",   "x",
                         "newline", None,
                         interactive=True)

    def test_diag_unrecognized(self):
        self.assertDiagnoses(
                         "$",
                         [("fatal", "unexpected '$'", (0, 1))])

    def test_diag_delim_mismatch(self):
        self.assertDiagnoses(
                         "[)",
                         [("fatal", "mismatched ')'", (1, 2))],
                         "[", None)

        self.assertDiagnoses(
                         ")",
                         [("fatal", "mismatched ')'", (0, 1))])

"""
    def test_(self):
        self.assertLexes("",
                         )
"""
