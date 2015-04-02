import unittest
import pyparser

class LexerTestCase(unittest.TestCase):

    def assertLexesVersion(self, input, version, *tokens):
        self.buffer = pyparser.source.Buffer(unicode(input))
        self.lexer = pyparser.lexer.Lexer(self.buffer, version)
        for (range, token, data) in self.lexer:
            if len(tokens) < 2:
                raise Exception(u"stray tokens: %s" % unicode((token,data)))
            expected_token, expected_data = tokens[:2]
            tokens = tokens[2:]
            self.assertEqual(unicode(expected_token), token)
            self.assertEqual(expected_data, data)
        self.assertEqual((), tokens)

    def assertLexes(self, input, *tokens):
        for version in [(2,6), (3,0), (3,1)]:
            self.assertLexesVersion(input, version, *tokens)

    def test_empty(self):
        self.assertLexes("")

    def test_newline(self):
        self.assertLexes("\n",
                         'newline', None)

    def test_comment(self):
        self.assertLexes("# foo")
        self.assertEqual([(pyparser.source.Range(self.buffer, 0, 5), "# foo")],
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
        self.assertLexes("123",
                         'int', 123)
        self.assertLexes("0123",
                         'int', 0123)
        self.assertLexes("0o123",
                         'int', 0o123)
        self.assertLexes("0x123af",
                         'int', 0x123af)
        self.assertLexes("0b0101",
                         'int', 0b0101)
        self.assertLexes("123L",
                         'int', 123L)
        self.assertLexes("123l",
                         'int', 123l)

    def test_string_literal(self):
        self.assertLexes("'",
                         "'", "")
        self.assertLexes("u'",
                         "'", "u")
        self.assertLexes("ur'",
                         "'", "ur")
        self.assertLexes("UR'",
                         "'", "ur")

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

        self.assertLexesVersion("<>", (2,6),
                                "<>", None)
        self.assertLexesVersion("<>", (3,0),
                                "<", None,
                                ">", None)
        self.assertLexesVersion("<>", (3,1),
                                "<>", None)

"""
    def test_(self):
        self.assertLexes("",
                         )
"""
