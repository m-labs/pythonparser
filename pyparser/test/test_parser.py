# coding:utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source, lexer, diagnostic, ast, coverage
from ..coverage import parser
import unittest, re

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

    _loc_re  = re.compile(r"\s*([~^]+)\s+([a-z_0-9.]+)")
    _path_re = re.compile(r"(([a-z_]+)|([0-9]+))(.)?")

    def match_loc(self, ast, matcher, root=lambda x: x):
        ast = root(ast)

        matcher_pos = 0
        while matcher_pos < len(matcher):
            matcher_match = self._loc_re.match(matcher, matcher_pos)
            if matcher_match is None:
                raise Exception("invalid location matcher %s" % matcher[matcher_pos:])

            range = source.Range(self.source_buffer,
                matcher_match.start(1) - matcher_pos,
                matcher_match.end(1) - matcher_pos)
            path = matcher_match.group(2)

            path_pos = 0
            obj = ast
            while path_pos < len(path):
                path_match = self._path_re.match(path, path_pos)
                if path_match is None:
                    raise Exception("invalid location matcher path %s" % path)

                path_field = path_match.group(1)
                path_index = path_match.group(2)
                path_last  = not path_match.group(3)

                if path_field is not None:
                    obj = getattr(obj, path_field)
                elif path_index is not None:
                    obj = obj[int(path_index)]

                if path_last:
                    self.assertEqual(obj, range)

                path_pos = path_match.end(0)

            matcher_pos = matcher_match.end(0)

    def assertParsesGen(self, expected_flat_ast, code):
        ast = self.parser_for(code + "\n").file_input()
        flat_ast = self.flatten_ast(ast)
        self.assertEqual({'ty': 'Module', 'body': expected_flat_ast},
                         flat_ast)
        return ast

    def assertParsesSuite(self, expected_flat_ast, code, loc_matcher=""):
        ast = self.assertParsesGen(expected_flat_ast, code)
        self.match_loc(ast, loc_matcher, lambda x: x.body)

    def assertParsesExpr(self, expected_flat_ast, code, loc_matcher=""):
        ast = self.assertParsesGen([{'ty': 'Expr', 'value': expected_flat_ast}], code)
        self.match_loc(ast, loc_matcher, lambda x: x.body[0].value)

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

    #
    # LITERALS
    #

    def test_int(self):
        self.assertParsesExpr(
            {'ty': 'Num', 'n': 1},
            "1",
            "^ loc")

    def test_float(self):
        self.assertParsesExpr(
            {'ty': 'Num', 'n': 1.0},
            "1.0",
            "~~~ loc")

    def test_complex(self):
        self.assertParsesExpr(
            {'ty': 'Num', 'n': 1j},
            "1j",
            "~~ loc")

    def test_string(self):
        self.assertParsesExpr(
            {'ty': 'Str', 's': 'foo'},
            "'foo'",
            "~~~~~ loc"
            "^ begin_loc"
            "    ^ end_loc")

