# coding:utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source, lexer, diagnostic, ast, coverage
from ..coverage import parser
import unittest, sys, re

if sys.version_info >= (3,):
    def unicode(x): return x

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
                self.assertTrue(attr in node._locs,
                                "%s not in %s._locs" % (attr, repr(node)))
        for loc in node._locs:
            self.assertTrue(loc in node.__dict__)

        flat_node = { 'ty': unicode(type(node).__name__) }
        for field in node._fields:
            value = getattr(node, field)
            if isinstance(value, ast.AST):
                value = self.flatten_ast(value)
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], ast.AST):
                value = list(map(self.flatten_ast, value))
            flat_node[unicode(field)] = value
        return flat_node

    _loc_re  = re.compile(r"\s*([~^]+)\s+([a-z_0-9.]+)")
    _path_re = re.compile(r"(([a-z_]+)|([0-9]+))(\.)?")

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

                path_field = path_match.group(2)
                path_index = path_match.group(3)
                path_last  = not path_match.group(4)

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

    def test_ident(self):
        self.assertParsesExpr(
            {'ty': 'Name', 'id': 'foo', 'ctx': None},
            "foo",
            "~~~ loc")

    #
    # OPERATORS
    #

    ast_1 = {'ty': 'Num', 'n': 1}

    def test_unary(self):
        self.assertParsesExpr(
            {'ty': 'UnaryOp', 'op': {'ty': 'UAdd'}, 'operand': self.ast_1},
            "+1",
            "~~ loc"
            "~ op.loc")

        self.assertParsesExpr(
            {'ty': 'UnaryOp', 'op': {'ty': 'USub'}, 'operand': self.ast_1},
            "-1",
            "~~ loc"
            "~ op.loc")

        self.assertParsesExpr(
            {'ty': 'UnaryOp', 'op': {'ty': 'Invert'}, 'operand': self.ast_1},
            "~1",
            "~~ loc"
            "~ op.loc")

    def test_binary(self):
        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'Pow'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 ** 1",
            "~~~~~~ loc"
            "  ~~ op.loc")

        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'Mult'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 * 1",
            "~~~~~ loc"
            "  ^ op.loc")

        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'Div'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 / 1",
            "~~~~~ loc"
            "  ^ op.loc")

        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'Mod'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 % 1",
            "~~~~~ loc"
            "  ^ op.loc")

        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'FloorDiv'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 // 1",
            "~~~~~~ loc"
            "  ~~ op.loc")

        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'Add'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 + 1",
            "~~~~~ loc"
            "  ^ op.loc")

        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'Sub'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 - 1",
            "~~~~~ loc"
            "  ^ op.loc")

    def test_bitwise(self):
        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'LShift'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 << 1",
            "~~~~~~ loc"
            "  ~~ op.loc")

        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'RShift'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 >> 1",
            "~~~~~~ loc"
            "  ~~ op.loc")

        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'BitAnd'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 & 1",
            "~~~~~ loc"
            "  ^ op.loc")

        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'BitOr'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 | 1",
            "~~~~~ loc"
            "  ^ op.loc")

        self.assertParsesExpr(
            {'ty': 'BinOp', 'op': {'ty': 'BitXor'}, 'left': self.ast_1, 'right': self.ast_1},
            "1 ^ 1",
            "~~~~~ loc"
            "  ^ op.loc")

    def test_compare(self):
        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'Lt'}],
             'left': self.ast_1, 'comparators': [self.ast_1]},
            "1 < 1",
            "~~~~~ loc"
            "  ^ ops.0.loc")

        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'LtE'}],
             'left': self.ast_1, 'comparators': [self.ast_1]},
            "1 <= 1",
            "~~~~~~ loc"
            "  ~~ ops.0.loc")

        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'Gt'}],
             'left': self.ast_1, 'comparators': [self.ast_1]},
            "1 > 1",
            "~~~~~ loc"
            "  ^ ops.0.loc")

        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'GtE'}],
             'left': self.ast_1, 'comparators': [self.ast_1]},
            "1 >= 1",
            "~~~~~~ loc"
            "  ~~ ops.0.loc")

        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'Eq'}],
             'left': self.ast_1, 'comparators': [self.ast_1]},
            "1 == 1",
            "~~~~~~ loc"
            "  ~~ ops.0.loc")

        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'NotEq'}],
             'left': self.ast_1, 'comparators': [self.ast_1]},
            "1 != 1",
            "~~~~~~ loc"
            "  ~~ ops.0.loc")

        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'NotEq'}],
             'left': self.ast_1, 'comparators': [self.ast_1]},
            "1 <> 1",
            "~~~~~~ loc"
            "  ~~ ops.0.loc")

        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'In'}],
             'left': self.ast_1, 'comparators': [self.ast_1]},
            "1 in 1",
            "~~~~~~ loc"
            "  ~~ ops.0.loc")

        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'NotIn'}],
             'left': self.ast_1, 'comparators': [self.ast_1]},
            "1 not in 1",
            "~~~~~~~~~~ loc"
            "  ~~~~~~ ops.0.loc")

        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'Is'}],
             'left': self.ast_1, 'comparators': [self.ast_1]},
            "1 is 1",
            "~~~~~~ loc"
            "  ~~ ops.0.loc")

        # self.assertParsesExpr(
        #     {'ty': 'Compare', 'ops': [{'ty': 'NotIs'}],
        #      'left': self.ast_1, 'comparators': [self.ast_1]},
        #     "1 is not 1",
        #     "~~~~~~~~~~ loc"
        #     "  ~~~~~~ ops.0.loc")

    def test_compare_multi(self):
        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'Lt'}, {'ty': 'LtE'}],
             'left': self.ast_1,
             'comparators': [{'ty': 'Num', 'n': 2}, {'ty': 'Num', 'n': 3}]},
            "1 < 2 <= 3",
            "~~~~~~~~~~ loc"
            "  ^ ops.0.loc"
            "      ~~ ops.1.loc")

    def test_boolop(self):
        self.assertParsesExpr(
            {'ty': 'BoolOp', 'op': {'ty': 'And'}, 'values': [self.ast_1, self.ast_1]},
            "1 and 1",
            "~~~~~~~ loc"
            "  ~~~ op_locs.0")

        self.assertParsesExpr(
            {'ty': 'BoolOp', 'op': {'ty': 'Or'}, 'values': [self.ast_1, self.ast_1]},
            "1 or 1",
            "~~~~~~ loc"
            "  ~~ op_locs.0")

        self.assertParsesExpr(
            {'ty': 'UnaryOp', 'op': {'ty': 'Not'}, 'operand': self.ast_1},
            "not 1",
            "~~~~~ loc"
            "~~~ op.loc")

    def test_boolop_multi(self):
        self.assertParsesExpr(
            {'ty': 'BoolOp', 'op': {'ty': 'Or'}, 'values': [self.ast_1, self.ast_1, self.ast_1]},
            "1 or 1 or 1",
            "~~~~~~~~~~~ loc"
            "  ~~ op_locs.0"
            "       ~~ op_locs.1")

    #
    # STATEMENTS
    #

    ast_x = {'ty': 'Name', 'id': 'x', 'ctx': None}
    ast_y = {'ty': 'Name', 'id': 'y', 'ctx': None}

    def test_assign(self):
        self.assertParsesSuite(
            [{'ty': 'Assign', 'targets': [self.ast_x], 'value': self.ast_1}],
            "x = 1",
            "~~~~~ 0.loc"
            "  ^ 0.op_locs.0")

        self.assertParsesSuite(
            [{'ty': 'Assign', 'targets': [self.ast_x, self.ast_y], 'value': self.ast_1}],
            "x = y = 1",
            "~~~~~~~~~ 0.loc"
            "  ^ 0.op_locs.0"
            "      ^ 0.op_locs.1")

    def test_augassign(self):
        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'Add'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x += 1",
            "~~~~~~ 0.loc"
            "  ~~ 0.op.loc")

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'Sub'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x -= 1",
            "~~~~~~ 0.loc"
            "  ~~ 0.op.loc")

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'Mult'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x *= 1",
            "~~~~~~ 0.loc"
            "  ~~ 0.op.loc")

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'Div'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x /= 1",
            "~~~~~~ 0.loc"
            "  ~~ 0.op.loc")

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'Mod'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x %= 1",
            "~~~~~~ 0.loc"
            "  ~~ 0.op.loc")

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'Pow'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x **= 1",
            "~~~~~~~ 0.loc"
            "  ~~~ 0.op.loc")

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'FloorDiv'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x //= 1",
            "~~~~~~~ 0.loc"
            "  ~~~ 0.op.loc")

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'RShift'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x >>= 1",
            "~~~~~~~ 0.loc"
            "  ~~~ 0.op.loc")

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'LShift'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x <<= 1",
            "~~~~~~~ 0.loc"
            "  ~~~ 0.op.loc")

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'BitAnd'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x &= 1",
            "~~~~~~ 0.loc"
            "  ~~ 0.op.loc")

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'BitOr'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x |= 1",
            "~~~~~~ 0.loc"
            "  ~~ 0.op.loc")

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'BitXor'}, 'target': self.ast_x, 'value': self.ast_1}],
            "x ^= 1",
            "~~~~~~ 0.loc"
            "  ~~ 0.op.loc")

