# coding:utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source, lexer, diagnostic, ast, coverage
from ..coverage import parser
import unittest, sys, re, ast as pyast

if sys.version_info >= (3,):
    def unicode(x): return x

def tearDownModule():
    coverage.report(parser)

class ParserTestCase(unittest.TestCase):

    maxDiff = None

    def parser_for(self, code, version=(2, 6), interactive=False):
        code = code.replace("·", "\n")

        self.source_buffer = source.Buffer(code)
        self.lexer = lexer.Lexer(self.source_buffer, version, interactive=interactive)

        old_next = self.lexer.next
        def lexer_next(**args):
            token = old_next(**args)
            # print(repr(token))
            return token
        self.lexer.next = lexer_next

        self.parser = parser.Parser(self.lexer)
        return self.parser

    def flatten_ast(self, node):
        # Validate locs
        for attr in node.__dict__:
            if attr.endswith('_loc') or attr.endswith('_locs'):
                self.assertTrue(attr in node._locs,
                                "%s not in %s._locs" % (attr, repr(node)))
        for loc in node._locs:
            self.assertTrue(loc in node.__dict__,
                            "%s not in %s._locs" % (loc, repr(node)))

        flat_node = { 'ty': unicode(type(node).__name__) }
        for field in node._fields:
            value = getattr(node, field)
            if isinstance(value, ast.AST):
                value = self.flatten_ast(value)
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], ast.AST):
                value = list(map(self.flatten_ast, value))
            flat_node[unicode(field)] = value
        return flat_node

    def flatten_python_ast(self, node):
        flat_node = { 'ty': unicode(type(node).__name__) }
        for field in node._fields:
            if field == 'ctx':
                flat_node['ctx'] = None
                continue

            value = getattr(node, field)
            if isinstance(value, ast.AST):
                value = self.flatten_python_ast(value)
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], ast.AST):
                value = list(map(self.flatten_python_ast, value))
            flat_node[unicode(field)] = value
        return flat_node

    _loc_re  = re.compile(r"\s*([~^]*)<?\s+([a-z_0-9.]+)")
    _path_re = re.compile(r"(([a-z_]+)|([0-9]+))(\.)?")

    def match_loc(self, ast, matcher, root=lambda x: (0, x)):
        offset, ast = root(ast)

        matcher_pos = 0
        while matcher_pos < len(matcher):
            matcher_match = self._loc_re.match(matcher, matcher_pos)
            if matcher_match is None:
                raise Exception("invalid location matcher %s" % matcher[matcher_pos:])

            range = source.Range(self.source_buffer,
                matcher_match.start(1) - matcher_pos + offset,
                matcher_match.end(1) - matcher_pos + offset)
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

    def _assertParsesGen(self, expected_flat_ast, code,
                         loc_matcher="", ast_slicer=lambda x: (0, x),
                         validate_if=lambda: True):
        ast = self.parser_for(code + "\n").file_input()
        flat_ast = self.flatten_ast(ast)
        python_ast = pyast.parse(code.replace("·", "\n") + "\n")
        flat_python_ast = self.flatten_python_ast(python_ast)
        self.assertEqual({'ty': 'Module', 'body': expected_flat_ast},
                         flat_ast)
        if validate_if():
            self.assertEqual({'ty': 'Module', 'body': expected_flat_ast},
                             flat_python_ast)
        self.match_loc(ast, loc_matcher, ast_slicer)

    def assertParsesSuite(self, expected_flat_ast, code, loc_matcher="", **kwargs):
        self._assertParsesGen(expected_flat_ast, code,
                              loc_matcher, lambda x: (0, x.body),
                              **kwargs)

    def assertParsesExpr(self, expected_flat_ast, code, loc_matcher="", **kwargs):
        self._assertParsesGen([{'ty': 'Expr', 'value': expected_flat_ast}], code,
                              loc_matcher, lambda x: (0, x.body[0].value),
                              **kwargs)

    def assertParsesArgs(self, expected_flat_ast, code, loc_matcher="", **kwargs):
        self._assertParsesGen([{'ty': 'Expr', 'value': {'ty': 'Lambda', 'body': self.ast_1,
                                    'args': expected_flat_ast}}],
                              "lambda %s: 1" % code,
                              loc_matcher, lambda x: (7, x.body[0].value.args),
                              **kwargs)

    def assertParsesToplevel(self, expected_flat_ast, code,
                             mode="file_input", interactive=False):
        ast = getattr(self.parser_for(code, interactive=interactive), mode)()
        self.assertEqual(expected_flat_ast, self.flatten_ast(ast))

    def assertDiagnoses(self, code, level, reason, args={}, loc_matcher=""):
        try:
            self.parser_for(code).file_input()
            self.fail("Expected a diagnostic")
        except diagnostic.DiagnosticException as e:
            self.assertEqual(level, e.diagnostic.level)
            self.assertEqual(reason, e.diagnostic.reason)
            for key in args:
                self.assertEqual(args[key], e.diagnostic.arguments[key],
                                 "{{%s}}: \"%s\" != \"%s\"" %
                                    (key, args[key], e.diagnostic.arguments[key]))
            self.match_loc([e.diagnostic.location] + e.diagnostic.highlights,
                           loc_matcher)

    def assertDiagnosesUnexpected(self, code, err_token, loc_matcher=""):
        self.assertDiagnoses(code,
            "fatal", "unexpected {actual}: expected {expected}",
            {'actual': err_token}, loc_matcher="")

    # Fixtures

    ast_1 = {'ty': 'Num', 'n': 1}
    ast_2 = {'ty': 'Num', 'n': 2}
    ast_3 = {'ty': 'Num', 'n': 3}

    ast_expr_1 = {'ty': 'Expr', 'value': {'ty': 'Num', 'n': 1}}
    ast_expr_2 = {'ty': 'Expr', 'value': {'ty': 'Num', 'n': 2}}
    ast_expr_3 = {'ty': 'Expr', 'value': {'ty': 'Num', 'n': 3}}

    ast_x = {'ty': 'Name', 'id': 'x', 'ctx': None}
    ast_y = {'ty': 'Name', 'id': 'y', 'ctx': None}
    ast_z = {'ty': 'Name', 'id': 'z', 'ctx': None}
    ast_t = {'ty': 'Name', 'id': 't', 'ctx': None}

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

        self.assertParsesExpr(
            {'ty': 'Str', 's': 'foobar'},
            "'foo' 'bar'",
            "~~~~~~~~~~~ loc"
            "^ begin_loc"
            "          ^ end_loc")

    def test_ident(self):
        self.assertParsesExpr(
            {'ty': 'Name', 'id': 'foo', 'ctx': None},
            "foo",
            "~~~ loc")

    #
    # OPERATORS
    #

    def test_unary(self):
        self.assertParsesExpr(
            {'ty': 'UnaryOp', 'op': {'ty': 'UAdd'}, 'operand': self.ast_1},
            "+1",
            "~~ loc"
            "~ op.loc")

        self.assertParsesExpr(
            {'ty': 'UnaryOp', 'op': {'ty': 'USub'}, 'operand': self.ast_x},
            "-x",
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

        self.assertParsesExpr(
            {'ty': 'Compare', 'ops': [{'ty': 'IsNot'}],
             'left': self.ast_1, 'comparators': [self.ast_1]},
            "1 is not 1",
            "~~~~~~~~~~ loc"
            "  ~~~~~~ ops.0.loc")

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
    # COMPOUND LITERALS
    #

    def test_tuple(self):
        self.assertParsesExpr(
            {'ty': 'Tuple', 'elts': [], 'ctx': None},
            "()",
            "^ begin_loc"
            " ^ end_loc"
            "~~ loc")

        self.assertParsesExpr(
            {'ty': 'Tuple', 'elts': [self.ast_1], 'ctx': None},
            "(1,)",
            "~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Tuple', 'elts': [self.ast_1, self.ast_1], 'ctx': None},
            "(1,1)",
            "~~~~~ loc")

        self.assertParsesExpr(
            self.ast_1,
            "(1)",
            " ~ loc")

    def test_list(self):
        self.assertParsesExpr(
            {'ty': 'List', 'elts': [], 'ctx': None},
            "[]",
            "^ begin_loc"
            " ^ end_loc"
            "~~ loc")

        self.assertParsesExpr(
            {'ty': 'List', 'elts': [self.ast_1], 'ctx': None},
            "[1]",
            "~~~ loc")

        self.assertParsesExpr(
            {'ty': 'List', 'elts': [self.ast_1, self.ast_1], 'ctx': None},
            "[1,1]",
            "~~~~~ loc")

    def test_dict(self):
        self.assertParsesExpr(
            {'ty': 'Dict', 'keys': [], 'values': []},
            "{}",
            "^ begin_loc"
            " ^ end_loc"
            "~~ loc")

        self.assertParsesExpr(
            {'ty': 'Dict', 'keys': [self.ast_x], 'values': [self.ast_1]},
            "{x: 1}",
            "^ begin_loc"
            "     ^ end_loc"
            "  ^ colon_locs.0"
            "~~~~~~ loc")

    def test_repr(self):
        self.assertParsesExpr(
            {'ty': 'Repr', 'value': self.ast_1},
            "`1`",
            "^ begin_loc"
            "  ^ end_loc"
            "~~~ loc")

    #
    # GENERATOR AND CONDITIONAL EXPRESSIONS
    #

    def test_list_comp(self):
        self.assertParsesExpr(
            {'ty': 'ListComp', 'elt': self.ast_x, 'generators': [
                {'ty': 'comprehension', 'iter': self.ast_z, 'target': self.ast_y, 'ifs': []}
            ]},
            "[x for y in z]",
            "^ begin_loc"
            "   ~~~ generators.0.for_loc"
            "         ~~ generators.0.in_loc"
            "   ~~~~~~~~~~ generators.0.loc"
            "             ^ end_loc"
            "~~~~~~~~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'ListComp', 'elt': self.ast_x, 'generators': [
                {'ty': 'comprehension', 'iter': self.ast_z, 'target': self.ast_y,
                 'ifs': [self.ast_t]}
            ]},
            "[x for y in z if t]",
            "              ~~ generators.0.if_locs.0"
            "   ~~~~~~~~~~~~~~~ generators.0.loc"
            "~~~~~~~~~~~~~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'ListComp', 'elt': self.ast_x, 'generators': [
                {'ty': 'comprehension', 'iter': self.ast_z, 'target': self.ast_y,
                 'ifs': [self.ast_x]},
                {'ty': 'comprehension', 'iter': self.ast_z, 'target': self.ast_t, 'ifs': []}
            ]},
            "[x for y in z if x for t in z]",
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ loc")

    def test_gen_comp(self):
        self.assertParsesExpr(
            {'ty': 'GeneratorExp', 'elt': self.ast_x, 'generators': [
                {'ty': 'comprehension', 'iter': self.ast_z, 'target': self.ast_y, 'ifs': []}
            ]},
            "(x for y in z)",
            "^ begin_loc"
            "   ~~~ generators.0.for_loc"
            "         ~~ generators.0.in_loc"
            "   ~~~~~~~~~~ generators.0.loc"
            "             ^ end_loc"
            "~~~~~~~~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'GeneratorExp', 'elt': self.ast_x, 'generators': [
                {'ty': 'comprehension', 'iter': self.ast_z, 'target': self.ast_y,
                 'ifs': [self.ast_t]}
            ]},
            "(x for y in z if t)",
            "              ~~ generators.0.if_locs.0"
            "   ~~~~~~~~~~~~~~~ generators.0.loc"
            "~~~~~~~~~~~~~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'GeneratorExp', 'elt': self.ast_x, 'generators': [
                {'ty': 'comprehension', 'iter': self.ast_z, 'target': self.ast_y,
                 'ifs': [self.ast_x]},
                {'ty': 'comprehension', 'iter': self.ast_z, 'target': self.ast_t, 'ifs': []}
            ]},
            "(x for y in z if x for t in z)",
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ loc")

    def test_yield_expr(self):
        self.assertParsesExpr(
            {'ty': 'Yield', 'value': self.ast_1},
            "(yield 1)",
            " ~~~~~ yield_loc"
            " ~~~~~~~ loc")

    def test_if_expr(self):
        self.assertParsesExpr(
            {'ty': 'IfExp', 'body': self.ast_x, 'test': self.ast_y, 'orelse': self.ast_z},
            "x if y else z",
            "  ~~ if_loc"
            "       ~~~~ else_loc"
            "~~~~~~~~~~~~~ loc")

    def test_lambda(self):
        self.assertParsesExpr(
            {'ty': 'Lambda',
             'args': {'ty': 'arguments', 'args': [], 'defaults': [],
                      'kwarg': None, 'vararg': None},
             'body': self.ast_x},
            "lambda: x",
            "~~~~~~ lambda_loc"
            "      < args.loc"
            "      ^ colon_loc"
            "~~~~~~~~~ loc")

    def test_old_lambda(self):
        self.assertParsesExpr(
            {'ty': 'ListComp', 'elt': self.ast_x, 'generators': [
                {'ty': 'comprehension', 'iter': self.ast_z, 'target': self.ast_y,
                 'ifs': [{'ty': 'Lambda',
                     'args': {'ty': 'arguments', 'args': [], 'defaults': [],
                              'kwarg': None, 'vararg': None},
                     'body': self.ast_t}
                ]}
            ]},
            "[x for y in z if lambda: t]",
            "                 ~~~~~~ generators.0.ifs.0.lambda_loc"
            "                       < generators.0.ifs.0.args.loc"
            "                       ^ generators.0.ifs.0.colon_loc"
            "                 ~~~~~~~~~ generators.0.ifs.0.loc")

    #
    # CALLS, ATTRIBUTES AND SUBSCRIPTS
    #

    def test_call(self):
        self.assertParsesExpr(
            {'ty': 'Call', 'func': self.ast_x, 'starargs': None, 'kwargs': None,
             'args': [], 'keywords': []},
            "x()",
            " ^ begin_loc"
            "  ^ end_loc"
            "~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Call', 'func': self.ast_x, 'starargs': None, 'kwargs': None,
             'args': [self.ast_y, self.ast_z], 'keywords': []},
            "x(y, z)",
            "~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Call', 'func': self.ast_x, 'starargs': None, 'kwargs': None,
             'args': [self.ast_y], 'keywords': [
                { 'ty': 'keyword', 'arg': 'z', 'value': self.ast_z}
            ]},
            "x(y, z=z)",
            "     ^ keywords.0.arg_loc"
            "      ^ keywords.0.equals_loc"
            "     ~~~ keywords.0.loc"
            "~~~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Call', 'func': self.ast_x, 'starargs': None, 'kwargs': None,
             'args': [self.ast_y], 'keywords': []},
            "x(y,)",
            "~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Call', 'func': self.ast_x, 'starargs': self.ast_y, 'kwargs': None,
             'args': [], 'keywords': []},
            "x(*y)",
            "  ^ star_loc"
            "~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Call', 'func': self.ast_x, 'starargs': self.ast_y, 'kwargs': self.ast_z,
             'args': [], 'keywords': []},
            "x(*y, **z)",
            "  ^ star_loc"
            "      ^^ dstar_loc"
            "~~~~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Call', 'func': self.ast_x, 'starargs': self.ast_y, 'kwargs': self.ast_z,
             'args': [], 'keywords': [{'ty': 'keyword', 'arg': 't', 'value': self.ast_t}]},
            "x(*y, t=t, **z)",
            "  ^ star_loc"
            "           ^^ dstar_loc"
            "~~~~~~~~~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Call', 'func': self.ast_x, 'starargs': self.ast_z, 'kwargs': self.ast_t,
             'args': [self.ast_y], 'keywords': []},
            "x(y, *z, **t)",
            "     ^ star_loc"
            "         ^^ dstar_loc"
            "~~~~~~~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Call', 'func': self.ast_x, 'starargs': None, 'kwargs': self.ast_z,
             'args': [], 'keywords': []},
            "x(**z)",
            "  ^^ dstar_loc"
            "~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Call', 'func': self.ast_x, 'starargs': None, 'kwargs': None,
             'keywords': [], 'args': [
                {'ty': 'GeneratorExp', 'elt': self.ast_y, 'generators': [
                    {'ty': 'comprehension', 'iter': self.ast_z, 'target': self.ast_y, 'ifs': []}
                ]}
            ]},
            "x(y for y in z)")

    def test_subscript(self):
        self.assertParsesExpr(
            {'ty': 'Subscript', 'value': self.ast_x, 'ctx': None,
             'slice': {'ty': 'Index', 'value': self.ast_1}},
            "x[1]",
            " ^ begin_loc"
            "  ^ slice.loc"
            "   ^ end_loc"
            "~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Subscript', 'value': self.ast_x, 'ctx': None,
             'slice': {'ty': 'Index', 'value': {'ty': 'Tuple', 'ctx': None, 'elts': [
                self.ast_1, self.ast_2
            ]}}},
            "x[1, 2]",
            "  ~~~~ slice.loc"
            "~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Subscript', 'value': self.ast_x, 'ctx': None,
             'slice': {'ty': 'Slice', 'lower': self.ast_1, 'upper': None, 'step': None}},
            "x[1:]",
            "   ^ slice.bound_colon_loc"
            "  ~~ slice.loc"
            "~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Subscript', 'value': self.ast_x, 'ctx': None,
             'slice': {'ty': 'Slice', 'lower': None, 'upper': self.ast_1, 'step': None}},
            "x[:1]",
            "  ^ slice.bound_colon_loc"
            "  ~~ slice.loc"
            "~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Subscript', 'value': self.ast_x, 'ctx': None,
             'slice': {'ty': 'Slice', 'lower': self.ast_1, 'upper': self.ast_2, 'step': None}},
            "x[1:2]",
            "   ^ slice.bound_colon_loc"
            "  ~~~ slice.loc"
            "~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Subscript', 'value': self.ast_x, 'ctx': None,
             'slice': {'ty': 'ExtSlice', 'dims': [
                {'ty': 'Slice', 'lower': self.ast_1, 'upper': self.ast_2, 'step': None},
                {'ty': 'Index', 'value': self.ast_2},
             ]}},
            "x[1:2, 2]",
            "  ~~~~~~ slice.loc"
            "~~~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Subscript', 'value': self.ast_x, 'ctx': None,
             'slice': {'ty': 'Slice', 'lower': self.ast_1, 'upper': self.ast_2, 'step': None}},
            "x[1:2:]",
            "   ^ slice.bound_colon_loc"
            "     ^ slice.step_colon_loc"
            "  ~~~~ slice.loc"
            "~~~~~~~ loc",
            # A Python bug places ast.Name(id='None') instead of None in step on <3.0
            validate_if=lambda: sys.version_info[0] > 2)

        self.assertParsesExpr(
            {'ty': 'Subscript', 'value': self.ast_x, 'ctx': None,
             'slice': {'ty': 'Slice', 'lower': self.ast_1, 'upper': self.ast_2, 'step': self.ast_3}},
            "x[1:2:3]",
            "   ^ slice.bound_colon_loc"
            "     ^ slice.step_colon_loc"
            "  ~~~~~ slice.loc"
            "~~~~~~~~ loc")

        self.assertParsesExpr(
            {'ty': 'Subscript', 'value': self.ast_x, 'ctx': None,
             'slice': {'ty': 'Ellipsis'}},
            "x[...]",
            "  ~~~ slice.loc"
            "~~~~~~ loc")

    def test_attribute(self):
        self.assertParsesExpr(
            {'ty': 'Attribute', 'value': self.ast_x, 'attr': 'zz', 'ctx': None},
            "x.zz",
            " ^ dot_loc"
            "  ~~ attr_loc"
            "~~~~ loc")

    #
    # SIMPLE STATEMENTS
    #

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

        self.assertParsesSuite(
            [{'ty': 'Assign', 'targets': [self.ast_x], 'value':
              {'ty': 'Yield', 'value': self.ast_y}}],
            "x = yield y",
            "~~~~~~~~~~~ 0.loc")

    def test_assign_tuplerhs(self):
        self.assertParsesSuite(
            [{'ty': 'Assign', 'targets': [self.ast_x], 'value':
                {'ty': 'Tuple', 'ctx': None, 'elts': [self.ast_1, self.ast_2]}}],
            "x = 1, 2",
            "    ~~~~ 0.value.loc"
            "~~~~~~~~ 0.loc")

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

        self.assertParsesSuite(
            [{'ty': 'AugAssign', 'op': {'ty': 'Add'}, 'target': self.ast_x, 'value':
              {'ty': 'Yield', 'value': self.ast_y}}],
            "x += yield y",
            "~~~~~~~~~~~~ 0.loc")

    def test_print(self):
        self.assertParsesSuite(
            [{'ty': 'Print', 'dest': None, 'values': [self.ast_1], 'nl': True}],
            "print 1",
            "~~~~~ 0.keyword_loc"
            "~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Print', 'dest': None, 'values': [self.ast_1], 'nl': False}],
            "print 1,",
            "~~~~~ 0.keyword_loc"
            "~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Print', 'dest': self.ast_2, 'values': [self.ast_1], 'nl': True}],
            "print >>2, 1",
            "~~~~~ 0.keyword_loc"
            "      ~~ 0.dest_loc"
            "~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Print', 'dest': self.ast_2, 'values': [self.ast_1], 'nl': False}],
            "print >>2, 1,",
            "~~~~~ 0.keyword_loc"
            "      ~~ 0.dest_loc"
            "~~~~~~~~~~~~~ 0.loc")

    def test_del(self):
        self.assertParsesSuite(
            [{'ty': 'Delete', 'targets': [self.ast_x]}],
            "del x",
            "~~~ 0.keyword_loc"
            "~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Delete', 'targets': [self.ast_x, self.ast_y]}],
            "del x, y",
            "~~~ 0.keyword_loc"
            "~~~~~~~~ 0.loc")

    def test_pass(self):
        self.assertParsesSuite(
            [{'ty': 'Pass'}],
            "pass",
            "~~~~ 0.keyword_loc"
            "~~~~ 0.loc")

    def test_break(self):
        self.assertParsesSuite(
            [{'ty': 'Break'}],
            "break",
            "~~~~~ 0.keyword_loc"
            "~~~~~ 0.loc")

    def test_continue(self):
        self.assertParsesSuite(
            [{'ty': 'Continue'}],
            "continue",
            "~~~~~~~~ 0.keyword_loc"
            "~~~~~~~~ 0.loc")

    def test_return(self):
        self.assertParsesSuite(
            [{'ty': 'Return', 'value': None}],
            "return",
            "~~~~~~ 0.keyword_loc"
            "~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Return', 'value': self.ast_x}],
            "return x",
            "~~~~~~ 0.keyword_loc"
            "~~~~~~~~ 0.loc")

    def test_yield(self):
        self.assertParsesSuite(
            [{'ty': 'Expr', 'value': {'ty': 'Yield', 'value': self.ast_x}}],
            "yield x",
            "~~~~~ 0.value.yield_loc"
            "~~~~~~~ 0.value.loc"
            "~~~~~~~ 0.loc")

    def test_raise(self):
        self.assertParsesSuite(
            [{'ty': 'Raise', 'type': None, 'inst': None, 'tback': None}],
            "raise",
            "~~~~~ 0.keyword_loc"
            "~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Raise', 'type': self.ast_x, 'inst': None, 'tback': None}],
            "raise x",
            "~~~~~ 0.keyword_loc"
            "~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Raise', 'type': self.ast_x, 'inst': self.ast_y, 'tback': None}],
            "raise x, y",
            "~~~~~ 0.keyword_loc"
            "~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Raise', 'type': self.ast_x, 'inst': self.ast_y, 'tback': self.ast_z}],
            "raise x, y, z",
            "~~~~~ 0.keyword_loc"
            "~~~~~~~~~~~~~ 0.loc")

    def test_import(self):
        self.assertParsesSuite(
            [{'ty': 'Import', 'names': [
                {'ty': 'alias', 'name': 'foo', 'asname': None}
            ]}],
            "import foo",
            "~~~~~~ 0.keyword_loc"
            "       ~~~ 0.names.0.name_loc"
            "       ~~~ 0.names.0.loc"
            "~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Import', 'names': [
                {'ty': 'alias', 'name': 'foo.bar', 'asname': None}
            ]}],
            "import foo. bar",
            "~~~~~~ 0.keyword_loc"
            "       ~~~~~~~~ 0.names.0.name_loc"
            "       ~~~~~~~~ 0.names.0.loc"
            "~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Import', 'names': [
                {'ty': 'alias', 'name': 'foo', 'asname': 'bar'}
            ]}],
            "import foo as bar",
            "~~~~~~ 0.keyword_loc"
            "       ~~~ 0.names.0.name_loc"
            "           ~~ 0.names.0.as_loc"
            "              ~~~ 0.names.0.asname_loc"
            "       ~~~~~~~~~~ 0.names.0.loc"
            "~~~~~~~~~~~~~~~~~ 0.loc")

    def test_from(self):
        self.assertParsesSuite(
            [{'ty': 'ImportFrom', 'names': [
                {'ty': 'alias', 'name': 'foo', 'asname': None}
            ], 'module': 'bar', 'level': 0}],
            "from bar import foo",
            "~~~~ 0.keyword_loc"
            "     ~~~ 0.module_loc"
            "         ~~~~~~ 0.import_loc"
            "                ~~~ 0.names.0.name_loc"
            "                ~~~ 0.names.0.loc"
            "~~~~~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'ImportFrom', 'names': [
                {'ty': 'alias', 'name': 'foo', 'asname': None}
            ], 'module': 'bar', 'level': 2}],
            "from ..bar import foo",
            "~~~~ 0.keyword_loc"
            "     ~~ 0.dots_loc"
            "       ~~~ 0.module_loc"
            "           ~~~~~~ 0.import_loc"
            "                  ~~~ 0.names.0.name_loc"
            "                  ~~~ 0.names.0.loc"
            "~~~~~~~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'ImportFrom', 'names': [
                {'ty': 'alias', 'name': 'foo', 'asname': 'bar'}
            ], 'module': 'bar', 'level': 2}],
            "from ..bar import foo as bar",
            "                  ~~~ 0.names.0.name_loc"
            "                      ~~ 0.names.0.as_loc"
            "                         ~~~ 0.names.0.asname_loc"
            "                  ~~~~~~~~~~ 0.names.0.loc"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'ImportFrom', 'names': [
                {'ty': 'alias', 'name': 'foo', 'asname': None}
            ], 'module': 'bar', 'level': 2}],
            "from ..bar import (foo)",
            "                  ^ 0.lparen_loc"
            "                   ~~~ 0.names.0.loc"
            "                      ^ 0.rparen_loc"
            "~~~~~~~~~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'ImportFrom', 'names': [
                {'ty': 'alias', 'name': '*', 'asname': None}
            ], 'module': 'bar', 'level': 0}],
            "from bar import *",
            "                ^ 0.names.0.name_loc"
            "                ^ 0.names.0.loc"
            "~~~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'ImportFrom', 'names': [
                {'ty': 'alias', 'name': 'foo', 'asname': None}
            ], 'module': None, 'level': 2}],
            "from .. import foo",
            "     ~~ 0.dots_loc"
            "~~~~~~~~~~~~~~~~~~ 0.loc")

    def test_global(self):
        self.assertParsesSuite(
            [{'ty': 'Global', 'names': ['x', 'y']}],
            "global x, y",
            "~~~~~~ 0.keyword_loc"
            "       ^ 0.name_locs.0"
            "          ^ 0.name_locs.1"
            "~~~~~~~~~~~ 0.loc")

    def test_exec(self):
        self.assertParsesSuite(
            [{'ty': 'Exec', 'body': self.ast_1, 'globals': None, 'locals': None}],
            "exec 1",
            "~~~~ 0.keyword_loc"
            "~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Exec', 'body': self.ast_1, 'globals': self.ast_2, 'locals': None}],
            "exec 1 in 2",
            "~~~~ 0.keyword_loc"
            "       ~~ 0.in_loc"
            "~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Exec', 'body': self.ast_1, 'globals': self.ast_2, 'locals': self.ast_3}],
            "exec 1 in 2, 3",
            "~~~~ 0.keyword_loc"
            "       ~~ 0.in_loc"
            "~~~~~~~~~~~~~~ 0.loc")

    def test_assert(self):
        self.assertParsesSuite(
            [{'ty': 'Assert', 'test': self.ast_1, 'msg': None}],
            "assert 1",
            "~~~~~~ 0.keyword_loc"
            "~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'Assert', 'test': self.ast_1, 'msg': self.ast_2}],
            "assert 1, 2",
            "~~~~~~ 0.keyword_loc"
            "~~~~~~~~~~~ 0.loc")

    #
    # COMPOUND STATEMENTS
    #

    def test_if(self):
        self.assertParsesSuite(
            [{'ty': 'If', 'test': self.ast_x, 'body': [self.ast_expr_1], 'orelse': []}],
            "if x:·  1",
            "^^ 0.keyword_loc"
            "    ^ 0.if_colon_loc"
            "~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'If', 'test': self.ast_x,
              'body': [self.ast_expr_1], 'orelse': [self.ast_expr_2] }],
            "if x:·  1·else:·  2",
            "^^ 0.keyword_loc"
            "    ^ 0.if_colon_loc"
            "          ^^^^ 0.else_loc"
            "              ^ 0.else_colon_loc"
            "~~~~~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'If', 'test': self.ast_x, 'body': [self.ast_expr_1], 'orelse': [
                {'ty': 'If', 'test': self.ast_y, 'body': [self.ast_expr_2],
                 'orelse': [self.ast_expr_3]}
            ]}],
            "if x:·  1·elif y:·  2·else:·  3",
            "^^ 0.keyword_loc"
            "    ^ 0.if_colon_loc"
            "          ~~~~ 0.orelse.0.keyword_loc"
            "                ^ 0.orelse.0.if_colon_loc"
            "                      ~~~~ 0.orelse.0.else_loc"
            "                          ^ 0.orelse.0.else_colon_loc"
            "          ~~~~~~~~~~~~~~~~~~~~~ 0.orelse.0.loc"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 0.loc")

    def test_while(self):
        self.assertParsesSuite(
            [{'ty': 'While', 'test': self.ast_x, 'body': [self.ast_expr_1], 'orelse': []}],
            "while x:·  1",
            "~~~~~ 0.keyword_loc"
            "       ^ 0.while_colon_loc"
            "~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'While', 'test': self.ast_x, 'body': [self.ast_expr_1],
              'orelse': [self.ast_expr_2]}],
            "while x:·  1·else:·  2",
            "~~~~~ 0.keyword_loc"
            "       ^ 0.while_colon_loc"
            "             ~~~~ 0.else_loc"
            "                 ^ 0.else_colon_loc"
            "~~~~~~~~~~~~~~~~~~~~~~ 0.loc")

    def test_for(self):
        self.assertParsesSuite(
            [{'ty': 'For', 'target': self.ast_x, 'iter': self.ast_y,
              'body': [self.ast_expr_1], 'orelse': []}],
            "for x in y:·  1",
            "~~~ 0.keyword_loc"
            "      ~~ 0.in_loc"
            "          ^ 0.for_colon_loc"
            "~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'For', 'target': self.ast_x, 'iter': self.ast_y,
              'body': [self.ast_expr_1], 'orelse': [self.ast_expr_2]}],
            "for x in y:·  1·else:·  2",
            "                ~~~~ 0.else_loc"
            "                    ^ 0.else_colon_loc"
            "~~~~~~~~~~~~~~~~~~~~~~~~~ 0.loc")

    def test_try(self):
        self.assertParsesSuite(
            [{'ty': 'TryExcept', 'body': [self.ast_expr_1], 'orelse': [],
              'handlers': [
                {'ty': 'ExceptHandler', 'type': None, 'name': None,
                 'body': [self.ast_expr_2]}
            ]}],
            "try:·  1·except:·  2",
            "~~~ 0.keyword_loc"
            "   ^ 0.try_colon_loc"
            "         ~~~~~~ 0.handlers.0.except_loc"
            "               ^ 0.handlers.0.colon_loc"
            "         ~~~~~~~~~~~ 0.handlers.0.loc"
            "~~~~~~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'TryExcept', 'body': [self.ast_expr_1], 'orelse': [],
              'handlers': [
                {'ty': 'ExceptHandler', 'type': self.ast_y, 'name': None,
                 'body': [self.ast_expr_2]}
            ]}],
            "try:·  1·except y:·  2")

        self.assertParsesSuite(
            [{'ty': 'TryExcept', 'body': [self.ast_expr_1], 'orelse': [],
              'handlers': [
                {'ty': 'ExceptHandler', 'type': self.ast_y, 'name': self.ast_t,
                 'body': [self.ast_expr_2]}
            ]}],
            "try:·  1·except y as t:·  2",
            "                  ~~ 0.handlers.0.as_loc")

        self.assertParsesSuite(
            [{'ty': 'TryExcept', 'body': [self.ast_expr_1], 'orelse': [],
              'handlers': [
                {'ty': 'ExceptHandler', 'type': self.ast_y, 'name': self.ast_t,
                 'body': [self.ast_expr_2]}
            ]}],
            "try:·  1·except y, t:·  2",
            "                 ^ 0.handlers.0.as_loc")

        self.assertParsesSuite(
            [{'ty': 'TryExcept', 'body': [self.ast_expr_1], 'orelse': [self.ast_expr_3],
              'handlers': [
                {'ty': 'ExceptHandler', 'type': None, 'name': None,
                 'body': [self.ast_expr_2]}
            ]}],
            "try:·  1·except:·  2·else:·  3",
            "                     ~~~~ 0.else_loc"
            "                         ^ 0.else_colon_loc"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 0.loc")

    def test_finally(self):
        self.assertParsesSuite(
            [{'ty': 'TryFinally', 'body': [self.ast_expr_1], 'finalbody': [self.ast_expr_2]}],
            "try:·  1·finally:·  2",
            "~~~ 0.keyword_loc"
            "   ^ 0.try_colon_loc"
            "         ~~~~~~~ 0.finally_loc"
            "                ^ 0.finally_colon_loc"
            "~~~~~~~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'TryFinally', 'finalbody': [self.ast_expr_3], 'body': [
                {'ty': 'TryExcept', 'body': [self.ast_expr_1], 'orelse': [], 'handlers': [
                    {'ty': 'ExceptHandler', 'type': None, 'name': None,
                     'body': [self.ast_expr_2]}
                ]}
            ]}],
            "try:·  1·except:·  2·finally:·  3",
            "~~~ 0.keyword_loc"
            "   ^ 0.try_colon_loc"
            "~~~ 0.body.0.keyword_loc"
            "   ^ 0.body.0.try_colon_loc"
            "~~~~~~~~~~~~~~~~~~~~ 0.body.0.loc"
            "                     ~~~~~~~ 0.finally_loc"
            "                            ^ 0.finally_colon_loc"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 0.loc")

    def test_with(self):
        self.assertParsesSuite(
            [{'ty': 'With', 'context_expr': self.ast_x, 'optional_vars': None,
              'body': [self.ast_expr_1]}],
            "with x:·  1",
            "~~~~ 0.keyword_loc"
            "      ^ 0.colon_loc"
            "~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'With', 'context_expr': self.ast_x, 'optional_vars': self.ast_y,
              'body': [self.ast_expr_1]}],
            "with x as y:·  1",
            "       ~~ 0.as_loc"
            "~~~~~~~~~~~~~~~~ 0.loc")

    def test_class(self):
        self.assertParsesSuite(
            [{'ty': 'ClassDef', 'name': 'x', 'bases': [],
              'body': [{'ty': 'Pass'}], 'decorator_list': []}],
            "class x:·  pass",
            "~~~~~ 0.keyword_loc"
            "      ^ 0.name_loc"
            "       ^ 0.colon_loc"
            "~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'ClassDef', 'name': 'x', 'bases': [self.ast_y, self.ast_z],
              'body': [{'ty': 'Pass'}], 'decorator_list': []}],
            "class x(y, z):·  pass",
            "       ^ 0.lparen_loc"
            "            ^ 0.rparen_loc"
            "~~~~~~~~~~~~~~~~~~~~~ 0.loc")

    def test_func(self):
        self.assertParsesSuite(
            [{'ty': 'FunctionDef', 'name': 'foo',
              'args': {'ty': 'arguments', 'args': [], 'defaults': [],
                       'kwarg': None, 'vararg': None},
              'body': [{'ty': 'Pass'}], 'decorator_list': []}],
            "def foo():·  pass",
            "~~~ 0.keyword_loc"
            "    ~~~ 0.name_loc"
            "       ^ 0.args.begin_loc"
            "        ^ 0.args.end_loc"
            "         ^ 0.colon_loc"
            "~~~~~~~~~~~~~~~~~ 0.loc")

    def test_decorated(self):
        self.assertParsesSuite(
            [{'ty': 'ClassDef', 'name': 'x', 'bases': [],
              'body': [{'ty': 'Pass'}], 'decorator_list': [self.ast_x]}],
            "@x·class x:·  pass",
            "^ 0.at_locs.0"
            " ^ 0.decorator_list.0.loc"
            "~~~~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'ClassDef', 'name': 'x', 'bases': [],
              'body': [{'ty': 'Pass'}], 'decorator_list': [
                {'ty': 'Call', 'func': self.ast_x,
                 'args': [], 'keywords': [], 'kwargs': None, 'starargs': None}
            ]}],
            "@x()·class x:·  pass",
            "^ 0.at_locs.0"
            " ~~~ 0.decorator_list.0.loc"
            "~~~~~~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'ClassDef', 'name': 'x', 'bases': [],
              'body': [{'ty': 'Pass'}], 'decorator_list': [
                {'ty': 'Call', 'func': self.ast_x,
                 'args': [self.ast_1], 'keywords': [], 'kwargs': None, 'starargs': None}
            ]}],
            "@x(1)·class x:·  pass",
            "^ 0.at_locs.0"
            " ~~~~ 0.decorator_list.0.loc"
            "~~~~~~~~~~~~~~~~~~~~~ 0.loc")

        self.assertParsesSuite(
            [{'ty': 'FunctionDef', 'name': 'x',
              'args': {'ty': 'arguments', 'args': [], 'defaults': [],
                       'kwarg': None, 'vararg': None},
              'body': [{'ty': 'Pass'}], 'decorator_list': [self.ast_x]}],
            "@x·def x():·  pass",
            "^ 0.at_locs.0"
            " ^ 0.decorator_list.0.loc"
            "~~~~~~~~~~~~~~~~~~ 0.loc")

    #
    # FUNCTION AND LAMBDA ARGUMENTS
    #

    def test_args(self):
        self.assertParsesArgs(
            {'ty': 'arguments', 'args': [], 'defaults': [],
             'vararg': None, 'kwarg': None},
            "")

        self.assertParsesArgs(
            {'ty': 'arguments', 'args': [self.ast_x], 'defaults': [],
             'vararg': None, 'kwarg': None},
            "x",
            "~ args.0.loc"
            "~ loc")

        self.assertParsesArgs(
            {'ty': 'arguments', 'args': [self.ast_x], 'defaults': [self.ast_1],
             'vararg': None, 'kwarg': None},
            "x=1",
            "~ args.0.loc"
            " ~ equals_locs.0"
            "  ~ defaults.0.loc"
            "~~~ loc")

        self.assertParsesArgs(
            {'ty': 'arguments', 'args': [self.ast_x, self.ast_y], 'defaults': [],
             'vararg': None, 'kwarg': None},
            "x, y",
            "~~~~ loc")

        self.assertParsesArgs(
            {'ty': 'arguments', 'args': [], 'defaults': [],
             'vararg': 'y', 'kwarg': None},
            "*y",
            "^ star_loc"
            " ~ vararg_loc"
            "~~ loc")

        self.assertParsesArgs(
            {'ty': 'arguments', 'args': [self.ast_x], 'defaults': [],
             'vararg': 'y', 'kwarg': None},
            "x, *y",
            "   ^ star_loc"
            "    ~ vararg_loc"
            "~~~~~ loc")

        self.assertParsesArgs(
            {'ty': 'arguments', 'args': [], 'defaults': [],
             'vararg': None, 'kwarg': 'y'},
            "**y",
            "^^ dstar_loc"
            "  ~ kwarg_loc"
            "~~~ loc")

        self.assertParsesArgs(
            {'ty': 'arguments', 'args': [self.ast_x], 'defaults': [],
             'vararg': None, 'kwarg': 'y'},
            "x, **y",
            "   ^^ dstar_loc"
            "     ~ kwarg_loc"
            "~~~~~~ loc")

        self.assertParsesArgs(
            {'ty': 'arguments', 'args': [self.ast_x], 'defaults': [],
             'vararg': 'y', 'kwarg': 'z'},
            "x, *y, **z",
            "   ^ star_loc"
            "    ~ vararg_loc"
            "       ^^ dstar_loc"
            "         ~ kwarg_loc"
            "~~~~~~~~~~ loc")

        self.assertParsesArgs(
            {'ty': 'arguments', 'defaults': [], 'vararg': None, 'kwarg': None,
             'args': [{'ty': 'Tuple', 'ctx': None, 'elts': [self.ast_x, self.ast_y]}]},
            "(x,y)",
            "^ args.0.begin_loc"
            "    ^ args.0.end_loc"
            "~~~~~ args.0.loc"
            "~~~~~ loc")

    def test_args_def(self):
        self.assertParsesSuite(
            [{'ty': 'FunctionDef', 'name': 'foo',
              'args': {'ty': 'arguments', 'args': [self.ast_x], 'defaults': [],
                       'kwarg': None, 'vararg': None},
              'body': [{'ty': 'Pass'}], 'decorator_list': []}],
            "def foo(x):·  pass")

    def test_args_oldlambda(self):
        self.assertParsesExpr(
            {'ty': 'ListComp', 'elt': self.ast_x, 'generators': [
                {'ty': 'comprehension', 'iter': self.ast_z, 'target': self.ast_y,
                 'ifs': [{'ty': 'Lambda',
                     'args': {'ty': 'arguments', 'args': [self.ast_x], 'defaults': [],
                              'kwarg': None, 'vararg': None},
                     'body': self.ast_t}
                ]}
            ]},
            "[x for y in z if lambda x: t]")

    #
    # PARSING MODES
    #

    def test_single_input(self):
        self.assertParsesToplevel(
            {'ty': 'Interactive', 'body': []},
            "·",
            mode='single_input', interactive=True)

        self.assertParsesToplevel(
            {'ty': 'Interactive', 'body': [
                {'ty': 'Expr', 'value': self.ast_1}
            ]},
            "1·",
            mode='single_input', interactive=True)

        self.assertParsesToplevel(
            {'ty': 'Interactive', 'body': [
                {'ty': 'If', 'test': self.ast_x, 'body': [
                    {'ty': 'Expr', 'value': self.ast_1}
                ], 'orelse': []}
            ]},
            "if x: 1··",
            mode='single_input', interactive=True)

    def test_file_input(self):
        self.assertParsesToplevel(
            {'ty': 'Module', 'body': []},
            "·",
            mode='file_input', interactive=True)

    def test_eval_input(self):
        self.assertParsesToplevel(
            {'ty': 'Expression', 'body': [self.ast_1]},
            "1·",
            mode='eval_input', interactive=True)

    #
    # FUTURE IMPORTS
    #

    def test_future_print(self):
        self.assertParsesSuite(
            [{'ty': 'ImportFrom',
              'names': [{'ty': 'alias', 'name': 'print_function', 'asname': None}],
              'module': '__future__', 'level': 0},
             {'ty': 'Expr', 'value':
                {'ty': 'Call', 'func': {'ty': 'Name', 'id': 'print', 'ctx': None},
                 'starargs': None, 'kwargs': None, 'args': [self.ast_x], 'keywords': []}}],
            "from __future__ import print_function·print(x)")

    #
    # DIAGNOSTICS
    #

    def test_diag_assignable(self):
        self.assertDiagnoses(
            "1 = 1",
            'fatal', "cannot assign to this expression", {},
            "^ 0")

        self.assertDiagnoses(
            "[1] = 1",
            'fatal', "cannot assign to this expression", {},
            " ^ 0")

        self.assertDiagnoses(
            "x() = 1",
            'fatal', "cannot assign to this expression", {},
            "~~~ 0")

        self.assertDiagnoses(
            "del 1",
            'fatal', "cannot delete this expression", {},
            "    ^ 0")

    def test_diag_def(self):
        self.assertDiagnoses(
            "def x(y=1, z): pass",
            'fatal', "non-default argument follows default argument", {},
            "           ^ 0"
            "      ~~~ 1")

    def test_diag_augassign(self):
        self.assertDiagnoses(
            "(1,) += 1",
            'fatal', "illegal expression for augmented assignment", {},
            "     ^^ 0"
            "~~~~ 1")

        self.assertDiagnoses(
            "[1] += 1",
            'fatal', "illegal expression for augmented assignment", {},
            "    ^^ 0"
            "~~~ 1")

    def test_diag_call(self):
        self.assertDiagnoses(
            "x(*y, z)",
            'fatal', "only named arguments may follow *expression", {},
            "      ^ 0"
            "  ~~ 1")

        self.assertDiagnoses(
            "x(y=1, z)",
            'fatal', "non-keyword arg after keyword arg", {},
            "       ^ 0"
            "  ~~~ 1")

        self.assertDiagnoses(
            "x(1=1)",
            'fatal', "keyword must be an identifier", {},
            "  ^ 0")

    def test_diag_generic(self):
        self.assertDiagnosesUnexpected(
            "x + ,", ",",
            "    ^ 0")
