# encoding:utf-8

"""
The :mod:`parser` module concerns itself with parsing Python source.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from functools import reduce
from . import source, diagnostic, lexer, ast

# A few notes about our approach to parsing:
#
# Python uses an LL(1) parser generator. It's a bit weird, because
# the usual reason to choose LL(1) is to make a handwritten parser
# possible, however Python's grammar is formulated in a way that
# is much more easily recognized if you make an FSM rather than
# the usual "if accept(token)..." ladder. So in a way it is
# the worst of both worlds.
#
# We don't use a parser generator because we want to have an unified
# grammar for all Python versions, and also have grammar coverage
# analysis and nice error recovery. To make the grammar compact,
# we use combinators to compose it from predefined fragments,
# such as "sequence" or "alternation" or "Kleene star". This easily
# gives us one token of lookahead in most cases, but e.g. not
# in the following one:
#
#     argument: test | test '=' test
#
# There are two issues with this. First, in an alternation, the first
# variant will be tried (and accepted) earlier. Second, if we reverse
# them, by the point it is clear ``'='`` will not be accepted, ``test``
# has already been consumed.
#
# The way we fix this is by reordering rules so that longest match
# comes first, and adding backtracking on alternations (as well as
# plus and star, since those have a hidden alternation inside).
#
# While backtracking can in principle make asymptotical complexity
# worse, it never makes parsing syntactically correct code supralinear
# with Python's LL(1) grammar, and we could not come up with any
# pathological incorrect input as well.

# Coverage data
_all_rules = []

# Generic LL parsing combinators
class Unmatched:
    def __init__(self, diagnostic=None):
        self.diagnostic = diagnostic

    def __repr__(self):
        if self.diagnostic:
            return "<can't parse: %s>" % repr(self.diagnostic)
        else:
            return "<can't parse>"

unmatched = Unmatched()

def llrule(loc, expected, cases=1):
    if loc is None:
        def decorator(rule):
            rule.expected = expected
            return rule
    else:
        def decorator(inner_rule):
            if cases == 1:
                def rule(*args, **kwargs):
                    result = inner_rule(*args, **kwargs)
                    if not isinstance(result, Unmatched):
                        rule.covered[0] = True
                    return result
            else:
                rule = inner_rule

            rule.loc, rule.expected, rule.covered = \
                loc, expected, [False] * cases
            _all_rules.append(rule)

            return rule
    return decorator

def action(inner_rule, loc=None):
    """
    A decorator returning a function that first runs ``inner_rule`` and then, if its
    return value is not None, maps that value using ``mapper``.

    If the value being mapped is a tuple, it is expanded into multiple arguments.

    Similar to attaching semantic actions to rules in traditional parser generators.
    """
    def decorator(mapper):
        @llrule(loc, inner_rule.expected)
        def outer_rule(parser):
            result = inner_rule(parser)
            if isinstance(result, Unmatched):
                return result
            if isinstance(result, tuple):
                return mapper(parser, *result)
            else:
                return mapper(parser, result)
        return outer_rule
    return decorator

def Eps(value=None, loc=None):
    """A rule that accepts no tokens (epsilon) and returns ``value``."""
    @llrule(loc, lambda parser: [])
    def rule(parser):
        return value
    return rule

def Tok(kind, loc=None):
    """A rule that accepts a token of kind ``kind`` and returns it, or returns None."""
    @llrule(loc, lambda parser: [kind])
    def rule(parser):
        return parser._accept(kind)
    return rule

def Loc(kind, loc=None):
    """A rule that accepts a token of kind ``kind`` and returns its location, or returns None."""
    @llrule(loc, lambda parser: [kind])
    def rule(parser):
        result = parser._accept(kind)
        if isinstance(result, Unmatched):
            return result
        return result.loc
    return rule

def Rule(name, loc=None):
    """A proxy for a rule called ``name`` which may not be yet defined."""
    @llrule(loc, lambda parser: getattr(parser, name).expected(parser))
    def rule(parser):
        return getattr(parser, name)()
    return rule

def Expect(inner_rule, loc=None):
    """A rule that executes ``inner_rule`` and emits a diagnostic error if it returns None."""
    @llrule(loc, inner_rule.expected)
    def rule(parser):
        result = inner_rule(parser)
        if isinstance(result, Unmatched):
            expected = inner_rule.expected(parser)
            if len(expected) > 1:
                expected = ' or '.join([', '.join(expected[0:-1]), expected[-1]])
            elif len(expected) == 1:
                expected = expected[0]
            else:
                expected = '(impossible)'
            error = diagnostic.Diagnostic(
                "error", "unexpected {actual}: expected {expected}",
                {'actual': parser.token.kind, 'expected': expected},
                parser.token.loc)
            return Unmatched(diagnostic.DiagnosticException(error))
        return result
    return rule

def Seq(first_rule, *rest_of_rules, **kwargs):
    """
    A rule that accepts a sequence of tokens satisfying ``rules`` and returns a tuple
    containing their return values, or None if the first rule was not satisfied.
    """
    @llrule(kwargs.get('loc', None), first_rule.expected)
    def rule(parser):
        result = first_rule(parser)
        if isinstance(result, Unmatched):
            return result

        results = [result]
        for rule in rest_of_rules:
            result = rule(parser)
            if isinstance(result, Unmatched):
                return result
            results.append(result)
        return tuple(results)
    return rule

def SeqN(n, *inner_rules, **kwargs):
    """
    A rule that accepts a sequence of tokens satisfying ``rules`` and returns
    the value returned by rule number ``n``, or None if the first rule was not satisfied.
    """
    @action(Seq(*inner_rules), loc=kwargs.get('loc', None))
    def rule(parser, *values):
        return values[n]
    return rule

def Alt(*inner_rules, **kwargs):
    """
    A rule that expects a sequence of tokens satisfying one of ``rules`` in sequence
    (a rule is satisfied when it returns anything but None) and returns the return
    value of that rule, or None if no rules were satisfied.
    """
    loc = kwargs.get('loc', None)
    expected = lambda parser: reduce(list.__add__, map(lambda x: x.expected(parser), inner_rules))
    if loc is not None:
        @llrule(loc, expected, cases=len(inner_rules))
        def rule(parser):
            data = parser._save()
            for idx, inner_rule in enumerate(inner_rules):
                result = inner_rule(parser)
                if isinstance(result, Unmatched):
                    parser._restore(data)
                else:
                    rule.covered[idx] = True
                    return result
            return unmatched
    else:
        @llrule(loc, expected, cases=len(inner_rules))
        def rule(parser):
            data = parser._save()
            for inner_rule in inner_rules:
                result = inner_rule(parser)
                if isinstance(result, Unmatched):
                    parser._restore(data)
                else:
                    return result
            return unmatched
    return rule

def Opt(inner_rule, loc=None):
    """Shorthand for ``Alt(inner_rule, Eps())``"""
    return Alt(inner_rule, Eps(), loc=loc)

def Star(inner_rule, loc=None):
    """
    A rule that accepts a sequence of tokens satisfying ``inner_rule`` zero or more times,
    and returns the returned values in a :class:`list`.
    """
    @llrule(loc, lambda parser: [])
    def rule(parser):
        results = []
        while True:
            data = parser._save()
            result = inner_rule(parser)
            if isinstance(result, Unmatched):
                parser._restore(data)
                return results
            results.append(result)
    return rule

def Plus(inner_rule, loc=None):
    """
    A rule that accepts a sequence of tokens satisfying ``inner_rule`` one or more times,
    and returns the returned values in a :class:`list`.
    """
    @llrule(loc, inner_rule.expected)
    def rule(parser):
        result = inner_rule(parser)
        if isinstance(result, Unmatched):
            return result

        results = [result]
        while True:
            data = parser._save()
            result = inner_rule(parser)
            if isinstance(result, Unmatched):
                parser._restore(data)
                return results
            results.append(result)
    return rule

class commalist(list):
    __slots__ = ('trailing_comma',)

def List(inner_rule, separator_tok, trailing, leading=True, loc=None):
    if not trailing:
        @action(Seq(inner_rule, Star(SeqN(1, Tok(separator_tok), inner_rule))), loc=loc)
        def outer_rule(parser, first, rest):
            return [first] + rest
        return outer_rule
    else:
        # A rule like this: stmt (';' stmt)* [';']
        # This doesn't yield itself to combinators above, because disambiguating
        # another iteration of the Kleene star and the trailing separator
        # requires two lookahead tokens (naively).
        separator_rule = Tok(separator_tok)
        @llrule(loc, inner_rule.expected)
        def rule(parser):
            results = commalist()

            if leading:
                result = inner_rule(parser)
                if isinstance(result, Unmatched):
                    return result
                else:
                    results.append(result)

            while True:
                result = separator_rule(parser)
                if isinstance(result, Unmatched):
                    results.trailing_comma = None
                    return results

                result_1 = inner_rule(parser)
                if isinstance(result_1, Unmatched):
                    results.trailing_comma = result
                    return results
                else:
                    results.append(result_1)
        return rule

# Python AST specific parser combinators
def Newline(loc=None):
    """A rule that accepts token of kind ``newline`` and returns an empty list."""
    @llrule(loc, lambda parser: ['newline'])
    def rule(parser):
        result = parser._accept('newline')
        if isinstance(result, Unmatched):
            return result
        return []
    return rule

def Oper(klass, *kinds, **kwargs):
    """
    A rule that accepts a sequence of tokens of kinds ``kinds`` and returns
    an instance of ``klass`` with ``loc`` encompassing the entire sequence
    or None if the first token is not of ``kinds[0]``.
    """
    @action(Seq(*map(Loc, kinds)), loc=kwargs.get('loc', None))
    def rule(parser, *tokens):
        return klass(loc=tokens[0].join(tokens[-1]))
    return rule

def BinOper(expr_rulename, op_rule, node=ast.BinOp, loc=None):
    @action(Seq(Rule(expr_rulename), Star(Seq(op_rule, Rule(expr_rulename)))), loc=loc)
    def rule(parser, lhs, trailers):
        for (op, rhs) in trailers:
            lhs = node(left=lhs, op=op, right=rhs,
                       loc=lhs.loc.join(rhs.loc))
        return lhs
    return rule

def BeginEnd(begin_tok, inner_rule, end_tok, empty=None, loc=None):
    @action(Seq(Loc(begin_tok), inner_rule, Loc(end_tok)), loc=loc)
    def rule(parser, begin_loc, node, end_loc):
        if node is None:
            node = empty()

        # Collection nodes don't have loc yet. If a node has loc at this
        # point, it means it's an expression passed in parentheses.
        if node.loc is None and type(node) in [
                ast.List, ast.Dict, ast.Tuple, ast.Repr,
                ast.ListComp, ast.GeneratorExp,
                ast.Call, ast.Subscript,
                ast.arguments]:
            node.begin_loc, node.end_loc, node.loc = \
                begin_loc, end_loc, begin_loc.join(end_loc)
        return node
    return rule

class Parser:

    # Generic LL parsing methods
    def __init__(self, lexer):
        self.lexer   = lexer
        self._tokens = []
        self._index  = -1
        self._advance()

    def _save(self):
        return self._index

    def _restore(self, data):
        self._index = data
        self._token = self._tokens[self._index]

    def _advance(self):
        self._index += 1
        if self._index == len(self._tokens):
            self._tokens.append(self.lexer.next(eof_token=True))
        self._token = self._tokens[self._index]

    def _accept(self, expected_kind):
        if self._token.kind == expected_kind:
            result = self._token
            self._advance()
            return result
        return unmatched

    # Helper methods
    def _wrap_tuple(self, elts):
        assert len(elts) > 0
        if len(elts) > 1:
            return ast.Tuple(ctx=None, elts=elts,
                             loc=elts[0].loc.join(elts[-1].loc), begin_loc=None, end_loc=None)
        else:
            return elts[0]

    def _assignable(self, node):
        # TODO
        return node

    def _empty_arguments(self):
        return ast.arguments(args=[], defaults=[], vararg=None, kwarg=None,
                             star_loc=None, vararg_loc=None, dstar_loc=None, kwarg_loc=None,
                             equals_locs=[], begin_loc=None, end_loc=None, loc=None)

    def _empty_arglist(self):
        return ast.Call(args=[], keywords=[], starargs=None, kwargs=None,
                        star_loc=None, dstar_loc=None, loc=None)

    # Python-specific methods
    def add_flags(self, flags):
        if 'print_function' in flags:
            self.lexer.print_function = True

    @action(Alt(Newline(),
                Rule('simple_stmt'),
                SeqN(0, Rule('compound_stmt'), Newline())))
    def single_input(self, body):
        """single_input: NEWLINE | simple_stmt | compound_stmt NEWLINE"""
        loc = None if body == [] else body[0].loc
        return ast.Interactive(body=body, loc=loc)

    @action(SeqN(0, Star(Alt(Newline(), Rule('stmt'))), Tok('eof')))
    def file_input(parser, body):
        """file_input: (NEWLINE | stmt)* ENDMARKER"""
        body = reduce(list.__add__, body, [])
        loc = None if body == [] else body[0].loc
        return ast.Module(body=body, loc=loc)

    @action(SeqN(0, Rule('testlist'), Star(Tok('newline')), Tok('eof')))
    def eval_input(self, expr):
        """eval_input: testlist NEWLINE* ENDMARKER"""
        return ast.Expression(body=[expr], loc=expr.loc)

    @action(Opt(Rule('arglist')))
    def decorator_1(self, args):
        if args is None:
            return self._empty_arglist()
        return args

    @action(Seq(Loc('@'), Rule('dotted_name'), Opt(BeginEnd('(', decorator_1, ')')),
            Loc('newline')))
    def decorator(self, at_loc, dotted_name, call_opt, newline_loc):
        """decorator: '@' dotted_name [ '(' [arglist] ')' ] NEWLINE"""
        name_loc, name = dotted_name
        expr = ast.Name(id=name, ctx=None, loc=name_loc)
        if call_opt:
            call_opt.func = expr
            call_opt.loc = name_loc.join(call_opt.loc)
            expr = call_opt
        return at_loc, expr

    decorators = Plus(Rule('decorator'))
    """decorators: decorator+"""

    @action(Seq(Rule('decorators'), Alt(Rule('classdef'), Rule('funcdef'))))
    def decorated(self, decorators, classfuncdef):
        """decorated: decorators (classdef | funcdef)"""
        classfuncdef.at_locs = list(map(lambda x: x[0], decorators))
        classfuncdef.decorator_list = list(map(lambda x: x[1], decorators))
        classfuncdef.loc = classfuncdef.loc.join(decorators[0][0])
        return classfuncdef

    @action(Seq(Loc('def'), Tok('ident'), Rule('parameters'), Loc(':'), Rule('suite')))
    def funcdef(self, def_loc, ident_tok, args, colon_loc, suite):
        """funcdef: 'def' NAME parameters ':' suite"""
        return ast.FunctionDef(name=ident_tok.value, args=args, body=suite, decorator_list=[],
                               at_locs=[], keyword_loc=def_loc, name_loc=ident_tok.loc,
                               colon_loc=colon_loc, loc=def_loc.join(suite[-1].loc))

    @action(Opt(Rule('varargslist')))
    def parameters_1(self, args):
        if args is None:
            args = self._empty_arguments()
        return args

    parameters = BeginEnd('(', parameters_1, ')')
    """parameters: '(' [varargslist] ')'"""

    varargslist_1 = Seq(Rule('fpdef'), Opt(Seq(Loc('='), Rule('test'))))

    @action(Seq(Loc('**'), Tok('ident')))
    def varargslist_2(self, dstar_loc, kwarg_tok):
        return ast.arguments(args=[], defaults=[], vararg=None, kwarg=kwarg_tok.value,
                             star_loc=None, vararg_loc=None,
                             dstar_loc=dstar_loc, kwarg_loc=kwarg_tok.loc,
                             begin_loc=None, end_loc=None, equals_locs=[],
                             loc=dstar_loc.join(kwarg_tok.loc))

    @action(Seq(Loc('*'), Tok('ident'),
                Opt(Seq(Tok(','), Loc('**'), Tok('ident')))))
    def varargslist_3(self, star_loc, vararg_tok, kwarg_opt):
        dstar_loc = kwarg = kwarg_loc = None
        loc = star_loc.join(vararg_tok.loc)
        if kwarg_opt:
            _, dstar_loc, kwarg_tok = kwarg_opt
            kwarg, kwarg_loc = kwarg_tok.value, kwarg_tok.loc
            loc = star_loc.join(kwarg_tok.loc)
        return ast.arguments(args=[], defaults=[], vararg=vararg_tok.value, kwarg=kwarg,
                             star_loc=star_loc, vararg_loc=vararg_tok.loc,
                             dstar_loc=dstar_loc, kwarg_loc=kwarg_loc,
                             begin_loc=None, end_loc=None, equals_locs=[], loc=loc)

    @action(Eps(value=()))
    def varargslist_4(self):
        return self._empty_arguments()

    @action(Alt(Seq(Star(SeqN(0, varargslist_1, Tok(','))),
                    Alt(varargslist_2, varargslist_3)),
                Seq(List(varargslist_1, ',', trailing=True),
                    varargslist_4)))
    def varargslist(self, fparams, args):
        """varargslist: ((fpdef ['=' test] ',')*
                         ('*' NAME [',' '**' NAME] | '**' NAME) |
                         fpdef ['=' test] (',' fpdef ['=' test])* [','])"""
        for fparam, default_opt in fparams:
            args.args.append(fparam)
            if default_opt:
                equals_loc, default = default_opt
                args.equals_locs.append(equals_loc)
                args.defaults.append(default)
            elif len(args.defaults) > 0:
                error = diagnostic.Diagnostic(
                    "error", "non-default argument follows default argument", {}, fparam.loc)
                raise diagnostic.DiagnosticException(error)

        def fparam_loc(fparam, default_opt):
            if default_opt:
                equals_loc, default = default_opt
                return fparam.loc.join(default.loc)
            else:
                return fparam.loc

        if args.loc is None:
            args.loc = fparam_loc(*fparams[0]).join(fparam_loc(*fparams[-1]))
        elif len(fparams) > 0:
            args.loc = args.loc.join(fparam_loc(*fparams[0]))

        return args

    @action(Tok('ident'))
    def fpdef_1(self, ident_tok):
        return ast.Name(id=ident_tok.value, loc=ident_tok.loc, ctx=None)

    fpdef = Alt(fpdef_1, BeginEnd('(', Rule('fplist'), ')',
                                  empty=lambda: ast.Tuple(elts=[], ctx=None, loc=None)))
    """fpdef: NAME | '(' fplist ')'"""

    @action(List(Rule('fpdef'), ',', trailing=True))
    def fplist(self, elts):
        """fplist: fpdef (',' fpdef)* [',']"""
        return ast.Tuple(elts=elts, ctx=None, loc=None)

    stmt = Alt(Rule('simple_stmt'), Rule('compound_stmt'))
    """stmt: simple_stmt | compound_stmt"""

    simple_stmt = SeqN(0, List(Rule('small_stmt'), ';', trailing=True), Tok('newline'))
    """simple_stmt: small_stmt (';' small_stmt)* [';'] NEWLINE"""

    small_stmt = Alt(Rule('expr_stmt'), Rule('print_stmt'),  Rule('del_stmt'),
                     Rule('pass_stmt'), Rule('flow_stmt'), Rule('import_stmt'),
                     Rule('global_stmt'), Rule('exec_stmt'), Rule('assert_stmt'))
    """small_stmt: (expr_stmt | print_stmt  | del_stmt | pass_stmt | flow_stmt |
                    import_stmt | global_stmt | exec_stmt | assert_stmt)"""

    @action(Seq(Rule('augassign'), Alt(Rule('yield_expr'), Rule('testlist'))))
    def expr_stmt_1(self, augassign, rhs_expr):
        return ast.AugAssign(op=augassign, value=rhs_expr)

    @action(Star(Seq(Loc('='), Alt(Rule('yield_expr'), Rule('testlist')))))
    def expr_stmt_2(self, seq):
        if len(seq) > 0:
            return ast.Assign(targets=list(map(lambda x: x[1], seq[:-1])), value=seq[-1][1],
                              op_locs=list(map(lambda x: x[0], seq)))
        else:
            return None

    @action(Seq(Rule('testlist'), Alt(expr_stmt_1, expr_stmt_2)))
    def expr_stmt(self, lhs, rhs):
        """expr_stmt: testlist (augassign (yield_expr|testlist) |
                                ('=' (yield_expr|testlist))*)"""
        if isinstance(rhs, ast.AugAssign):
            if isinstance(lhs, ast.Tuple):
                error = diagnostic.Diagnostic(
                    "error", "illegal expression for augmented assignment", {}, rhs.loc)
                raise diagnostic.DiagnosticException(error)
            else:
                rhs.target = self._assignable(lhs)
                rhs.loc = rhs.target.loc.join(rhs.value.loc)
                return rhs
        elif rhs is not None:
            rhs.targets = list(map(self._assignable, [lhs] + rhs.targets))
            rhs.loc = lhs.loc.join(rhs.value.loc)
            return rhs
        else:
            return ast.Expr(value=lhs, loc=lhs.loc)

    augassign = Alt(Oper(ast.Add, '+='), Oper(ast.Sub, '-='), Oper(ast.Mult, '*='),
                    Oper(ast.Div, '/='), Oper(ast.Mod, '%='), Oper(ast.BitAnd, '&='),
                    Oper(ast.BitOr, '|='), Oper(ast.BitXor, '^='), Oper(ast.LShift, '<<='),
                    Oper(ast.RShift, '>>='), Oper(ast.Pow, '**='), Oper(ast.FloorDiv, '//='))
    """augassign: ('+=' | '-=' | '*=' | '/=' | '%=' | '&=' | '|=' | '^=' |
                   '<<=' | '>>=' | '**=' | '//=')"""

    @action(List(Rule('test'), ',', trailing=True))
    def print_stmt_1(self, values):
        loc = values.trailing_comma.loc if values.trailing_comma else values[-1].loc
        nl = False if values.trailing_comma else True
        return ast.Print(dest=None, values=values, nl=nl,
                         dest_loc=None, loc=loc)

    @action(Seq(Loc('>>'), Rule('test'), Tok(','), List(Rule('test'), ',', trailing=True)))
    def print_stmt_2(self, dest_loc, dest, comma_tok, values):
        loc = values.trailing_comma.loc if values.trailing_comma else values[-1].loc
        nl = False if values.trailing_comma else True
        return ast.Print(dest=dest, values=values, nl=nl,
                         dest_loc=dest_loc, loc=loc)

    @action(Seq(Loc('print'), Alt(print_stmt_1, print_stmt_2)))
    def print_stmt(self, print_loc, stmt):
        """
        (2.6-2.7)
        print_stmt: 'print' ( [ test (',' test)* [','] ] |
                              '>>' test [ (',' test)+ [','] ] )
        """
        stmt.keyword_loc = print_loc
        stmt.loc = print_loc.join(stmt.loc)
        return stmt

    @action(Seq(Loc('del'), List(Rule('expr'), ',', trailing=True)))
    def del_stmt(self, stmt_loc, exprs):
        # Python uses exprlist here, but does *not* obey the usual
        # tuple-wrapping semantics, so we embed the rule directly.
        """del_stmt: 'del' exprlist"""
        return ast.Delete(targets=list(map(self._assignable, exprs)),
                          loc=stmt_loc.join(exprs[-1].loc), keyword_loc=stmt_loc)

    @action(Loc('pass'))
    def pass_stmt(self, stmt_loc):
        """pass_stmt: 'pass'"""
        return ast.Pass(loc=stmt_loc, keyword_loc=stmt_loc)

    flow_stmt = Alt(Rule('break_stmt'), Rule('continue_stmt'), Rule('return_stmt'),
                    Rule('raise_stmt'), Rule('yield_stmt'))
    """flow_stmt: break_stmt | continue_stmt | return_stmt | raise_stmt | yield_stmt"""

    @action(Loc('break'))
    def break_stmt(self, stmt_loc):
        """break_stmt: 'break'"""
        return ast.Break(loc=stmt_loc, keyword_loc=stmt_loc)

    @action(Loc('continue'))
    def continue_stmt(self, stmt_loc):
        """continue_stmt: 'continue'"""
        return ast.Continue(loc=stmt_loc, keyword_loc=stmt_loc)

    @action(Seq(Loc('return'), Opt(Rule('testlist'))))
    def return_stmt(self, stmt_loc, values):
        """return_stmt: 'return' [testlist]"""
        loc = stmt_loc
        if values:
            loc = loc.join(values.loc)
        return ast.Return(value=values,
                          loc=loc, keyword_loc=stmt_loc)

    @action(Rule('yield_expr'))
    def yield_stmt(self, expr):
        """yield_stmt: yield_expr"""
        return ast.Expr(value=expr, loc=expr.loc)

    @action(Seq(Loc('raise'), Opt(Seq(Rule('test'),
                                      Opt(Seq(Tok(','), Rule('test'),
                                              Opt(SeqN(1, Tok(','), Rule('test')))))))))
    def raise_stmt(self, raise_loc, type_opt):
        """raise_stmt: 'raise' [test [',' test [',' test]]]"""
        type_ = inst = tback = None
        loc = raise_loc
        if type_opt:
            type_, inst_opt = type_opt
            loc = loc.join(type_.loc)
            if inst_opt:
                _, inst, tback = inst_opt
                loc = loc.join(inst.loc)
                if tback:
                    loc = loc.join(tback.loc)
        return ast.Raise(type=type_, inst=inst, tback=tback,
                         keyword_loc=raise_loc, loc=loc)

    import_stmt = Alt(Rule('import_name'), Rule('import_from'))
    """import_stmt: import_name | import_from"""

    @action(Seq(Loc('import'), Rule('dotted_as_names')))
    def import_name(self, import_loc, names):
        """import_name: 'import' dotted_as_names"""
        return ast.Import(names=names,
                          keyword_loc=import_loc, loc=import_loc.join(names[-1].loc))

    @action(Seq(Star(Loc('.')), Rule('dotted_name')))
    def import_from_1(self, dots, dotted_name):
        return dots, dotted_name

    @action(Plus(Loc('.')))
    def import_from_2(self, dots):
        return dots, None

    @action(Loc('*'))
    def import_from_3(self, star_loc):
        return None, \
               [ast.alias(name='*', asname=None,
                          name_loc=star_loc, as_loc=None, asname_loc=None, loc=star_loc)], \
               None

    @action(Rule('import_as_names'))
    def import_from_4(self, names):
        return None, names, None

    @action(Seq(Loc('from'), Alt(import_from_1, import_from_2),
                Loc('import'), Alt(import_from_3,
                                   Seq(Loc('('), Rule('import_as_names'), Loc(')')),
                                   import_from_4)))
    def import_from(self, from_loc, module_name, import_loc, names):
        """import_from: ('from' ('.'* dotted_name | '.'+)
                         'import' ('*' | '(' import_as_names ')' | import_as_names))"""
        dots, dotted_name_opt = module_name
        module_loc = module = None
        if dotted_name_opt:
            module_loc, module = dotted_name_opt
        lparen_loc, names, rparen_loc = names
        dots_loc = None
        if dots != []:
            dots_loc = dots[0].join(dots[-1])
        loc = from_loc.join(names[-1].loc)
        if rparen_loc:
            loc = loc.join(rparen_loc)

        if module == '__future__':
            self.add_flags([x.name for x in names])

        return ast.ImportFrom(names=names, module=module, level=len(dots),
                              keyword_loc=from_loc, dots_loc=dots_loc, module_loc=module_loc,
                              import_loc=import_loc, lparen_loc=lparen_loc, rparen_loc=rparen_loc,
                              loc=loc)

    @action(Seq(Tok('ident'), Opt(Seq(Loc('as'), Tok('ident')))))
    def import_as_name(self, name_tok, as_name_opt):
        """import_as_name: NAME ['as' NAME]"""
        asname_name = asname_loc = as_loc = None
        loc = name_tok.loc
        if as_name_opt:
            as_loc, asname = as_name_opt
            asname_name = asname.value
            asname_loc = asname.loc
            loc = loc.join(asname.loc)
        return ast.alias(name=name_tok.value, asname=asname_name,
                         loc=loc, name_loc=name_tok.loc, as_loc=as_loc, asname_loc=asname_loc)

    @action(Seq(Rule('dotted_name'), Opt(Seq(Loc('as'), Tok('ident')))))
    def dotted_as_name(self, dotted_name, as_name_opt):
        """dotted_as_name: dotted_name ['as' NAME]"""
        asname_name = asname_loc = as_loc = None
        dotted_name_loc, dotted_name_name = dotted_name
        loc = dotted_name_loc
        if as_name_opt:
            as_loc, asname = as_name_opt
            asname_name = asname.value
            asname_loc = asname.loc
            loc = loc.join(asname.loc)
        return ast.alias(name=dotted_name_name, asname=asname_name,
                         loc=loc, name_loc=dotted_name_loc, as_loc=as_loc, asname_loc=asname_loc)

    import_as_names = List(Rule('import_as_name'), ',', trailing=True)
    """import_as_names: import_as_name (',' import_as_name)* [',']"""

    dotted_as_names = List(Rule('dotted_as_name'), ',', trailing=False)
    """dotted_as_names: dotted_as_name (',' dotted_as_name)*"""

    @action(List(Tok('ident'), '.', trailing=False))
    def dotted_name(self, idents):
        """dotted_name: NAME ('.' NAME)*"""
        return idents[0].loc.join(idents[-1].loc), \
               '.'.join(list(map(lambda x: x.value, idents)))

    @action(Seq(Loc('global'), List(Tok('ident'), ',', trailing=False)))
    def global_stmt(self, global_loc, names):
        """global_stmt: 'global' NAME (',' NAME)*"""
        return ast.Global(names=list(map(lambda x: x.value, names)),
                          name_locs=list(map(lambda x: x.loc, names)),
                          keyword_loc=global_loc, loc=global_loc.join(names[-1].loc))

    @action(Seq(Loc('exec'), Rule('expr'),
                Opt(Seq(Loc('in'), Rule('test'),
                        Opt(SeqN(1, Loc(','), Rule('test')))))))
    def exec_stmt(self, exec_loc, body, in_opt):
        """exec_stmt: 'exec' expr ['in' test [',' test]]"""
        in_loc, globals, locals = None, None, None
        loc = exec_loc.join(body.loc)
        if in_opt:
            in_loc, globals, locals = in_opt
            if locals:
                loc = loc.join(locals.loc)
            else:
                loc = loc.join(globals.loc)
        return ast.Exec(body=body, locals=locals, globals=globals,
                        loc=loc, keyword_loc=exec_loc, in_loc=in_loc)

    @action(Seq(Loc('assert'), Rule('test'), Opt(SeqN(1, Tok(','), Rule('test')))))
    def assert_stmt(self, assert_loc, test, msg):
        """assert_stmt: 'assert' test [',' test]"""
        loc = assert_loc.join(test.loc)
        if msg:
            loc = loc.join(msg.loc)
        return ast.Assert(test=test, msg=msg,
                          loc=loc, keyword_loc=assert_loc)

    @action(Alt(Rule('if_stmt'), Rule('while_stmt'), Rule('for_stmt'),
                Rule('try_stmt'), Rule('with_stmt'), Rule('funcdef'),
                Rule('classdef'), Rule('decorated')))
    def compound_stmt(self, stmt):
        """compound_stmt: if_stmt | while_stmt | for_stmt | try_stmt | with_stmt |
                          funcdef | classdef | decorated"""
        return [stmt]

    @action(Seq(Loc('if'), Rule('test'), Loc(':'), Rule('suite'),
                Star(Seq(Loc('elif'), Rule('test'), Loc(':'), Rule('suite'))),
                Opt(Seq(Loc('else'), Loc(':'), Rule('suite')))))
    def if_stmt(self, if_loc, test, if_colon_loc, body, elifs, else_opt):
        """if_stmt: 'if' test ':' suite ('elif' test ':' suite)* ['else' ':' suite]"""
        stmt = ast.If(orelse=[],
                      else_loc=None, else_colon_loc=None)

        if else_opt:
            stmt.else_loc, stmt.else_colon_loc, stmt.orelse = else_opt

        for elif_ in elifs:
            stmt.keyword_loc, stmt.test, stmt.if_colon_loc, stmt.body = elif_
            stmt.loc = stmt.keyword_loc.join(stmt.body[-1].loc)
            if stmt.orelse: stmt.loc = stmt.loc.join(stmt.orelse[-1].loc)
            stmt = ast.If(orelse=[stmt],
                          else_loc=None, else_colon_loc=None)

        stmt.keyword_loc, stmt.test, stmt.if_colon_loc, stmt.body = \
            if_loc, test, if_colon_loc, body
        stmt.loc = stmt.keyword_loc.join(stmt.body[-1].loc)
        if stmt.orelse: stmt.loc = stmt.loc.join(stmt.orelse[-1].loc)
        return stmt

    @action(Seq(Loc('while'), Rule('test'), Loc(':'), Rule('suite'),
                Opt(Seq(Loc('else'), Loc(':'), Rule('suite')))))
    def while_stmt(self, while_loc, test, while_colon_loc, body, else_opt):
        """while_stmt: 'while' test ':' suite ['else' ':' suite]"""
        stmt = ast.While(test=test, body=body, orelse=[],
                         keyword_loc=while_loc, while_colon_loc=while_colon_loc,
                         else_loc=None, else_colon_loc=None,
                         loc=while_loc.join(body[-1].loc))
        if else_opt:
            stmt.else_loc, stmt.else_colon_loc, stmt.orelse = else_opt
            stmt.loc = stmt.loc.join(stmt.orelse[-1].loc)

        return stmt

    @action(Seq(Loc('for'), Rule('exprlist'), Loc('in'), Rule('testlist'),
                Loc(':'), Rule('suite'),
                Opt(Seq(Loc('else'), Loc(':'), Rule('suite')))))
    def for_stmt(self, for_loc, target, in_loc, iter, for_colon_loc, body, else_opt):
        """for_stmt: 'for' exprlist 'in' testlist ':' suite ['else' ':' suite]"""
        stmt = ast.For(target=self._assignable(target), iter=iter, body=body, orelse=[],
                       keyword_loc=for_loc, in_loc=in_loc, for_colon_loc=for_colon_loc,
                       else_loc=None, else_colon_loc=None,
                       loc=for_loc.join(body[-1].loc))
        if else_opt:
            stmt.else_loc, stmt.else_colon_loc, stmt.orelse = else_opt
            stmt.loc = stmt.loc.join(stmt.orelse[-1].loc)

        return stmt

    @action(Seq(Plus(Seq(Rule('except_clause'), Loc(':'), Rule('suite'))),
                Opt(Seq(Loc('else'), Loc(':'), Rule('suite'))),
                Opt(Seq(Loc('finally'), Loc(':'), Rule('suite')))))
    def try_stmt_1(self, clauses, else_opt, finally_opt):
        handlers = []
        for clause in clauses:
            handler, handler.colon_loc, handler.body = clause
            handler.loc = handler.loc.join(handler.body[-1].loc)
            handlers.append(handler)

        else_loc, else_colon_loc, orelse = None, None, []
        loc = handlers[-1].loc
        if else_opt:
            else_loc, else_colon_loc, orelse = else_opt
            loc = orelse[-1].loc

        stmt = ast.TryExcept(body=None, handlers=handlers, orelse=orelse,
                             else_loc=else_loc, else_colon_loc=else_colon_loc,
                             loc=loc)
        if finally_opt:
            finally_loc, finally_colon_loc, finalbody = finally_opt
            return ast.TryFinally(body=[stmt], finalbody=finalbody,
                                  finally_loc=finally_loc, finally_colon_loc=finally_colon_loc,
                                  loc=finalbody[-1].loc)
        else:
            return stmt

    @action(Seq(Loc('finally'), Loc(':'), Rule('suite')))
    def try_stmt_2(self, finally_loc, finally_colon_loc, finalbody):
        return ast.TryFinally(body=None, finalbody=finalbody,
                              finally_loc=finally_loc, finally_colon_loc=finally_colon_loc,
                              loc=finalbody[-1].loc)

    @action(Seq(Loc('try'), Loc(':'), Rule('suite'), Alt(try_stmt_1, try_stmt_2)))
    def try_stmt(self, try_loc, try_colon_loc, body, stmt):
        """
        try_stmt: ('try' ':' suite
                   ((except_clause ':' suite)+
                    ['else' ':' suite]
                    ['finally' ':' suite] |
                    'finally' ':' suite))
        """
        stmt.keyword_loc, stmt.try_colon_loc = try_loc, try_colon_loc
        if stmt.body is None: # try..finally or try..except
            stmt.body = body
        else: # try..except..finally
            stmt.body[0].keyword_loc, stmt.body[0].try_colon_loc, stmt.body[0].body = \
                try_loc, try_colon_loc, body
            stmt.body[0].loc = stmt.body[0].loc.join(try_loc)
        stmt.loc = stmt.loc.join(try_loc)
        return stmt

    @action(Seq(Loc('with'), Rule('test'), Opt(Rule('with_var')), Loc(':'), Rule('suite')))
    def with_stmt(self, with_loc, context, with_var, colon_loc, body):
        """with_stmt: 'with' test [ with_var ] ':' suite"""
        as_loc = optional_vars = None
        if with_var:
            as_loc, optional_vars = with_var
        return ast.With(context_expr=context, optional_vars=optional_vars, body=body,
                        keyword_loc=with_loc, as_loc=as_loc, colon_loc=colon_loc,
                        loc=with_loc.join(body[-1].loc))

    with_var = Seq(Loc('as'), Rule('expr'))
    """with_var: 'as' expr"""

    @action(Seq(Loc('except'),
                Opt(Seq(Rule('test'),
                        Opt(Seq(Alt(Loc('as'), Loc(',')), Rule('test')))))))
    def except_clause(self, except_loc, exc_opt):
        """except_clause: 'except' [test [('as' | ',') test]]"""
        type_ = name = as_loc = None
        loc = except_loc
        if exc_opt:
            type_, name_opt = exc_opt
            loc = loc.join(type_.loc)
            if name_opt:
                as_loc, name = name_opt
                loc = loc.join(name.loc)
        return ast.ExceptHandler(type=type_, name=name,
                                 except_loc=except_loc, as_loc=as_loc, loc=loc)

    @action(Plus(Rule('stmt')))
    def suite_1(self, stmts):
        return reduce(list.__add__, stmts, [])

    suite = Alt(Rule('simple_stmt'),
                SeqN(2, Tok('newline'), Tok('indent'), suite_1, Tok('dedent')))
    """suite: simple_stmt | NEWLINE INDENT stmt+ DEDENT"""

    # 2.x-only backwards compatibility start
    testlist_safe = action(List(Rule('old_test'), ',', trailing=False))(_wrap_tuple)
    """testlist_safe: old_test [(',' old_test)+ [',']]"""

    old_test = Alt(Rule('or_test'), Rule('old_lambdef'))
    """old_test: or_test | old_lambdef"""

    @action(Seq(Loc('lambda'), Opt(Rule('varargslist')), Loc(':'), Rule('old_test')))
    def old_lambdef(self, lambda_loc, args_opt, colon_loc, body):
        """old_lambdef: 'lambda' [varargslist] ':' old_test"""
        if args_opt is None:
            args_opt = self._empty_arguments()
            args_opt.loc = colon_loc.begin()
        return ast.Lambda(args=args_opt, body=body,
                          lambda_loc=lambda_loc, colon_loc=colon_loc,
                          loc=lambda_loc.join(body.loc))
    # 2.x-only backwards compatibility end

    @action(Seq(Rule('or_test'), Opt(Seq(Loc('if'), Rule('or_test'),
                                         Loc('else'), Rule('test')))))
    def test_1(self, lhs, rhs_opt):
        if rhs_opt is not None:
            if_loc, test, else_loc, orelse = rhs_opt
            return ast.IfExp(test=test, body=lhs, orelse=orelse,
                             if_loc=if_loc, else_loc=else_loc, loc=lhs.loc.join(orelse.loc))
        return lhs

    test = Alt(test_1, Rule('lambdef'))
    """test: or_test ['if' or_test 'else' test] | lambdef"""

    @action(Seq(Rule('and_test'), Star(Seq(Loc('or'), Rule('and_test')))))
    def or_test(self, lhs, rhs):
        """or_test: and_test ('or' and_test)*"""
        if len(rhs) > 0:
            return ast.BoolOp(op=ast.Or(),
                              values=[lhs] + list(map(lambda x: x[1], rhs)),
                              loc=lhs.loc.join(rhs[-1][1].loc),
                              op_locs=list(map(lambda x: x[0], rhs)))
        else:
            return lhs

    @action(Seq(Rule('not_test'), Star(Seq(Loc('and'), Rule('not_test')))))
    def and_test(self, lhs, rhs):
        """and_test: not_test ('and' not_test)*"""
        if len(rhs) > 0:
            return ast.BoolOp(op=ast.And(),
                              values=[lhs] + list(map(lambda x: x[1], rhs)),
                              loc=lhs.loc.join(rhs[-1][1].loc),
                              op_locs=list(map(lambda x: x[0], rhs)))
        else:
            return lhs

    @action(Seq(Oper(ast.Not, 'not'), Rule('not_test')))
    def not_test_1(self, op, operand):
        return ast.UnaryOp(op=op, operand=operand,
                           loc=op.loc.join(operand.loc))

    not_test = Alt(not_test_1, Rule('comparison'))
    """not_test: 'not' not_test | comparison"""

    @action(Seq(Rule('expr'), Star(Seq(Rule('comp_op'), Rule('expr')))))
    def comparison(self, lhs, rhs):
        """comparison: expr (comp_op expr)*"""
        if len(rhs) > 0:
            return ast.Compare(left=lhs, ops=list(map(lambda x: x[0], rhs)),
                               comparators=list(map(lambda x: x[1], rhs)),
                               loc=lhs.loc.join(rhs[-1][1].loc))
        else:
            return lhs

    comp_op = Alt(Oper(ast.Lt, '<'), Oper(ast.Gt, '>'), Oper(ast.Eq, '=='),
                  Oper(ast.GtE, '>='), Oper(ast.LtE, '<='), Oper(ast.NotEq, '<>'),
                  Oper(ast.NotEq, '!='),
                  Oper(ast.In, 'in'), Oper(ast.NotIn, 'not', 'in'),
                  Oper(ast.IsNot, 'is', 'not'), Oper(ast.Is, 'is'))
    """comp_op: '<'|'>'|'=='|'>='|'<='|'<>'|'!='|'in'|'not' 'in'|'is'|'is' 'not'"""

    expr = BinOper('xor_expr', Oper(ast.BitOr, '|'))
    """expr: xor_expr ('|' xor_expr)*"""

    xor_expr = BinOper('and_expr', Oper(ast.BitXor, '^'))
    """xor_expr: and_expr ('^' and_expr)*"""

    and_expr = BinOper('shift_expr', Oper(ast.BitAnd, '&'))
    """and_expr: shift_expr ('&' shift_expr)*"""

    shift_expr = BinOper('arith_expr', Alt(Oper(ast.LShift, '<<'), Oper(ast.RShift, '>>')))
    """shift_expr: arith_expr (('<<'|'>>') arith_expr)*"""

    arith_expr = BinOper('term', Alt(Oper(ast.Add, '+'), Oper(ast.Sub, '-')))
    """arith_expr: term (('+'|'-') term)*"""

    term = BinOper('factor', Alt(Oper(ast.Mult, '*'), Oper(ast.Div, '/'),
                                 Oper(ast.Mod, '%'), Oper(ast.FloorDiv, '//')))
    """term: factor (('*'|'/'|'%'|'//') factor)*"""

    @action(Seq(Alt(Oper(ast.UAdd, '+'), Oper(ast.USub, '-'), Oper(ast.Invert, '~')),
                Rule('factor')))
    def factor_1(self, op, factor):
        return ast.UnaryOp(op=op, operand=factor,
                           loc=op.loc.join(factor.loc))

    factor = Alt(factor_1, Rule('power'))
    """factor: ('+'|'-'|'~') factor | power"""

    @action(Seq(Rule('atom'), Star(Rule('trailer')), Opt(Seq(Loc('**'), Rule('factor')))))
    def power(self, atom, trailers, factor_opt):
        """power: atom trailer* ['**' factor]"""
        for trailer in trailers:
            if isinstance(trailer, ast.Attribute) or isinstance(trailer, ast.Subscript):
                trailer.value = atom
            elif isinstance(trailer, ast.Call):
                trailer.func = atom
            trailer.loc = atom.loc.join(trailer.loc)
            atom = trailer
        if factor_opt:
            op_loc, factor = factor_opt
            return ast.BinOp(left=atom, op=ast.Pow(loc=op_loc), right=factor,
                             loc=atom.loc.join(factor.loc))
        return atom

    @action(Rule('testlist1'))
    def atom_1(self, expr):
        return ast.Repr(value=expr, loc=None)

    @action(Tok('ident'))
    def atom_2(self, tok):
        return ast.Name(id=tok.value, loc=tok.loc, ctx=None)

    @action(Alt(Tok('int'), Tok('float'), Tok('complex')))
    def atom_3(self, tok):
        return ast.Num(n=tok.value, loc=tok.loc)

    @action(Seq(Tok('strbegin'), Tok('strdata'), Tok('strend')))
    def atom_4(self, begin_tok, data_tok, end_tok):
        return ast.Str(s=data_tok.value,
                       begin_loc=begin_tok.loc, end_loc=end_tok.loc,
                       loc=begin_tok.loc.join(end_tok.loc))

    @action(Plus(atom_4))
    def atom_5(self, strings):
        return ast.Str(s=''.join([x.s for x in strings]),
                       begin_loc=strings[0].begin_loc, end_loc=strings[-1].end_loc,
                       loc=strings[0].loc.join(strings[-1].loc))

    atom = Alt(BeginEnd('(', Opt(Alt(Rule('yield_expr'), Rule('testlist_gexp'))), ')',
                        empty=lambda: ast.Tuple(elts=[], ctx=None, loc=None)),
               BeginEnd('[', Opt(Rule('listmaker')), ']',
                        empty=lambda: ast.List(elts=[], ctx=None, loc=None)),
               BeginEnd('{', Opt(Rule('dictmaker')), '}',
                        empty=lambda: ast.Dict(keys=[], values=[], colon_locs=[],
                                               ctx=None, loc=None)),
               BeginEnd('`', atom_1, '`'),
               atom_2, atom_3, atom_5)
    """atom: ('(' [yield_expr|testlist_gexp] ')' |
              '[' [listmaker] ']' |
              '{' [dictmaker] '}' |
              '`' testlist1 '`' |
              NAME | NUMBER | STRING+)"""

    def list_gen_action(self, lhs, rhs):
        if rhs is None: # (x)
            return lhs
        elif isinstance(rhs, ast.Tuple) or isinstance(rhs, ast.List):
            rhs.elts = [lhs] + rhs.elts
            return rhs
        elif isinstance(rhs, ast.ListComp) or isinstance(rhs, ast.GeneratorExp):
            rhs.elt = lhs
            return rhs

    @action(Rule('list_for'))
    def listmaker_1(self, compose):
        return ast.ListComp(generators=compose([]), loc=None)

    @action(List(Rule('test'), ',', trailing=True, leading=False))
    def listmaker_2(self, elts):
        return ast.List(elts=elts, ctx=None, loc=None)

    listmaker = action(
        Seq(Rule('test'),
            Alt(listmaker_1, listmaker_2))) \
        (list_gen_action)
    """listmaker: test ( list_for | (',' test)* [','] )"""

    @action(Rule('gen_for'))
    def testlist_gexp_1(self, compose):
        return ast.GeneratorExp(generators=compose([]), loc=None)

    @action(List(Rule('test'), ',', trailing=True, leading=False))
    def testlist_gexp_2(self, elts):
        if elts == [] and not elts.trailing_comma:
            return None
        else:
            return ast.Tuple(elts=elts, ctx=None, loc=None)

    testlist_gexp = action(
        Seq(Rule('test'), Alt(testlist_gexp_1, testlist_gexp_2))) \
        (list_gen_action)
    """testlist_gexp: test ( gen_for | (',' test)* [','] )"""

    @action(Seq(Loc('lambda'), Opt(Rule('varargslist')), Loc(':'), Rule('test')))
    def lambdef(self, lambda_loc, args_opt, colon_loc, body):
        """lambdef: 'lambda' [varargslist] ':' test"""
        if args_opt is None:
            args_opt = self._empty_arguments()
            args_opt.loc = colon_loc.begin()
        return ast.Lambda(args=args_opt, body=body,
                          lambda_loc=lambda_loc, colon_loc=colon_loc,
                          loc=lambda_loc.join(body.loc))

    @action(Seq(Loc('.'), Tok('ident')))
    def trailer_1(self, dot_loc, ident_tok):
        return ast.Attribute(attr=ident_tok.value, ctx=None,
                             loc=dot_loc.join(ident_tok.loc),
                             attr_loc=ident_tok.loc, dot_loc=dot_loc)

    @action(Opt(Rule('arglist')))
    def trailer_2(self, args):
        if args is None:
            return self._empty_arglist()
        return args

    trailer = Alt(BeginEnd('(', trailer_2, ')'),
                  BeginEnd('[', Rule('subscriptlist'), ']'),
                  trailer_1)
    """trailer: '(' [arglist] ')' | '[' subscriptlist ']' | '.' NAME"""

    @action(List(Rule('subscript'), ',', trailing=True))
    def subscriptlist(self, subscripts):
        """subscriptlist: subscript (',' subscript)* [',']"""
        if len(subscripts) == 1:
            return ast.Subscript(slice=subscripts[0], ctx=None, loc=None)
        elif all([isinstance(x, ast.Index) for x in subscripts]):
            elts  = [x.value for x in subscripts]
            loc   = subscripts[0].loc.join(subscripts[-1].loc)
            index = ast.Index(value=ast.Tuple(elts=elts, ctx=None,
                                              begin_loc=None, end_loc=None, loc=loc),
                              loc=loc)
            return ast.Subscript(slice=index, ctx=None, loc=None)
        else:
            extslice = ast.ExtSlice(dims=subscripts,
                                    loc=subscripts[0].loc.join(subscripts[-1].loc))
            return ast.Subscript(slice=extslice, ctx=None, loc=None)

    @action(Seq(Loc('.'), Loc('.'), Loc('.')))
    def subscript_1(self, dot_1_loc, dot_2_loc, dot_3_loc):
        return ast.Ellipsis(loc=dot_1_loc.join(dot_3_loc))

    @action(Seq(Opt(Rule('test')), Loc(':'), Opt(Rule('test')), Opt(Rule('sliceop'))))
    def subscript_2(self, lower_opt, colon_loc, upper_opt, step_opt):
        loc = colon_loc
        if lower_opt:
            loc = loc.join(lower_opt.loc)
        if upper_opt:
            loc = loc.join(upper_opt.loc)
        step_colon_loc = step = None
        if step_opt:
            step_colon_loc, step = step_opt
            loc = loc.join(step_colon_loc)
            if step:
                loc = loc.join(step.loc)
        return ast.Slice(lower=lower_opt, upper=upper_opt, step=step,
                         loc=loc, bound_colon_loc=colon_loc, step_colon_loc=step_colon_loc)

    @action(Rule('test'))
    def subscript_3(self, expr):
        return ast.Index(value=expr, loc=expr.loc)

    subscript = Alt(subscript_1, subscript_2, subscript_3)

    sliceop = Seq(Loc(':'), Opt(Rule('test')))
    """sliceop: ':' [test]"""

    @action(List(Rule('expr'), ',', trailing=True))
    def exprlist(self, exprs):
        """exprlist: expr (',' expr)* [',']"""
        return self._wrap_tuple(exprs)

    @action(List(Rule('test'), ',', trailing=True))
    def testlist(self, exprs):
        """testlist: test (',' test)* [',']"""
        return self._wrap_tuple(exprs)

    @action(List(Seq(Rule('test'), Loc(':'), Rule('test')), ',', trailing=True))
    def dictmaker(self, elts):
        """dictmaker: test ':' test (',' test ':' test)* [',']"""
        return ast.Dict(keys=list(map(lambda x: x[0], elts)),
                        values=list(map(lambda x: x[2], elts)),
                        colon_locs=list(map(lambda x: x[1], elts)),
                        loc=None)

    @action(Seq(Loc('class'), Tok('ident'),
                Opt(Seq(Loc('('), List(Rule('test'), ',', trailing=True), Loc(')'))),
                Loc(':'), Rule('suite')))
    def classdef(self, class_loc, name_tok, bases_opt, colon_loc, body):
        """classdef: 'class' NAME ['(' [testlist] ')'] ':' suite"""
        bases, lparen_loc, rparen_loc = [], None, None
        if bases_opt:
            lparen_loc, bases, rparen_loc = bases_opt

        return ast.ClassDef(name=name_tok.value, bases=bases, body=body,
                            decorator_list=[], at_locs=[],
                            keyword_loc=class_loc, lparen_loc=lparen_loc, rparen_loc=rparen_loc,
                            name_loc=name_tok.loc, colon_loc=colon_loc,
                            loc=class_loc.join(body[-1].loc))

    @action(Rule('argument'))
    def arglist_1(self, arg):
        return [arg], self._empty_arglist()

    @action(Seq(Loc('*'), Rule('test'), Star(SeqN(1, Tok(','), Rule('argument'))),
                Opt(Seq(Tok(','), Loc('**'), Rule('test')))))
    def arglist_2(self, star_loc, stararg, postargs, kwarg_opt):
        dstar_loc = kwarg = None
        if kwarg_opt:
            _, dstar_loc, kwarg = kwarg_opt

        for postarg in postargs:
            if not isinstance(postarg, ast.keyword):
                error = diagnostic.Diagnostic(
                    "error", "only named arguments may follow *expression", {}, postarg.loc)
                raise diagnostic.DiagnosticException(error)

        return postargs, \
               ast.Call(args=[], keywords=[], starargs=stararg, kwargs=kwarg,
                        star_loc=star_loc, dstar_loc=dstar_loc, loc=None)

    @action(Seq(Loc('**'), Rule('test')))
    def arglist_3(self, dstar_loc, kwarg):
        return [], \
               ast.Call(args=[], keywords=[], starargs=None, kwargs=kwarg,
                        star_loc=None, dstar_loc=dstar_loc, loc=None)

    @action(SeqN(0, Rule('argument'), Tok(',')))
    def arglist_4(self, arg):
        return [], ([arg], self._empty_arglist())

    @action(Alt(Seq(Star(SeqN(0, Rule('argument'), Tok(','))),
                    Alt(arglist_1, arglist_2, arglist_3)),
                arglist_4))
    def arglist(self, pre_args, rest):
        # Python's grammar is very awkwardly formulated here in a way
        # that is not easily amenable to our combinator approach.
        # Thus it is changed to the equivalent:
        #
        #     arglist: (argument ',')* ( argument | ... ) | argument ','
        #
        """arglist: (argument ',')* (argument [','] |
                                     '*' test (',' argument)* [',' '**' test] |
                                     '**' test)"""
        post_args, call = rest

        for arg in pre_args + post_args:
            if isinstance(arg, ast.keyword):
                call.keywords.append(arg)
            elif len(call.keywords) > 0:
                error = diagnostic.Diagnostic(
                    "error", "non-keyword arg after keyword arg", {}, arg.loc)
                raise diagnostic.DiagnosticException(error)
            else:
                call.args.append(arg)
        return call

    @action(Seq(Rule('test'), Loc('='), Rule('test')))
    def argument_1(self, lhs, equals_loc, rhs):
        if not isinstance(lhs, ast.Name):
            error = diagnostic.Diagnostic(
                "error", "keyword must be an identifier", {}, lhs.loc)
            raise diagnostic.DiagnosticException(error)
        return ast.keyword(arg=lhs.id, value=rhs,
                           loc=lhs.loc.join(rhs.loc),
                           arg_loc=lhs.loc, equals_loc=equals_loc)

    @action(Seq(Rule('test'), Opt(Rule('gen_for'))))
    def argument_2(self, lhs, compose_opt):
        if compose_opt:
            generators = compose_opt([])
            return ast.GeneratorExp(elt=lhs, generators=generators,
                                    begin_loc=None, end_loc=None,
                                    loc=lhs.loc.join(generators[-1].loc))
        return lhs

    argument = Alt(argument_1, argument_2)
    """argument: test [gen_for] | test '=' test  # Really [keyword '='] test"""

    list_iter = Alt(Rule("list_for"), Rule("list_if"))
    """list_iter: list_for | list_if"""

    def list_gen_for_action(self, for_loc, target, in_loc, iter, next_opt):
        def compose(comprehensions):
            comp = ast.comprehension(
                target=target, iter=iter, ifs=[],
                loc=for_loc.join(iter.loc), for_loc=for_loc, in_loc=in_loc, if_locs=[])
            comprehensions += [comp]
            if next_opt:
                return next_opt(comprehensions)
            else:
                return comprehensions
        return compose

    def list_gen_if_action(self, if_loc, cond, next_opt):
        def compose(comprehensions):
            comprehensions[-1].ifs.append(cond)
            comprehensions[-1].if_locs.append(if_loc)
            comprehensions[-1].loc = comprehensions[-1].loc.join(cond.loc)
            if next_opt:
                return next_opt(comprehensions)
            else:
                return comprehensions
        return compose

    list_for = action(
        Seq(Loc('for'), Rule('exprlist'),
            Loc('in'), Rule('testlist_safe'), Opt(Rule('list_iter')))) \
        (list_gen_for_action)
    """list_for: 'for' exprlist 'in' testlist_safe [list_iter]"""

    list_if = action(
        Seq(Loc('if'), Rule('old_test'), Opt(Rule('list_iter')))) \
        (list_gen_if_action)
    """list_if: 'if' old_test [list_iter]"""

    gen_iter = Alt(Rule("gen_for"), Rule("gen_if"))
    """gen_iter: gen_for | gen_if"""

    gen_for = action(
        Seq(Loc('for'), Rule('exprlist'),
            Loc('in'), Rule('or_test'), Opt(Rule('gen_iter')))) \
        (list_gen_for_action)
    """gen_for: 'for' exprlist 'in' or_test [gen_iter]"""

    gen_if = action(
        Seq(Loc('if'), Rule('old_test'), Opt(Rule('gen_iter')))) \
        (list_gen_if_action)
    """gen_if: 'if' old_test [gen_iter]"""

    testlist1 = action(List(Rule('test'), ',', trailing=False))(_wrap_tuple)
    """testlist1: test (',' test)*"""

    @action(Seq(Loc('yield'), Rule('testlist')))
    def yield_expr(self, stmt_loc, exprs):
        """yield_expr: 'yield' [testlist]"""
        return ast.Yield(value=exprs,
                         yield_loc=stmt_loc, loc=stmt_loc.join(exprs.loc))

def for_code(code, version=(2,7)):
    return Parser(lexer.Lexer(source.Buffer(code), version))

def main():
    import sys, time, codecs
    for filename in sys.argv[1:]:
        with codecs.open(filename, encoding='utf-8') as f:
            input = f.read()
            try:
                start = time.time()
                root = for_code(input).file_input()
                interval = time.time() - start

                print(root)
                print("elapsed: %.2f (%.2f kb/s)" % (interval, len(input)/interval/1000),
                      file=sys.stderr)
            except diagnostic.DiagnosticException as e:
                print(e.render,
                      file=sys.stderr)

if __name__ == "__main__":
    main()
