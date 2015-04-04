"""
The :mod:`parser` module concerns itself with LL(1) parsing.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from . import source, diagnostic, lexer

# Generic LL parsing combinators
def T(kind):
    """A rule that accepts a token of kind ``kind`` and returns it, or returns None."""
    def rule(parser):
        return parser._accept(kind)
    rule.expected = lambda parser: [kind]
    return rule

def L(kind):
    """A rule that accepts a token of kind ``kind`` and returns its location, or returns None."""
    def rule(parser):
        result = parser._accept(kind)
        if result is not None:
            return result.loc
    rule.expected = lambda parser: [kind]
    return rule

def R(name):
    """A proxy for a rule called ``name`` which may not be yet defined."""
    def rule(parser):
        return getattr(parser, name)()
    rule.expected = lambda parser: getattr(parser, name).expected(parser)
    return rule

def Expect(inner_rule):
    """A rule that executes ``inner_rule`` and emits a diagnostic error if it returns None."""
    def rule(parser):
        result = inner_rule(parser)
        if result is None:
            expected = inner_rule.expected(parser)
            if len(expected) > 1:
                expected = ' or '.join([', '.join(expected[0:-1]), expected[-1]])
            else:
                expected = expected[0]
            error = diagnostic.Diagnostic(
                "error", "unexpected {actual}: expected {expected}",
                {'actual': parser.token.kind, 'expected': expected},
                parser.token.loc)
            raise diagnostic.DiagnosticException(error)
        return result
    return rule

def Seq(first_rule, *rest_of_rules):
    """
    A rule that accepts a sequence of tokens satisfying ``rules`` and returns a tuple
    containing their return values, or returns None if the first rule is not satisfied.
    """
    rest_of_rules = map(Expect, rest_of_rules)
    def rule(parser):
        first_result = first_rule(parser)
        if first_result:
            return tuple([first_result] + map(lambda rule: rule(parser), rest_of_rules))
    rule.expected = first_rule.expected
    return rule

def Alt(*inner_rules):
    """
    A rule that expects a sequence of tokens satisfying one of ``rules`` in sequence
    (a rule is satisfied when it returns anything but None) and returns the return
    value of that rule, or None if no rules were satisfied.
    """
    def rule(parser):
        # semantically reduce(), but faster.
        for inner_rule in inner_rules:
            result = inner_rule(parser)
            if result is not None:
                return result
    rule.expected = \
        lambda parser: reduce(list.__add__, map(lambda x: x.expected(parser), inner_rules))
    return rule

def rule(inner_rule):
    """
    A decorator returning a function that first runs ``inner_rule`` and then, if its
    return value is not None, maps that value using ``mapper``.

    If the value being mapped is a tuple, it is expanded into multiple arguments.

    Similar to attaching semantic actions to rules in traditional parser generators.
    """
    def decorator(mapper):
        def outer_rule(parser):
            result = inner_rule(parser)
            if isinstance(result, tuple):
                result = mapper(parser, *result)
            elif result is not None:
                result = mapper(parser, result)
            return result
        outer_rule.expected = inner_rule.expected
        return outer_rule
    return decorator

class Parser:

    # Generic LL parsing methods
    def __init__(self, lexer):
        self.lexer = lexer
        self._advance()

    def _advance(self):
        self.token = self.lexer.next(eof_token=True)
        return self.token

    def _accept(self, expected_kind):
        if self.token.kind == expected_kind:
            result = self.token
            self._advance()
            return result

    # Python-specific methods
    @rule(Alt(T('int'), T('float')))
    def value(parser, token):
        return token.value

    @rule(Seq(L('('), R('expr'), L(')')))
    def paren(parser, lft, value, rgt):
        return value

    expr = Alt(R('value'), R('paren'))
