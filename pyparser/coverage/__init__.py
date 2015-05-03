from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source, lexer
import os

_buf = None
with open(os.path.join(os.path.dirname(__file__), '..', 'parser.py')) as f:
    _buf = source.Buffer(f.read(), f.name)

# Inject the grammar with locations of rules, because Python's
# builtin tracebacks don't include column numbers.
# This would really be more elegant if it used the parser,
# but the parser doesn't work yet at the time of writing.
def instrument():
    rewriter = source.Rewriter(_buf)
    lex = lexer.Lexer(_buf, (3, 4))
    in_grammar = False
    stack = []
    for token in lex:
        if token.kind == 'from':
            token = lex.next()
            if token.kind == '.':
                rewriter.replace(token.loc, "..")

        if token.kind == 'class':
            token = lex.next()
            if token.kind == 'ident' and token.value == 'Parser':
                in_grammar = True

        if not in_grammar:
            continue

        if token.kind == 'ident' and \
                token.value in ('action', 'Eps', 'Tok', 'Loc', 'Rule', 'Expect',
                                'Seq', 'SeqN', 'Alt', 'Opt', 'Star', 'Plus', 'List',
                                'Newline', 'Oper', 'BinOper', 'BeginEnd'):
            lparen = lex.next()
            if lparen.kind == '(':
                rparen = lex.peek()
                if rparen.kind == ')':
                    lex.next()
                    rewriter.insert_before(rparen.loc,
                        "loc=(%d,%d)" % (token.loc.begin_pos, token.loc.end_pos))
                else:
                    stack.append(", loc=(%d,%d)" % (token.loc.begin_pos, token.loc.end_pos))

        if token.kind == '(':
            stack.append(None)

        if token.kind == ')':
            data = stack.pop()
            if data is not None:
                rewriter.insert_before(token.loc, data)

    with open(os.path.join(os.path.dirname(__file__), 'parser.py'), 'w') as f:
        f.write(rewriter.rewrite().source)

# Produce an HTML report for test coverage of parser rules.
def report(parser, name='parser'):
    rewriter = source.Rewriter(_buf)
    total_pts = 0
    total_covered = 0
    for rule in parser._all_rules:
        pts = len(rule.covered)
        covered = len(list(filter(lambda x: x, rule.covered)))
        if covered == 0:
            klass, hint = 'uncovered', None
        elif covered < pts:
            klass, hint = 'partial', ', '.join(map(lambda x: "yes" if x else "no", rule.covered))
        else:
            klass, hint = 'covered', None

        loc = source.Range(_buf, *rule.loc)
        if hint:
            rewriter.insert_before(loc, r"<span class='%s' title='%s'>" % (klass, hint))
        else:
            rewriter.insert_before(loc, r"<span class='%s'>" % klass)
        rewriter.insert_after(loc, r"</span>")

        total_pts += pts
        total_covered += covered

    print("GRAMMAR COVERAGE: %.2f%%" % (total_covered / total_pts * 100))

    content = rewriter.rewrite().source
    content = '\n'.join(map(
        lambda x: r"<span id='{0}' class='line'>{1}</span>".format(*x),
        enumerate(content.split("\n"), 1)))

    with open(os.path.join(os.path.dirname(__file__), '..', '..',
              'doc', 'coverage', name + '.html'), 'w') as f:
        f.write(r"""
<!DOCTYPE html>
<html>
    <head>
        <title>{percentage:.2f}%: {file} coverage report</title>
        <style type="text/css">
        .uncovered {{ background-color: #FFCAAD; }}
        .partial {{
            background-color: #FFFFB4;
            border-bottom: 1px dotted black;
        }}
        .covered {{ background-color: #9CE4B7; }}
        pre {{ counter-reset: line; }}
        .line::before {{
            display: inline-block;
            width: 4ex;
            padding-right: 1em;
            text-align: right;
            color: gray;
            content: counter(line);
            counter-increment: line;
        }}
        </style>
    </head>
<body>
    <h1>{percentage:.2f}% ({covered}/{pts}): {file} coverage report</h1>
    <pre>{content}</pre>
</body>
</html>
""".format(percentage=total_covered / total_pts * 100,
           pts=total_pts, covered=total_covered,
           file=os.path.basename(_buf.name),
           content=content))

# Create the instrumented parser when `import pyparser.coverage.parser`
# is invoked. Not intended for any use except running the internal testsuite.
instrument()
