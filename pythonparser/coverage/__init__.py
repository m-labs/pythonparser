from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source, lexer, diagnostic
import os, codecs

_buf = None
with codecs.open(os.path.join(os.path.dirname(__file__), "..", "parser.py"),
                 encoding="utf-8") as f:
    _buf = source.Buffer(f.read(), f.name)

# Inject the grammar with locations of rules, because Python's
# builtin tracebacks don't include column numbers.
# This would really be more elegant if it used the parser,
# but the parser doesn't work yet at the time of writing.
def instrument():
    rewriter = source.Rewriter(_buf)
    lex = lexer.Lexer(_buf, (3, 4), diagnostic.Engine())
    in_grammar = False
    paren_stack = []
    stmt_stack = [None]
    all_stmt_list = []
    in_square = 0
    stmt_init = None
    for token in lex:
        if token.kind == "from":
            token = lex.next()
            if token.kind == ".":
                rewriter.replace(token.loc, "..")

        if token.kind == "class":
            token = lex.next()
            if token.kind == "ident" and token.value == "Parser":
                in_grammar = True

        if token.kind == "ident" and token.value == "_all_stmts":
            lex.next() # =
            stmt_init = lex.next().loc # {

        if not in_grammar:
            continue

        if token.kind == ".": # skip ast.List
            lex.next()

        if token.kind == "ident" and \
                token.value in ("action", "Eps", "Tok", "Loc", "Rule", "Expect",
                                "Seq", "SeqN", "Alt", "Opt", "Star", "Plus", "List",
                                "Newline", "Oper", "BinOper", "BeginEnd"):
            lparen = lex.next()
            if lparen.kind == "(":
                rparen = lex.peek()
                if rparen.kind == ")":
                    lex.next()
                    rewriter.insert_before(rparen.loc,
                        "loc=(%d,%d)" % (token.loc.begin_pos, token.loc.end_pos))
                else:
                    paren_stack.append(", loc=(%d,%d)" % (token.loc.begin_pos, token.loc.end_pos))

        if token.kind == "(":
            paren_stack.append(None)

        if token.kind == ")":
            data = paren_stack.pop()
            if data is not None:
                rewriter.insert_before(token.loc, data)

        if token.kind == "[":
            in_square += 1
        elif token.kind == "]":
            in_square -= 1

        if token.kind in ("def", "if", "elif", "else", "for") and in_square == 0:
            all_stmt_list.append((token.loc.begin_pos, token.loc.end_pos))
            stmt_stack.append("_all_stmts[(%d,%d)] = True\n" %
                                (token.loc.begin_pos, token.loc.end_pos))
        elif token.kind == "lambda":
            stmt_stack.append(None)

        if token.kind == "indent":
            data = stmt_stack.pop()
            if data is not None:
                rewriter.insert_after(token.loc, data + " " * token.loc.column())

    rewriter.insert_after(stmt_init, ", ".join(["(%d,%d): False" % x for x in all_stmt_list]))

    with codecs.open(os.path.join(os.path.dirname(__file__), "parser.py"), "w",
                     encoding="utf-8") as f:
        f.write(rewriter.rewrite().source)

# Produce an HTML report for test coverage of parser rules.
def report(parser, name="parser"):
    all_locs = dict()
    for rule in parser._all_rules:
        pts = len(rule.covered)
        covered = len(list(filter(lambda x: x, rule.covered)))
        if rule.loc in all_locs:
            pts_, covered_, covered_bool_ = all_locs[rule.loc]
            if covered > covered_:
                all_locs[rule.loc] = pts, covered, rule.covered
        else:
            all_locs[rule.loc] = pts, covered, rule.covered

    rewriter = source.Rewriter(_buf)
    total_pts = 0
    total_covered = 0
    for loc in all_locs:
        pts, covered, covered_bool = all_locs[loc]
        if covered == 0:
            klass, hint = "uncovered", None
        elif covered < pts:
            klass, hint = "partial", ", ".join(map(lambda x: "yes" if x else "no", covered_bool))
        else:
            klass, hint = "covered", None

        sloc = source.Range(_buf, *loc)
        if hint:
            rewriter.insert_before(sloc, r"<span class='%s' title='%s'>" % (klass, hint))
        else:
            rewriter.insert_before(sloc, r"<span class='%s'>" % klass)
        rewriter.insert_after(sloc, r"</span>")

        total_pts += pts
        total_covered += covered

    for stmt in parser._all_stmts:
        loc = source.Range(_buf, *stmt)
        if parser._all_stmts[stmt]:
            rewriter.insert_before(loc, r"<span class='covered'>")
        else:
            rewriter.insert_before(loc, r"<span class='uncovered'>")
        rewriter.insert_after(loc, r"</span>")

        total_pts += 1
        total_covered += (1 if parser._all_stmts[stmt] else 0)

    print("GRAMMAR COVERAGE: %.2f%%" % (total_covered / total_pts * 100))

    content = rewriter.rewrite().source
    content = "\n".join(map(
        lambda x: r"<span id='{0}' class='line'>{1}</span>".format(*x),
        enumerate(content.split("\n"), 1)))

    with codecs.open(os.path.join(os.path.dirname(__file__), "..", "..",
              "doc", "coverage", name + ".html"), "w", encoding="utf-8") as f:
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

# Create the instrumented parser when `import pythonparser.coverage.parser`
# is invoked. Not intended for any use except running the internal testsuite.
instrument()
