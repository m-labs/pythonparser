import sys, re
from pythonparser import source, lexer, diagnostic

buf = None
with open(sys.argv[1]) as f:
    buf = source.Buffer(f.read(), f.name)

engine   = diagnostic.Engine()
rewriter = source.Rewriter(buf)
in_quot  = False
replace  = { "'": "\"", "'''": "\"\"\"" }
for token in lexer.Lexer(buf, (3, 4), engine):
    source = token.loc.source()
    if token.kind == "strbegin" and source in replace.keys():
        rewriter.replace(token.loc, replace[source])
        in_quot = True
    elif token.kind == "strdata" and in_quot:
        rewriter.replace(token.loc, re.sub(r'([^\\"]|)"', r'\1\\"', source))
    elif token.kind == "strend" and in_quot:
        rewriter.replace(token.loc, replace[source])
        in_quot = False

buf = rewriter.rewrite()

with open(sys.argv[1], "w") as f:
    f.write(buf.source)
