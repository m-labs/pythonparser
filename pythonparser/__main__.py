from __future__ import absolute_import, division, print_function, unicode_literals
from . import parse, diagnostic
import sys, time, codecs

for filename in sys.argv[1:]:
    with codecs.open(filename, encoding="utf-8") as f:
        input = f.read()
        try:
            start = time.time()
            root = parse(input, filename)
            interval = time.time() - start

            print(root)
            print("elapsed: %.2f (%.2f kb/s)" % (interval, len(input)/interval/1000),
                  file=sys.stderr)
        except diagnostic.Error as e:
            print("\n".join(e.diagnostic.render()),
                  file=sys.stderr)
