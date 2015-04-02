from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source, diagnostic
import unittest

class DiagnosticTestCase(unittest.TestCase):

    def setUp(self):
        self.buffer = source.Buffer("x + (1 + 'a')\n")

    def test_message(self):
        diag = diagnostic.Diagnostic(
            "error", "{x} doesn't work", {"x": "everything"},
            source.Range(self.buffer, 0, 0))
        self.assertEqual("everything doesn't work", diag.message())

    def test_render(self):
        diag = diagnostic.Diagnostic(
            "error", "cannot add {lft} and {rgt}",
            {"lft": "integer", "rgt": "string"},
            source.Range(self.buffer, 7, 8),
            [source.Range(self.buffer, 5, 6),
             source.Range(self.buffer, 9, 12)])
        self.assertEqual(
            ["<input>:1:8: error: cannot add integer and string",
             "x + (1 + 'a')",
             "     ~ ^ ~~~ "],
            diag.render())
