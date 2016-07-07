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
            ["<input>:1:8-1:9: error: cannot add integer and string",
             "x + (1 + 'a')",
             "     ~ ^ ~~~ "],
            diag.render())

class DiagnosticEngineTestCase(unittest.TestCase):

    def setUp(self):
        self.buffer = source.Buffer("x + (1 + 'a')\n")
        self.last_diagnostic = None
        self.engine = diagnostic.Engine()
        def render_diagnostic(diag):
            self.last_diagnostic = diag
        self.engine.render_diagnostic = render_diagnostic

    def test_context(self):
        note1 = diagnostic.Diagnostic(
            "note", "broken here", {},
            source.Range(self.buffer, 0, 0))
        note2 = diagnostic.Diagnostic(
            "note", "also broken there", {},
            source.Range(self.buffer, 0, 0))

        with self.engine.context(note1):
            with self.engine.context(note2):
                diag = diagnostic.Diagnostic(
                    "error", "{x} doesn't work", {"x": "everything"},
                    source.Range(self.buffer, 0, 0))
                self.engine.process(diag)
                self.assertEqual(self.last_diagnostic.notes, [note1, note2])

            diag = diagnostic.Diagnostic(
                "error", "{x} doesn't work", {"x": "everything"},
                source.Range(self.buffer, 0, 0))
            self.engine.process(diag)
            self.assertEqual(self.last_diagnostic.notes, [note1])

        diag = diagnostic.Diagnostic(
            "error", "{x} doesn't work", {"x": "everything"},
            source.Range(self.buffer, 0, 0))
        self.engine.process(diag)
        self.assertEqual(self.last_diagnostic.notes, [])
