import unittest
import pyparser.source as source
import pyparser.diagnostic as diagnostic

class DiagnosticTestCase(unittest.TestCase):

    def setUp(self):
        self.buffer = source.Buffer(u'x + (1 + "a")\n')

    def test_message(self):
        diag = diagnostic.Diagnostic(
            'error', u"{x} doesn't work", {'x': 'everything'},
            source.Range(self.buffer, 0, 0))
        self.assertEqual(u"everything doesn't work", diag.message())

    def test_render(self):
        diag = diagnostic.Diagnostic(
            'error', u"cannot add {lft} and {rgt}",
            {'lft': u'integer', 'rgt': u'string'},
            source.Range(self.buffer, 7, 8),
            [source.Range(self.buffer, 5, 6),
             source.Range(self.buffer, 9, 12)])
        self.assertEqual(
            [u'<input>:1:8: error: cannot add integer and string',
             u'x + (1 + "a")',
             u'     ~ ^ ~~~ '],
            diag.render())
