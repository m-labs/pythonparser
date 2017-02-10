from __future__ import absolute_import, division, print_function, unicode_literals
from .. import source
import unittest

class BufferTestCase(unittest.TestCase):

    def setUp(self):
        self.buffer = source.Buffer("line one\nline two\n\nline four")

    def test_repr(self):
        self.assertEqual("Buffer(\"<input>\")", repr(self.buffer))

    def test_source_line(self):
        self.assertEqual("line one\n", self.buffer.source_line(1))
        self.assertEqual("line two\n", self.buffer.source_line(2))
        self.assertEqual("\n", self.buffer.source_line(3))
        self.assertEqual("line four", self.buffer.source_line(4))
        self.assertRaises(IndexError, lambda: self.buffer.source_line(0))

    def test_decompose_position(self):
        self.assertEqual((1,0), self.buffer.decompose_position(0))
        self.assertEqual((1,2), self.buffer.decompose_position(2))
        self.assertEqual((1,8), self.buffer.decompose_position(8))
        self.assertEqual((2,0), self.buffer.decompose_position(9))
        self.assertRaises(IndexError, lambda: self.buffer.decompose_position(90))

    def test_bug_one_past_end(self):
        self.assertEqual((2, 0),
                         source.Buffer("\n").decompose_position(1))

    def test_encoding(self):
        self.assertEqual("ascii", source.Buffer("\n").encoding)
        self.assertEqual("ascii", source.Buffer("coding: wut").encoding)
        self.assertEqual("ascii", source.Buffer("\n\n# coding: wut").encoding)
        self.assertEqual("utf-8", source.Buffer("# coding=utf-8").encoding)
        self.assertEqual("iso-8859-1", source.Buffer("\n# -*- coding: iso-8859-1 -*-").encoding)


class RangeTestCase(unittest.TestCase):

    def setUp(self):
        self.buffer = source.Buffer("line one\nline two\n\nline four")

    def range(self, lft, rgt):
        return source.Range(self.buffer, lft, rgt)

    def test_repr(self):
        self.assertEqual("Range(\"<input>\", 0, 2, None)",
                         repr(self.range(0, 2)))

    def test_begin(self):
        self.assertEqual(self.range(1, 1),
                         self.range(1, 2).begin())

    def test_end(self):
        self.assertEqual(self.range(2, 2),
                         self.range(1, 2).end())

    def test_size(self):
        self.assertEqual(1, self.range(2, 3).size())

    def test_column(self):
        self.assertEqual(2, self.range(2, 2).column())
        self.assertEqual(0, self.range(9, 11).column())
        self.assertEqual(2, self.range(11, 11).column())

    def test_column_range(self):
        self.assertEqual((2,2), self.range(2, 2).column_range())
        self.assertEqual((0,2), self.range(9, 11).column_range())
        self.assertEqual((2,2), self.range(11, 11).column_range())
        self.assertEqual((0,8), self.range(0, 11).column_range())

    def test_line(self):
        self.assertEqual(1, self.range(2, 2).line())
        self.assertEqual(2, self.range(9, 11).line())
        self.assertEqual(2, self.range(11, 11).line())

    def test_line(self):
        self.assertEqual(1, self.range(2, 2).line())
        self.assertEqual(2, self.range(9, 11).line())
        self.assertEqual(2, self.range(11, 11).line())

    def test_join(self):
        self.assertEqual(self.range(1, 6),
                         self.range(2, 6).join(self.range(1, 5)))
        self.assertRaises(ValueError,
                          lambda: self.range(0, 0).join(
                                source.Range(source.Buffer(""), 0, 0)))

    def test_source(self):
        self.assertEqual("one", self.range(5, 8).source())

    def test_source_line(self):
        self.assertEqual("line two\n", self.range(9, 9).source_line())

    def test_source_lines(self):
        self.assertEqual(["line two\n"], self.range(9, 9).source_lines())
        self.assertEqual(["line two\n", "\n"], self.range(9, 18).source_lines())

    def test___str__(self):
        self.assertEqual("<input>:2:1-2:2", str(self.range(9, 10)))
        self.assertEqual("<input>:2:1", str(self.range(9, 9)))

    def test___ne__(self):
        self.assertTrue(self.range(0,0) != self.range(0,1))
        self.assertFalse(self.range(0,0) != self.range(0,0))
