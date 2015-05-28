from __future__ import absolute_import, division, print_function, unicode_literals
from .. import parse, algorithm
import unittest

class AlgorithmTestCase(unittest.TestCase):

    def test_visitor(self):
        class FooVisitor(algorithm.Visitor):
            def __init__(self):
                self.num_count = self.other_count = 0

            def visit_Num(self, node):
                self.num_count += 1
                algorithm.Visitor.generic_visit(self, node)

            def generic_visit(self, node):
                self.other_count += 1
                algorithm.Visitor.generic_visit(self, node)

        visitor = FooVisitor()
        visitor.visit(parse("[1,2,x,y]\n"))

        self.assertEqual(2, visitor.num_count)
        self.assertEqual(5, visitor.other_count)

    def test_compare(self):
        self.assertFalse(algorithm.compare(parse("1 + 2\n"), parse("1 + 3\n")))
        self.assertTrue( algorithm.compare(parse("1 + 2\n"), parse("1 + 2\n")))
        self.assertTrue( algorithm.compare(parse("1 + 2\n"), parse("1 +  2\n")))
        self.assertFalse(algorithm.compare(parse("1 + 2\n"), parse("1 +  2\n"),
                                           compare_locs=True))

    def test_transform(self):
        class FooTransformer(algorithm.Transformer):
            def visit_Num(self, node):
                if node.n == 42:
                    return None
                return node

        transformer = FooTransformer()

        self.assertTrue(algorithm.compare(parse("return 10\n"),
                        transformer.visit(parse("return 10\n"))))
        self.assertTrue(algorithm.compare(parse("return\n"),
                        transformer.visit(parse("return 42\n"))))
        self.assertTrue(algorithm.compare(parse("[1,2,3]\n"),
                        transformer.visit(parse("[1,2,3]\n"))))
        self.assertTrue(algorithm.compare(parse("[1,3]\n"),
                        transformer.visit(parse("[1,42,3]\n"))))
