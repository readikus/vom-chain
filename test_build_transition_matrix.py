import unittest
import numpy
from build_transition_matrix import MarkovChain
from fractions import Fraction

class TestMarkovChain(unittest.TestCase):

    def test_tokenize_unigrams(self):

        mm = MarkovChain()
        unigrams = mm.tokenize('a b c', 1)
        expected_unigrams = [('a',), ('b',), ('c',)]
        self.assertEqual(unigrams, expected_unigrams)

    def test_tokenize_trigrams(self):

        mm = MarkovChain()
        unigrams = mm.tokenize('a b c', 3)
        expected_unigrams = [('<S>', '<S>', 'a'),
            ('<S>', 'a', 'b'),
            ('a', 'b', 'c'),
            ('b', 'c', '</S>'),
            ('c', '</S>', '</S>')]
        self.assertEqual(unigrams, expected_unigrams)

    def test_build_vocab(self):

        mm = MarkovChain()
        vocab = mm.build_vocab(['a b c'], 3)
        expected_vocab = [('b',),
            ('<S>', '<S>'),
            ('c', '</S>'),
            ('c',),
            ('<S>',),
            ('a', 'b'),
            ('</S>',),
            ('<S>', 'a'),
            ('b', 'c'),
            ('a',)]
        self.assertEqual(sorted(vocab), sorted(expected_vocab))

    def test_train(self):
        mm = MarkovChain()
        mm.train(['a b c', 'd e f'], 3)
        expected = [[Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 2), 0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 2), 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 1), 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 1), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 1), 0.0, 0.0, 0.0],
            [Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Fraction(1, 1), 0.0],
            [Fraction(1, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        expected = numpy.array([numpy.array(xi, dtype=object) for xi in expected])
        self.assertEqual(numpy.array2string(expected), numpy.array2string(mm.transitions))
