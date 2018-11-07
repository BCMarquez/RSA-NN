import unittest

from irony_rsa import *

class TestNormalization(unittest.TestCase):

    def test_literal_listener(self):
        for u in utterances:
            tot_prob = 0.0
            for s in states:
                for A in affects:
                    tot_prob += literal_listener(s, A, u)
            self.assertAlmostEqual(tot_prob, 1.0)

    def test_speaker(self):
        for s in states:
            for A in affects:
                for q in quds:
                    tot_prob = 0.0
                    for u in utterances:
                        tot_prob += speaker(u, s, A, q)
                    self.assertAlmostEqual(tot_prob, 1.0)

    def test_pragmatic_listener(self):
        for context in contexts:
            for u in utterances:
                tot_prob = 0.0
                for s in states:
                    for A in affects:
                        tot_prob += pragmatic_listener(s, A, u, context)
                self.assertAlmostEqual(tot_prob, 1.0)

if __name__ == '__main__':
    unittest.main()