'''
Unit tests for Movie Mood utilities
Usage: from project directory run: pytest test/unittests.py
'''

import unittest as unittest
import numpy as np
from src.emotions_seven import Emotions7


class TestEmotions7(unittest.TestCase):
    def setUp(self):
        self.emotions = Emotions7(test=True)

    def tearDown(self):
        self.emotions = None

    def test_class_emotions(self):
        self.assertEqual(list(self.emotions.emotions_df.columns),
            ['disgust', 'surprise', 'neutral', 'anger', 'sad', 'happy', 'fear'])
        self.assertAlmostEqual(
            self.emotions.get_emotions(['abuse', 'aaa'], normalize=False)[6],
            0.00265863,places=8)
        self.assertAlmostEqual(
            self.emotions.get_emotions(['aaa', 'absorb', 'absorbed'],
                                       normalize=False)[3], 0.11020408, places=8)

if __name__ == '__main__':
    unittest.main()
