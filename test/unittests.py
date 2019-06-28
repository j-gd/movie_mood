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
        self.assertEqual(
            self.emotions.get_emotions(
                ['abuse','aaa'], normalize=False).loc['anger'], 0.13736264)

if __name__ == '__main__':
    unittest.main()
