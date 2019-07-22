import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import re


class Clean():
    def __init__(self):
        self.TO_REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;+\#_]')
        self.TO_REMOVE_RE = re.compile('[^a-z -]')

        # Don't remove negations or sense of level
        self.stop_words = set(['then', "you're", 'she', 'yourself', 'or', 'itself', 'does', 'until',
        'themselves', 'you', 'y', 'because', 'of', 'on', 'is', 'further', 'himself',
        'myself', 'so', 'only', 'll', 'my', 'same', 'am', 'her', 'had', 'a',
        'where', 'other', 'be', 'him', "you've", 'own', 'the', 'are', 'were',
        'this', 'at', 'our', 'from', 'm', 'before',
        'off', 've', 'by', "you'd", 'after', 'who', 'has', 'some', 'those', 'as',
        'about', 'no', 'me', 'its', "you'll", 'have', 'through', 'over',
        'too', 'such', 'yourselves', 'he', 'his', 'them', 'under', 'they',
        'above', 'herself', 'between', "it's", 're', 'against', 'all',
        'whom', 'during', 'here', 'now', 'do', 'to', 'can', 'up', 'ma',
        'it', 'theirs', 'we', 'what', 'yours', 'o', 'which', 'once', 'why', 'doing',
        'when', 'been', 'with', 'hers', 'just', 'will', "she's", 'for', 'in', 'their',
        's', 'these', 'an', 'down', 'ours', 'into', 'd', 'having', 'both', 'each',
        'your', 'did', 'how', 'there', 'that', 't', 'i', "that'll", 'any', 'being',
        'ourselves',''])

        self.stop_words_tiny = set(['i','a'])

    def clean_text(self, text, remove_stop_words=False,remove_accents=False):
        """
            text: a string
            
            return: modified initial string
        """
        if remove_accents:
            text = self.remove_accents(text)
        # text = text.lower()  # lowercase text
        text = self.TO_REPLACE_BY_SPACE_RE.sub(' ', text)
        text = self.TO_REMOVE_RE.sub('', text)
        # text = re.sub(r'\d+', '', text)
        if remove_stop_words:
            text = ' '.join(word for word in text.split()
                          if word not in self.stop_words_tiny)  # remove stopwors from text
        return text
