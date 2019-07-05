import numpy as np
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

class WordBag():
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

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

    def create(self, reviews):
        return [self.comment_to_bag_of_words(review) for review in reviews]

    def comment_to_bag_of_words(self, comment):
        input_string = self.remove_accents(comment)
        words = word_tokenize(input_string)
        words_lower = np.array([word.lower() for word in words])
        useful_words = words_lower[[self.keep(word) for word in words_lower]]
        roots = [self.lemmatizer.lemmatize(w) for w in useful_words]
        return roots

    @staticmethod
    def remove_accents(input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        only_ascii = nfkd_form.encode('ASCII', 'ignore')
        return only_ascii.decode()

    def keep(self, word):
        cleaned_word = re.sub('[^A-Za-z]+', '', word)
        return cleaned_word not in self.stop_words
