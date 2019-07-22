import numpy as np
import unicodedata
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re


class WordBag():
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
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

    def clean_text(self, text, remove_stop_words=False, remove_accents=False):
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
            text = ' '.join(word for word in text.split() if word not in
                            self.stop_words_tiny)  # remove stopwors from text
        return text

    def create_word_bag(self, reviews, remove_stop_words=False, lemmatize=True):
        return [
            self.comment_to_bag_of_words(review, remove_stop_words, lemmatize)
            for review in reviews
        ]

    def comment_to_sentences(self, comment):
        input_string = self.remove_accents(comment)
        sent_tokens = sent_tokenize(input_string)
        return [self.comment_to_bag_of_words(sent_token) for sent_token in sent_tokens]

    # IMPORTANT: Obj-Sub filter relies on the default configs of:
    # remove_stop_words=False, lemmatize=True
    def comment_to_bag_of_words(self, sentence, remove_stop_words=False, lemmatize=True):
        words = word_tokenize(sentence)
        words_lower = np.array([word.lower() for word in words])
        if remove_stop_words:
            words_lower = words_lower[[self.keep(word) for word in words_lower]]
        if lemmatize:
            return [self.lemmatizer.lemmatize(w) for w in words_lower]
        else:
            return words_lower

    @staticmethod
    def remove_accents(input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        only_ascii = nfkd_form.encode('ASCII', 'ignore')
        return only_ascii.decode()

    def keep(self, word):
        cleaned_word = re.sub('[^A-Za-z]+', '', word)
        return cleaned_word not in self.stop_words


class AboutMovie():
    '''
    Prerequisite: all words are in lowercase
    '''
    def __init__(self):
        self.not_about_movie = set(['amazon', 'amzn', 'dvd', 'vhs', 'hd', 'edition',
        'blue-ray', 'blueray', 'blu-ray', 'bluray', 'price'])

    def check(self, words):
        common = set(words) & self.not_about_movie
        for word in common:
            if not word in self.not_about_movie:
                print('problem with', common)
        return True if len(common) == 0 else False

'''
Note:
There seems to be a problem with input: revID AYLFYY8AHPFOW	 asin B0002Z0EXQ
classified as failing check when it shouldn't be	
s1 = set(['troy', 'amazing', 'adaption', 'homer', 'illiad', 'rich', 'detail', 'horner', 'brilliant', 'usual', 'and', 'brad', 'pitt', 'and', 'eric', 'bana', 'great', 'job', 'portraying', 'achiles', 'and', 'hector'])
len(s1 & about_movie.not_about_movie)
'''
