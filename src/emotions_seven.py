import pickle

class Emotions7():
    def __init__(self, test=False):
        if test:
            path = 'data/'
        else:
            path = '../data/'
        pickle_in = open(path + "keywords_seven_emotions.pickle", "rb")
        self.emotions = pickle.load(pickle_in)

if __name__ == "__main__":
    e = Emotions7()

