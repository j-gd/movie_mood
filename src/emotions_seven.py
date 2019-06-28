import numpy as np
import pandas as pd

class Emotions7():
    def __init__(self, test=False):
        if test == True:
            path = '../datasets/'
        else:
            path = '../../datasets/'
        emotions_df = pd.read_csv(path + 'andbrainDataSet108.csv')
        emotions_df['word'] = emotions_df['word'].apply(lambda w: w.lower().strip())
        emotions_df.set_index('word',inplace=True)
        self.emotions_df = emotions_df

    def transform(self, df, column):
        # Add 7 columns, one for each emotion
        pass

    def get_emotions(self, word_list,normalize=True):
        '''
        INPUT:
            word_list: list of lowercase words

        OUTPUT
            panda Series with word_list values for:
                disgust, surprise, neutral, anger, sad, happy, fear
        '''
        word_emotions = pd.Series(
            np.zeros((len(self.emotions_df.columns))), index=self.emotions_df.columns)

        for word in word_list:
            if word in self.emotions_df.index:
                word_emotions += self.emotions_df.loc[word,:]

        if normalize:
            # normalize the vector: best reviews have very small text compared to others
            word_emotions = word_emotions / np.sqrt(
                np.dot(word_emotions, word_emotions))

        return word_emotions

if __name__ == "__main__":
    e = Emotions7()
    words = ['abuse','aaa']
    print('Emotions for:', words)
    print(e.get_emotions(words, normalize=False))
    print(e.get_emotions(words))
