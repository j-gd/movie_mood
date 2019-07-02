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
        self.emotions = emotions_df.transpose().to_dict(orient='list')


    def vectorize(self, data, column=None, normalize=True):
        '''
        Create and fill new emotion columns for the text in dataframe[column]
        INPUT:
            pandas dataframe or series
            column of the dataframe used as input for the emotion detection: list of words
        OUTPUT:
            2D numpy array with dimensions (input_data, 7 emotion columns)
    
        '''
        if isinstance(data, pd.DataFrame):
            list_of_lists = data[column].to_numpy()
        elif isinstance(data, pd.Series):
            list_of_lists = data.to_numpy()
        else:
            print('Unsupported data type')
            return None

        return np.array([self.get_emotions(w_list,normalize=normalize) for w_list in list_of_lists])


    def get_emotions(self, word_list,normalize=True):
        '''
        INPUT:
            word_list: list of lowercase words

        OUTPUT
            numpy array of 7 emotion values: disgust, surprise, neutral, anger, sad, happy, fear
        '''
        word_emotions = np.zeros((len(self.emotions_df.columns)))

        for word in word_list:
            if word in self.emotions.keys():
                word_emotions += self.emotions[word]

        length = np.sqrt(np.dot(word_emotions, word_emotions))

        if normalize:
            # normalize the vector: best reviews have very small text compared to others
            if length > 0:
                word_emotions = word_emotions / length
            else:
                # print('Emotions empty for comment: ', word_list)
                pass
        return word_emotions

if __name__ == "__main__":
    e = Emotions7()
    words = ['abuse','aaa']
    print('Emotions for:', words)
    print(e.get_emotions(words, normalize=False))
    print(e.get_emotions(words))
