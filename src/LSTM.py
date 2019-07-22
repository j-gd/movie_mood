import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from sklearn.svm import SVC
from nltk.stem.snowball import SnowballStemmer

## remove special symbols
def rm_sym(df):
    df['review'] = df['review'].str.replace("&#039;",'\'')
    df['review'].head()
    df['rating_cate'] = ''
    df.loc[df['rating'] >= 7,'rating_cate'] = 'high'
    df.loc[df['rating'] <= 4,'rating_cate'] = 'low'
    df.loc[(df['rating'] > 4) & (df['rating'] < 7),'rating_cate'] = 'medium'
    return df

def clean_text(df_tem3):
    df_tem3['review'] = df_tem3['review'].str.replace("\"","").str.lower()
    df_tem3['review'] = df_tem3['review'].str.replace( r"(\\r)|(\\n)|(\\t)|(\\f)|(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(&#039;)|(\d\s)|(\d)|(\/)","")
    df_tem3['review'] = df_tem3['review'].str.replace("\"","").str.lower()
    df_tem3['review'] = df_tem3['review'].str.replace( r"(\$)|(\-)|(\\)|(\s{2,})"," ")
    df_tem3['review'].sample(1).iloc[0]

    stemmer = SnowballStemmer('english')
    df_tem3['review'] = df_tem3['review'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split(" ")]))
    return df_tem3


np.random.seed(9)


class LSTM():
  def __init__(self):
    pass

  
  