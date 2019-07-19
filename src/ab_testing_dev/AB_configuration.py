'''
Configuration file for A/B testing
'''

import numpy as np
import pandas as pd
import pickle
import gzip
import math
import random
from IPython.display import Markdown, display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, \
GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, \
classification_report, make_scorer
import statsmodels.api as sm

# From this project
from utils import rmse, rmse_train_cv, classifier_report, confusion_rmse
from NLP import WordBag, AboutMovie





