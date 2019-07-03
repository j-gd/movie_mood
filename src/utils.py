from nltk.tokenize import word_tokenize
import numpy as np
from IPython.display import Markdown, display
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def split_n_lower(text):
    words = word_tokenize(text)
    return [word.lower() for word in words]



support_keywords = {'dvd': 1, 'vhs': 1,'edition': 1, 'blue-ray': 1, 'blueray': 1,
                    'blu-ray': 1, 'bluray': 1, 'price': 1}

def not_about_support(word_list):
    # print(word_list)
    for word in word_list:
        if word in support_keywords.keys():
            return False

    return True

def rmse_train_cv(model, X_train, X_cv, y_train, y_cv):
    display(Markdown('### RMSE for {} on {}:'.format(type(model).__name__)))
    rmse(model.predict(X_train), y_train, 'Training:')
    rmse(model.predict(X_cv), y_cv, 'Test:    ')


def rmse(y_hat,y,label=None):
    if len(y_hat) != len(y):
        print('Error: mismatch in y and y_hat lengths')
        return None

    error = y_hat - y
    rmse = np.sqrt(np.dot(error, error) / len(y))

    if label != None:
        display(Markdown('#### {0} {1:3.3f}'.format(label,rmse)))
    return rmse


def classifier_report(model, test_X, true_y, label):
    pred = model.predict(test_X)
    display(Markdown('### Report for {}:'.format(label)))
    display(Markdown('#### Classification Report:'))
    print(classification_report(true_y, pred))
    display(Markdown('#### Confusion Matrix:'))
    print(confusion_matrix(true_y, pred))