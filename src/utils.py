from nltk.tokenize import word_tokenize
import numpy as np
from IPython.display import Markdown, display
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.tokenize import sent_tokenize


def split_n_lower(text):
    words = word_tokenize(text)
    return [word.lower() for word in words]

# Support detection
support_keywords = frozenset(['dvd', 'vhs','edition', 'blue-ray', 'blueray',
                    'blu-ray', 'bluray', 'price', 'amazon', 'amzn', 'amazn', 'ship','shipped'])

def _list_not_about_support(word_list):
    return len(support_keywords & set(word_list)) == 0

def _string_not_about_support(string):
    return _list_not_about_support(split_n_lower(string))

def not_about_support(ds):
    return ds.map(_string_not_about_support)


# Measures
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
    prediction_report(test_X, true_y, model.predict(test_X), label)

def prediction_report(test_x, true_y, pred_y, label):
    confusion_mat = confusion_matrix(true_y, pred_y)
    display(Markdown('### Report for {}:'.format(label)))
    # display(
    #     Markdown('##### Confusion RMSE: {0:.3f}'.format(
    #         confusion_matrix_rmse(confusion_mat))))
    display(
        Markdown('##### Off diagonal: {0:.0%}'.format(confusion_off_diagonal(confusion_mat))))
    display(Markdown('#### Confusion Matrix:'))
    print(confusion_mat)
    display(Markdown('#### Classification Report:'))
    print(classification_report(true_y, pred_y))

def confusion_matrix_rmse(confusion_matrix):
    '''
  INPUT
  confusion_matrix: square np array

  OUTPUT
  Returns the square root of the mean squared error, where the error 
  is the distance to the true label (e.g. a prediction of 3 for a correct answer 
  of 1 has a distance of (3-1) = 2)
  '''
    weights = np.zeros(confusion_matrix.shape)

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            weights[i][j]=(i-j)*(i-j)
    # print(np.sum(confusion_matrix))
    # print(np.sum(weights * confusion_matrix) / np.sum(confusion_matrix))
    return np.sqrt(np.sum(weights * confusion_matrix) / np.sum(confusion_matrix))

def confusion_rmse(y_true, y_pred):
    return confusion_matrix_rmse(confusion_matrix(y_true, y_pred))

def confusion_off_diagonal(confusion_matrix):
    total = np.sum(confusion_matrix)
    diagonal = 0
    for i in range(confusion_matrix.shape[0]):
        diagonal += confusion_matrix[i][i]
    return (total - diagonal) / total

def _nb_sentences(string):
    return len(sent_tokenize(string))

def nb_sentences(ds):
    return ds.map(_nb_sentences)
