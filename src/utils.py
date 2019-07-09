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


'''
INPUT:
  Estimator: model to evaluate
  X: validation data
  y: ground truth target for X
OUTPUT:
  Returns a floating point number that quantifies the estimator prediction quality on X, 
  with reference to y. Again, by convention higher numbers are better, so if your scorer 
  returns loss, that value should be negated.
'''
