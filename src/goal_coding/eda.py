import numpy as np
import pandas as pd
import pandas_profiling as pp
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import warnings
from IPython.display import display, Markdown


class eda():
    def __init__(self, goal):
        self._ = goal

    def basic(self, nb_lines = 10):
        warnings.simplefilter('error', UserWarning)
        try:
            display(Markdown('### Info'))
            display(self._.current.info())
            display(Markdown('### First {} rows:'.format(nb_lines)))
            display(self._.current.iloc[:nb_lines])
        except:
            raise

    def in_depth(self,nb_lines = 10):
        warnings.simplefilter('error', UserWarning)
        try:
            self.basic(nb_lines)
            display(Markdown('### Profile report:'))
            display(pp.ProfileReport(self._.current))
        except:
            raise
