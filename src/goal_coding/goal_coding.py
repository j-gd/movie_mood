'''
Goal-oriented programming
* Provide answers to goals
* Help people learn Python
'''
import pandas as pd
import numpy as np
import warnings
from IPython.display import display, Markdown

from eda import eda

class GoalCoding():
    def __init__(self,
    file_or_object=None,
    its_name = None,
    nb_rows_to_read=-1,
    debug_head=3):
        '''
        PARADIGM:
        A class that lets you code by goals and sub-goals selection via tabbing

        First you need to define the data to work on, either at creation or using the configure action

        Then, hit tab on your instance to explore the actions you can perform

        By default, the instance performs actions on the last object manipulated and saves it to the same, but you can specify other ones as arguments


        OPTIONAL INPUT (sent to your_instance.configure)
          debug_head: 
            >=1 configure the instance to print a visual check of the result of actions that modify objects
            -1  configure the instance to not print a visual check
          file_or_object: the initial data to process
          its_name: the name of the initial data copied into this instance
          nb_rows_to_read (int): -1 reads everything. Only used if its_name is provided
        '''
        self.eda = eda(self)
        self.subset_rows = SubsetRows(self)
        self.configure = Configure(self)
        self._debug_head = debug_head
        self.my_objects = MyObjects(self)
        self.o = {}  # dictionary of user objects
        self.current = None

        if file_or_object != None:
            self.my_objects.add_object(file_or_object,
                                    its_name,
                                    nb_rows_to_read)


class Configure():
    def __init__(self, goal):
        self._ = goal

    def set_visual_check(self, debug_head=3):
        '''
      debug_head: number of lines of the result to display
        -1: don't display anything
      '''
        self._._debug_head = debug_head


class MyObjects():
    ''' 
  Where all user objects are stored
  '''
    def __init__(self, goal):
        self._ = goal
        self.config = [
          ('NB_SAMPLES', 360000),
          ]
        self.paths = []

    def add_object(self, file_or_object, its_name, nb_rows_to_read=-1):

        warnings.simplefilter('error', UserWarning)
        try:
            if file_or_object == None:
                print('Usage: GoalCoding(file_or_object)')
            elif isinstance(file_or_object, (pd.DataFrame,pd.Series)):
                if nb_rows_to_read < 1:
                    self._.o[its_name] = file_or_object.copy()
                else:
                    self._.o[its_name] = file_or_object[:nb_rows_to_read].copy()
            elif isinstance(file_or_object, str):
                if nb_rows_to_read < 1:
                    self._.o[its_name] = pd.read_csv(file_or_object)
                else:
                    self._.o[its_name] = pd.read_csv(file_or_object,
                                        nrows = nb_rows_to_read)
            else:
                print('Unsupported input type')
                return
        except:
            raise

        self.set_current(its_name)

    def set_current(self, object_name):
        if not object_name in self._.o.keys():
            print("Error, the object {} is not in this instance's objects".format(object_name))
            return

        self._.current = self._.o[object_name]

    def get_current(self, object_name):
        if not object_name in self._.o.keys():
            print("Error, the object {} is not in this instance's objects".
                  format(object_name))
            return
        return self._.current

    def print_info(self):
        print('Your objects:')
        for key, val in self._.o.items():
            print('Name: {} , Type: {}'.format(key, type(val)))


class SubsetRows():
    def __init__(self, goal):
        self._ = goal

    def meet_logical_criteria(self, condition_or_filter_for_samples_to_keep,
                              on_object='current', to_object='inplace'):
        '''
        on_object:
          'current' (default)
          name given to the object with add_object or at instance creation
        to_object:
          'inplace' (default) to replace the content of the on_object
          name of the new object to create
        '''
        if on_object == 'current':
            on_object = self._.current
        elif on_object not in self._.o.keys():
            print('Please add object {} first'.format(on_object))


        if to_object == 'inplace':
            to_object = on_object
            # print('replacing object')

        if isinstance(on_object, pd.DataFrame) or isinstance(on_object, pd.DataFrame) \
          or isinstance(on_object, np.array):
            print('df_or_series_or_np_array[condition_or_filter_for_samples_to_keep]\n')

            self._.o['train_df'] = self._.current[
                condition_or_filter_for_samples_to_keep]

            if self._._debug_head > 0:
                display(to_object[:self._._debug_head])
        else:
            print('Unsupported object type:', type(on_object))
            print('Supported types are Pandas DataFrame, Series or Numpy array')


    def drop_duplicates(self,
                              object_to_filter='current'):
        '''
        object_to_filter: 
            'current' (default): use on the last object processed by this GoalCode instance
            or specify a different object to process, e.g. object_to_filter=df2

        save: 
            'inplace' (default) saves in place
            TODO: name of new object, to save as a new object of same type
            TODO: 'no' don't save the result

        To turn debug-print on/off for the result of the operation, call manage_goal_coding.set_debug()
        '''
        if object_to_filter == 'current':
            object_to_filter = self._.current

        if isinstance(object_to_filter, pd.DataFrame) or isinstance(
                object_to_filter, pd.DataFrame):
            print('df_or_series.drop_duplicates()')
            object_to_filter.drop_duplicates(inplace=True)
        else:
            print('Unsupported object type:', type(object_to_filter))
            print(
                'Currently supported types are Pandas DataFrame or Series')
