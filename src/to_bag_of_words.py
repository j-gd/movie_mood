import unicodedata
from nltk.tokenize import sent_tokenize



def create_bag_of_words(df, input_col_name, output_col_name):
    return df.assign(output_col_name=lambda row: comment_to_bag_of_words(row[
        input_col_name]))



def comment_to_bag_of_words(comment):
    input_string = remove_accents(comment)
    sent_tokens = sent_tokenize(input_string)
    return sent_tokens


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()
