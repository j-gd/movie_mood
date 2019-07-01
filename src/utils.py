from nltk.tokenize import word_tokenize

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
