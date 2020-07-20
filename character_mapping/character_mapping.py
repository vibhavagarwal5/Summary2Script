import string

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import gensim.models.keyedvectors.KeyedVectors as word2vec

w2v = word2vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',
                                    binary=True)

all_stopwords = stopwords.words('english')
all_stopwords.append('19')
all_stopwords.append('``')
all_stopwords.append('—')
all_stopwords.append('’')
all_stopwords.append('20s')

for i in string.punctuation:
    all_stopwords.append(i)


def isWordPresent(sentence, word):
    # To break the sentence in words
    s = sentence.split(" ")
    for i in s:
        # Comparing the current word
        # with the word to be searched
        if (i == word):
            return True
    return False


chars = ['Blake', 'Smith', 'Mackenzie', 'Lance', 'Ecoust',
         'Schofield', 'Hepburn', 'Corporal', 'Leslie', 'Line']

file1 = open('../example_1917/1917.txt', 'r')
text = file1.read()
a_list = nltk.tokenize.sent_tokenize(text)
sentence = dict.fromkeys(chars, [])

for i in chars:
    l = str()
    for j in a_list:
        if(isWordPresent(j, i)):
            l = l + j + " "
    sentence[i] = l


# Word2Vec code
chars_emb = dict.fromkeys(chars, None)
for key, value in sentence.items():
    z = np.empty(300)
    text_tokens = word_tokenize(value)
    tokens_without_sw = [
        word for word in text_tokens if not word in all_stopwords]
    words = list()
    for i in tokens_without_sw:
        try:
            words.append(i)
            z = z + w2v[i]
        except:
            print(i, 'not found')
            words.remove(i)
    print(value)
    print(words)
    for i in z:
        i = i / len(words)
    chars_emb[key] = z

print(chars_emb)
