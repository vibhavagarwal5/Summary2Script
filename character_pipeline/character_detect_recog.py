from collections import Counter

import nltk
import numpy as np
import pandas as pd
import spacy

import neuralcoref


class CharacterDetector():
    def __init__(self, data_path, test=False):
        self.nlp = spacy.load("en_core_web_md")
        data = self.load_data(data_path)
        self.data = self.preprocess(data)
        if test:
            self.test_verbose(self.data)
        else:
            self.data['characters'] = self.data['summary_doc'].apply(
                lambda x: self.extract_characters(x))

    def load_data(self, path):
        df = pd.DataFrame(np.load(path))
        df.columns = ['movie', 'summary']
        return df

    def preprocess(self, df):
        def prep(x):
            def rem_up(i):
                if i.isupper():
                    return i.title()
                else:
                    return i
            x = x.replace('\n', ' ')
            x = list(map(rem_up, x.split(' ')))
            x = ' '.join(x)
            return x

        df['summary_preprocess'] = df['summary'].apply(lambda x: prep(x))
        df['summary_doc'] = list(
            self.nlp.pipe(df['summary_preprocess'], n_process=20, batch_size=64))
        return df

    def spacy_extraction(self, doc):
        return [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

    def nltk_extraction(self, text):
        text = ' '.join([i for i in text.split()])
        tokens = nltk.tokenize.word_tokenize(text)
        pos = nltk.pos_tag(tokens)
        sentt = nltk.ne_chunk(pos, binary=False)

        person = []
        name = ""
        for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
            person.append(
                " ".join([leaf[0] for leaf in subtree.leaves() if leaf[1] == 'NNP']))
        return person

    def filter_names(self, t_names):
        t_names = [" ".join([i.orth_ for i in self.nlp(n) if i.pos_ in [
                            'PROPN', 'NOUN']]) for n in t_names]
        t_names = list(filter(None, t_names))
        t_names = sorted(t_names, key=lambda x: (len(x.split(" ")), len(x)))
        for i, v in enumerate(t_names):
            for s in t_names[i + 1:]:
                if v in s and v != s:
                    idx = t_names.index(s)
                    t_names.remove(s)
                    for k_idx, k in enumerate(s.split(" ")):
                        t_names.insert(idx + k_idx, k)
        t_names = [i for i in t_names if len(
            i) > 2 and not any(c.isdigit() for c in i)]
        return Counter(t_names)

    def extract_characters(self, text, verbose=False):
        l2 = self.filter_names(self.spacy_extraction(text))
        l3 = self.filter_names(self.nltk_extraction(str(text)))
        f_l = list(set(l2) & set(l3))
        if verbose:
            print(l2)
            print(l3)
            print(f_l)
        f_l_f = [i for i in f_l if max(l2[i], l3[i]) > 2] + [i for i in list(set(l3) - set(
            f_l)) if l3[i] > 3] + [i for i in list(set(l2) - set(f_l)) if l2[i] > 3]
        return f_l_f

    def save_data(self, path):
        self.data.to_csv(path, index=None)

    def test_verbose(self, df):
        a_ = df['summary_doc'][0]
        print(df['summary_doc'][0])
        print('---------------------------------')

        l2 = self.filter_names(self.spacy_extraction(a_))
        l3 = self.filter_names(self.nltk_extraction(str(a_)))

        print(l2, l3)
        print('---------------------------------')

        f_l = list(set(l2) & set(l3))
        for i in f_l:
            print(f"{i}: (l2={l2[i]}), (l3={l3[i]})")
        print('---------------------------------')
        for i in list(set(l2) - set(f_l)):
            print(i, l2[i])
        print('---------------------------------')
        for i in list(set(l3) - set(f_l)):
            print(i, l3[i])

        f_l = list(set(l2) & set(l3))
        f_l_f = [i for i in f_l if max(l2[i], l3[i]) > 2] + [i for i in list(set(l3) - set(
            f_l)) if l3[i] > 3] + [i for i in list(set(l2) - set(f_l)) if l2[i] > 3]
        for i in f_l_f:
            print(f"{i}: (l2={l2[i]}), (l3={l3[i]})")


class CharacterRecognizer(CharacterDetector):
    def __init__(self, data_path):
        super.__init__(data_path)
        # self.nlp = spacy.load("en_core_web_md")
        self.nlp.add_pipe(neuralcoref.NeuralCoref(self.nlp.vocab),
                          name='neuralcoref')

        self.data = self.load_data(data_path)
        self.data = self.coref_resolve(self.data)

    def load_data(self, path):
        df = pd.read_csv(path)
        # df = pd.read_csv('./movie_w_characters.csv')
        df['characters'] = df['characters'].apply(eval)
        return df

    def coref_resolve(self, df):
        df['summary_doc'] = df['summary_preprocess'].apply(
            lambda x: self.nlp(x))
        df['summary_resolved'] = df['summary_doc'].apply(
            lambda x: x._.coref_resolved)

        self.nlp.remove_pipe('neuralcoref')
        self.nlp.has_pipe('neuralcoref')

        df['summary_r_doc'] = list(
            self.nlp.pipe(df['summary_resolved'], n_process=20, batch_size=64))
        df['characters_nc'] = df['summary_r_doc'].apply(
            lambda x: self.extract_characters(x))
        return df


# -----  Neuralcoref  -----------


if __name__ == "__main__":
    CharacterDetector = CharacterDetector('./data/outfile.npy', test=True)
    CharacterDetector.save_data('./data/movie_w_characters.csv')

    CharacterRecognizer = CharacterRecognizer('./data/movie_w_characters.csv')
    CharacterRecognizer.save_data('./data/movie_w_characters_nc.csv')
