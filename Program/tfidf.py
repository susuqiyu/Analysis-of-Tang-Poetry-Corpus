import os
import json
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


__author__ = "Singularity Point"
__version__ = "1.5.0"


TANG_PATH = "/Users/apple/Documents/SingularityPnt/Analysis-of-Tang-Poetry-Corpus/Corpus"


def refine_corpus(path):
    if os.path.isdir(path):
        files = [os.path.join(path, file) for file in os.listdir(path)]
    else:
        files = [path, ]

    titles = []
    corpus = []

    for file in files:
        with open(file) as f:
            poem_set = json.load(f)

        for poem in poem_set:
            titles.append(poem["title"])
            paragraphs = "".join(poem["paragraphs"])
            token_poem = " ".join(jieba.lcut(paragraphs))
            corpus.append(token_poem)

        del poem_set

    return titles, corpus


def calc_tfidf(corpus):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tf_matrix = vectorizer.fit_transform(corpus)
    tfidf = transformer.fit_transform(tf_matrix)
    return vectorizer.get_feature_names(), tfidf.toarray()


def obtain_tfidf(path=TANG_PATH, save_file=None):
    print("Start loading the corpus...")
    titles, corpus = refine_corpus(TANG_PATH)
    print("Finish loading")
    print()
    print("Start calculating the TF-IDF indices...")
    words, tfidf = calc_tfidf(corpus)
    print("Finish calculating")
    print()

    del corpus

    if save_file:
        print("Start saving the results...")
        temp = {"titles":titles, "words":words, "tfidf":tfidf.tolist()}
        with open(save_file, "w") as f:
            json.dump(temp, f)
        print("Finish saving")
    else:
        return titles, words, tfidf
