import os
import re
import json
import pynlpir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


__author__ = "Suyu Wang"
__version__ = "1.7.3"


TANG_PATH = "/Users/apple/Documents/SingularityPnt/Analysis-of-Tang-Poetry-Corpus/Corpus"


pynlpir.open()


def refine_corpus(path):

    if os.path.isdir(path):
        files = [os.path.join(path, file) for file in os.listdir(path)]
    else:
        files = [path, ]

    titles = []
    corpus = []

    ambiguity = re.compile("\{.*?\}")
    redundancy = re.compile("\(.*?\)|\d")

    for file in files:
        with open(file) as f:
            poem_set = json.load(f)

        for poem in poem_set:
            paragraphs = "".join(poem["paragraphs"])
            if ambiguity.search(paragraphs):
                continue
            else:
                paragraphs = redundancy.sub("", paragraphs)

            titles.append(poem["title"])
            token_poem = " ".join(pynlpir.segment(paragraphs, pos_tagging=False))

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
        temp = {"titles": titles, "words": words, "tfidf": tfidf.tolist()}
        with open(save_file, "w") as f:
            json.dump(temp, f)
        print("Finish saving")

    return titles, words, tfidf


def view(titles, words, tfidf, view_title=None, scope=None):

    if view_title:
        index = titles.index(view_title)
        if scope:
            scope.append(index)
        else:
            scope = [index, ]

    for i in scope:
        title = titles[i]
        print("-{}".format(title))

        temp = []
        row = tfidf[i]
        for j, num in enumerate(row):
            if num > 0:
                temp.append((words[j], num))
        temp = sorted(temp, reverse=True, key=lambda x: x[1])

        for word, num in temp:
            print("{} -> {}".format(word, num))

        print()
        print()
