from __future__ import division

import argparse
import nltk
import os
import pickle
import string
import math

from collections import defaultdict

IGNORE_STOPWORDS = False
stopwords = set()
if IGNORE_STOPWORDS:
    stopwords = set(nltk.corpus.stopwords.words('english'))

PUNCTUATION = set(string.punctuation)
UNIVERSAL_SET_KEY = '.'


def main():
    parser = argparse.ArgumentParser(
        prog='CS3245 HW3', description='CS3245 HW3')
    parser.add_argument('-i', required=True, help='directory-of-documents')
    parser.add_argument('-d', required=True, help='dictionary-file')
    parser.add_argument('-p', required=True, help='postings-file')

    args = parser.parse_args()
    directory_path = args.i
    dictionary_file_name = args.d
    postings_file_name = args.p

    build_index(directory_path, dictionary_file_name, postings_file_name)


def tf_wt(tf):
    if tf is 0:
        return tf
    else:
        return 1 + log(tf)


def log(x):
    if x is 0:
        return 0
    else:
        return math.log(x, 10)


def build_index(directory_path='/Users/WD/nltk_data/corpora/reuters/training/',
                dictionary_file_name='dictionary.txt',
                postings_file_name='postings.txt'):
    # Sort file names numerically, not lexicographically
    doc_file_names = os.listdir(directory_path)
    doc_file_names.sort(key=int)

    stemmer = nltk.stem.porter.PorterStemmer()

    postings_lists = defaultdict(lambda: defaultdict(lambda: 0))
    postings_lists[UNIVERSAL_SET_KEY]

    # Build postings lists; treat doc file names as doc ids
    for doc_file_name in doc_file_names:
        doc_file_path = os.path.join(directory_path, doc_file_name)

        with open(doc_file_path, 'r') as doc:
            # Add doc id to universal set
            postings_lists[UNIVERSAL_SET_KEY][int(doc_file_name)]

            for line in doc:
                tokens = nltk.word_tokenize(line)
                tokens = map(lambda t: stemmer.stem(str(t).lower()), tokens)
                # filter out punctuations
                terms = filter(lambda t: t not in PUNCTUATION and
                               t not in stopwords, tokens)

                for term in terms:
                    postings_lists[term][int(doc_file_name)] += 1

    postings_lists = normalise_postings_lists(postings_lists)

    # Stores start/end pointers to postings list within postings file
    # To be written to the dictionary file
    ptr_dictionary = {}

    # Stores postings lists
    with open(postings_file_name, 'w') as postings_file:

        # Write postings lists to file, storing start/end pointers
        for term, postings_list in postings_lists.iteritems():
            start_ptr = postings_file.tell()

            postings_list = pickle.dumps(dict(postings_list))
            postings_file.write(postings_list)

            end_ptr = postings_file.tell()

            ptr_dictionary[term] = (start_ptr, end_ptr)

    # Write dictionary to file
    with open(dictionary_file_name, 'w') as dictionary_file:
        pickle.dump(ptr_dictionary, dictionary_file)


def normalise_postings_lists(postings_lists):
    sum_of_squares_tf = defaultdict(lambda: 0)

    # Calculates the weighted tf for each term, keeps track of the sum of tf
    # squares to be used to calculate the doc length
    for key, postings_list in postings_lists.iteritems():
        if key is not UNIVERSAL_SET_KEY:
            for doc_id, tf in postings_list.iteritems():
                tf = tf_wt(tf)
                postings_list[doc_id] = tf
                sum_of_squares_tf[doc_id] += tf**2

    # Calculates the doc length for each document to be used for normalizing
    doc_length = {}
    for doc_id, square_doc_length in sum_of_squares_tf.iteritems():
        doc_length[doc_id] = math.sqrt(square_doc_length)

    # Normalizes the tf_idf for each term
    for key, postings_list in postings_lists.iteritems():
        if key is not UNIVERSAL_SET_KEY:
            for doc_id, tf in postings_list.iteritems():
                postings_list[doc_id] = tf/doc_length[doc_id]

    return postings_lists

if __name__ == '__main__':
    build_index()
    # main()
