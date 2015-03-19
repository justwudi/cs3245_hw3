import argparse
import math
import nltk
import pickle
import string
import operator
from colelctions import defaultdict

PUNCTUATION = set(string.punctuation)
UNIVERSAL_SET_KEY = '.'
NOT_PREFIX = 'N_'
OR_PREFIX = 'O_'
AND_PREFIX = 'A_'

ptr_dictionary = None
postings_file = None
universal_set = None
stemmer = None
results = None


def main():
    parser = argparse.ArgumentParser(
        prog='CS3245 HW3', description='CS3245 HW3')
    parser.add_argument('-d', required=True, help='dictionary-file')
    parser.add_argument('-p', required=True, help='postings-file')
    parser.add_argument('-q', required=True, help='file-of-queries')
    parser.add_argument('-o', required=True, help='output-file-of-results')

    args = parser.parse_args()
    dictionary_file_name = args.d
    postings_file_name = args.p
    query_file_name = args.q
    output_file_name = args.o

    execute_queries(dictionary_file_name, postings_file_name,
                    query_file_name, output_file_name)


def execute_queries(dictionary_file_name='dictionary.txt',
                    postings_file_name='postings.txt',
                    query_file_name='query.txt',
                    output_file_name='output.txt'):
    with open(dictionary_file_name, 'r') as dictionary_file:
        global ptr_dictionary
        ptr_dictionary = pickle.load(dictionary_file)

    global postings_file
    postings_file = open(postings_file_name, 'r')

    global stemmer
    stemmer = nltk.stem.porter.PorterStemmer()

    query_file = open(query_file_name, 'r')
    output_file = open(output_file_name, 'w')

    for line in query_file:
        query_terms = line.split(' ')
        query_terms = map(lambda t: stemmer.stem(t.lower()), query_terms)
        result = rankedSearch(query_terms)
        output_file.write(' '.join(str(doc_id) for doc_id in result[:10]))
        output_file.write('\n')

    output_file.close()
    query_file.close()
    postings_file.close()


def rankedSearch(query_terms):
    result = defaultdict(lambda: 0)
    for term in query_terms:
        postings = read_postings_dict(term)

        for doc_id, score in postings.iteritems():
            result[doc_id] += score

    return sorted(result.items(), key=operator.itemgetter(1))


def read_postings_dict(term):
    if term not in ptr_dictionary:
        return {}

    start_ptr, end_ptr = ptr_dictionary[term]

    postings_file.seek(start_ptr)
    postings_list_pickle = postings_file.read(end_ptr - start_ptr)
    postings_dict = pickle.loads(postings_list_pickle)

    return postings_dict


if __name__ == '__main__':
    execute_queries()
    # main()
