# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 20:55:15 2017

@author: Chase
"""

import math
import argparse
import pandas
from time import time
from pandas.core.frame import DataFrame
from pandas.core.series import Series

# Lookup cache for dummy variables
# Avoids frequent list comprehension
g_dummy_cache = {}

# Lookup cache for row vectors calculated
# in alt_proximity
g_vec_cache = {}

def dot(left: list, right: list) -> float:
    """generic dot product between two arbitrary vectors

    left: vector
    right: vector
    """
    return sum(float(x[0]) * x[1] for x in zip(left, right))


def count_binary_vectors(left: Series, right: Series, cols: tuple) -> tuple:
    """create an (m01, m10, m00, m11) tuple by counting matches in cols

    left: vector
    right: vector
    cols: list of column names to include in the subvector
    """
    m01 = m10 = m00 = m11 = 0
    for c in cols:
        l = left[c]
        r = right[c]
        if l == 1:
            if r == 0:
                m10 += 1
            else:
                m11 += 1
        elif r == 0:
            m00 += 1
        else:
            m01 += 1

    return (m01, m10, m00, m11)


def smc(left: Series, right: Series, cols: list) -> float:
    """create a simple matching coefficient of specified columns
        within two vectors

    left: vector
    right: vector
    cols: list of column names to include in the subvector
    """
    m01, m10, m00, m11 = count_binary_vectors(left, right, cols)
    return (m11 + m00) / (m01 + m10 + m11 + m00)


def jaccard(left: Series, right: Series, cols: list) -> float:
    """create a Jaccard coefficient of specified columns
        within two vectors

    left: vector
    right: vector
    cols: list of column names to include in the subvector
    """
    m01, m10, m00, m11 = count_binary_vectors(left, right, cols)
    return m11 / (m01 + m10 + m11)


def fast_jaccard(left: list, right: list) -> float:
    try:
        m01 = m10 = m00 = m11 = 0
        for i, l in enumerate(left):
            if l == 0:
                if i >= len(right) or right[i] == 0:
                    m00 += 1
                else:
                    m01 += 1
            elif i >= len(right) or right[i] == 0:
                m10 += 1
            else:
                m11 += 1
    except:
        print(left)
        print(right)
        raise

    return m11 / (m01 + m10 + m11)


def cosine_similarity(left: list, right: list) -> float:
    """calculate the cosine similarity between two vectors

    left: vector
    right: vector
    """
    # (dot(a, b) / (||a|| * ||b||))

    d = dot(left, right)
    l_mag = math.sqrt(dot(left, left))
    r_mag = math.sqrt(dot(right, right))

    return d / (l_mag * r_mag)


def distance(left: Series, right: Series, col: str) -> float:
    """basic distance calculation of an interval attribute

    left: Pandas series
    right: Pandas series
    col: interval attribute name
    """
    return 1 / (1 + abs(left[col] - right[col]))


def print_prox_map_header(k: int) -> str:
    """
    k: number of proximity columns to print
    """
    # Number to ordinal prettifier
    # code golf credit to: https://stackoverflow.com/a/20007730
    ordinal = lambda n: '%d%s' % (n,'tsnrhtdd'[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])

    cols = [ordinal(x+1) + '\t' + str(x+1) + '-prox' for x in range(k)]
    return 'ID\t' + '\t'.join(cols) + '\n'


def print_proximities(id: str, p_arr: list, sigfigs: int) -> str:
    """
    id: id value to print
    p_array: proximity list to print (list of tuples)
    """
    cols = [str(x[1]) + '\t' + str(round(x[0], sigfigs)) for x in p_arr]
    return str(id) + '\t' + '\t'.join(cols) + '\n'


def get_dummy_variables(df: DataFrame, col: str) -> list:
    """
    """
    # Uses a cache so we don't have to constantly do list comprehension
    if col not in g_dummy_cache:
        g_dummy_cache[col] = [x for x in df[col].unique()]

    return g_dummy_cache[col]


def proximity(df: DataFrame, left: Series, right: Series) -> float:
    """calculate proximity between two records

    df: Pandas DataFrame
    left: Pandas series
    right: Pandas series
    """
    vector = [
        # for categoricals we expanded out into dummy variables,
        # we perform SMC (0 relevant) or Jaccard (for long 0 sequences)
        smc(left, right, get_dummy_variables(df, 'gender')),
        #jaccard(left, right, get_dummy_variables(df, 'race')),
        #jaccard(left, right, get_dummy_variables(df, 'workclass')),
        #jaccard(left, right, get_dummy_variables(df, 'marital_status')),
        #jaccard(left, right, get_dummy_variables(df, 'relationship')),
        #jaccard(left, right, get_dummy_variables(df, 'native_country')),

        fast_jaccard(left['race_vec'], right['race_vec']),
        fast_jaccard(left['workclass_vec'], right['workclass_vec']),
        fast_jaccard(left['marital_status_vec'], right['marital_status_vec']),
        fast_jaccard(left['relationship_vec'], right['relationship_vec']),
        fast_jaccard(left['native_country_vec'], right['native_country_vec']),
        fast_jaccard(left['occupation_vec'], right['occupation_vec']),

        # For intervals, we do a basic distance calculation 1/(1+|p-q|)
        # TODO: 0 rejection? Need to see if the data has some bad values or not
        distance(left, right, 'age'),
        distance(left, right, 'education_cat'),
        distance(left, right, 'hour_per_week')
    ]

    # For sparse data with some major outliers, we'll ignore 0's
    # and just use distance
    if left['capital_gain'] > 0 or right['capital_gain'] > 0:
        vector.append(distance(left, right, 'capital_gain'))

    if left['capital_loss'] > 0 or right['capital_loss'] > 0:
        vector.append(distance(left, right, 'capital_loss'))

    # Finally perform a combined similarity of all the similarities
    # calculated per attribute (or set of attributes)
    return sum(vector) / len(vector)


def alt_proximity(df: DataFrame, left: Series, right: Series) -> float:
    """alternative proximity function for two records

    This proximity method vectorizes the entire row and
    performs a simple cosine similarity between left and right.

    df: Pandas DataFrame
    left: Pandas series
    right: Pandas series
    """
    def vectorize(s: Series) -> list:
        if s['ID'] not in g_vec_cache:
            # Aggregate vector caches
            v = s['gender_vec'] + s['race_vec'] + s['workclass_vec'] \
                + s['marital_status_vec'] + s['relationship_vec'] \
                + s['native_country_vec'] + s['occupation_vec']

            # Aggregate other numeric attributes
            # normalized to min/max of each attribute
            # so that they don't overtake one another for cos-sim.
            # Note I hardcode these to the full income_tr
            # dataset. I *could* read during initial transfoms
            # and cache, but... meh.
            g_vec_cache[s['ID']] = v + [
                1 - (s['age'] - 17) / (82 - 17),  # 17 - 82
                1 - (s['education_cat'] - 1) / (16 - 1),  # 1 - 16
                1 - (s['hour_per_week'] - 2) / (99 - 2),  # 2 - 99
                1 - s['capital_gain'] / 99999,  # 0 - 99999
                1 - s['capital_loss'] / 4356 # 0 - 4356
            ]

        return g_vec_cache[s['ID']]

    return cosine_similarity(
        vectorize(left),
        vectorize(right)
    )


def calculate(df: DataFrame, k: int) -> dict:
    """naive implementation of calculating proximities.

        tl;dr:
            for left in rows:
                for right in rows:
                    map <- proximity(left, right)

        Fine for small number of rows, but performance
        is impacted exponentially for large rowsets.
    """
    prox_map = {}
    for i, left in df.iterrows():
        prox_map[i] = []

        for j, right in df.iterrows():
            if i != j and len(prox_map[i]) < k:
                prox_map[i].append(
                    (proximity(df, left, right), right['ID'])
                )

    return prox_map


def print_prox_map(df: DataFrame, prox_map: dict, k: int, sigfigs: int) -> str:
    """create a matrix of proximities

    """
    # Print headers
    out = print_prox_map_header(k)

    for i, row in df.iterrows():
        prox_map[i].sort(key = lambda x: x[0], reverse = True)
        out += print_proximities(row['ID'], prox_map[i][:k], sigfigs)

    return out


def fast_calculate(df: DataFrame, k: int, prox_func) -> list:
    """optimized variation over calculate()"""
    # We first extract every series immediately, as every time
    # df.iterrows() is called, it creates a new Series instance
    # per iteration.
    series_set = [x for i, x in df.iterrows()]

    n = len(series_set)
    prox_map = [[] for _ in range(n)] # Preallocate indices

    #prox_map = {}

    # Iterate through, performing proximity comparisons
    # forward of each index. Should be about O(nlog(n))
    for i in range(0, n):
        left = series_set[i]
        l_id = left['ID']
        for j in range(i + 1, n):
            right = series_set[j]
            r_id = right['ID']

            p = prox_func(df, left, right)
            prox_map[i].append((p, r_id))
            prox_map[j].append((p, l_id))

    return prox_map


def create_dummy_variables(df: DataFrame, cat: str) -> None:
    """expand a categorical attribute into dummy variables
        named after each unique category
    """
    # some may have " ?" as a category, clean these out first
    # so there aren't collisions later on
    df[cat] = df[cat].apply(lambda x: x if x.strip() != '?' else cat + '-unk')

    uniques = [x for x in df[cat].unique()]
    for u in uniques:
        df[u] = df[cat].apply(lambda x: int(x == u))

    # Add a cache for fast vector lookup
    df[cat + '_vec'] = df.apply(lambda r: r[uniques].tolist(), axis = 1)


def apply_base_transformations(df: DataFrame) -> None:
    """apply initial transformations to normalize/binarize attributes

    df: Pandas DataFrame to transform
    """
    # Transformation rules are based on initial data exploration
    # of the income dataset

    # Expand categorical attributes we care about
    create_dummy_variables(df, 'gender')
    create_dummy_variables(df, 'race')
    create_dummy_variables(df, 'workclass')
    create_dummy_variables(df, 'native_country')
    create_dummy_variables(df, 'marital_status')
    create_dummy_variables(df, 'relationship')
    create_dummy_variables(df, 'occupation')


def knn_majority_vote(proximities: list, k: int):
    """calculate class by majority vote

        Accepts a list of tuples (prox, class) and
        returns a class and a posterior probability in [0,1]
    """
    classes = {
        ' >50K': 0, # positive test
        ' <=50K': 0 # negative test
    }

    # sort proximities by closest first
    proximities.sort(key = lambda x: x[0], reverse = True)

    # Grab the majority class of the k-nearest proximities
    for i in range(k):
       # if proximities[i][1] not in classes:
       #     classes[proximities[i][1]] = 0

        classes[proximities[i][1]] += 1

    # keys = list(classes.keys())
    # v = list(classes.values())
    # selected_class = keys[v.index(max(v))]
    # return selected_class, max(v) / qk
    if classes[' >50K'] > classes[' <=50K']:
        selected_class = ' >50K'
    else:
        selected_class = ' <=50K'
    
    # posterior probability is the probability the result was in >50K (positive)
    posterior_probability = classes[' >50K'] / sum(classes.values())
    return selected_class, posterior_probability


def knn_weighted(proximities: list, k: int):
    """calculate class by proximity distances

        Accepts a list of tuples (prox, class) and
        returns a class and a posterior probability in [0,1]
    """
    classes = {
        ' >50K': 0, # positive test
        ' <=50K': 0 # negative test
    }

    # sort proximities by closest first
    proximities.sort(key = lambda x: x[0], reverse = True)

    # for k-nearest proximities, sum their proximities
    # under their class
    for i in range(k):
        # if proximities[i][1] not in classes:
        #    classes[proximities[i][1]] = 0

        classes[proximities[i][1]] += 1 / (proximities[i][0] ** 2)

    # keys = list(classes.keys())
    # v = list(classes.values())
    # selected_class = keys[v.index(max(v))]
    # return selected_class, max(v) / qk
    if classes[' >50K'] > classes[' <=50K']:
        selected_class = ' >50K'
    else:
        selected_class = ' <=50K'
    
    # posterior probability is the probability the result was in >50K (positive)
    posterior_probability = classes[' >50K'] / sum(classes.values())
    return selected_class, posterior_probability


def knn_classifier_k_sweep(
    training_df: DataFrame,
    test_df: DataFrame,
    k_limit: int,
    prox_func,
    knn_func
):
    """runs the bulk of generating a prox map and then does a sweep of
        a range of K values, outputting match percentages for each
    """
    training_set = [x for i, x in training_df.iterrows()]
    test_set = [x for i, x in test_df.iterrows()]

    training_len = len(training_set)
    test_len = len(test_set)

    prox_map = [[] for _ in range(test_len)] # Preallocate indices

    # Generate proximities between everything in the
    # test set against everything in the training set
    for i in range(0, test_len):
        left = test_set[i]
        for j in range(0, training_len):
            right = training_set[j]

            p = prox_func(training_df, left, right)
            prox_map[i].append((p, right['class']))

    for k in range(1, k_limit):
        for i in range(0, test_len):
            positives = 0

            # Calculate class and add to classification map as:
            # (test ID, actual class, predicted class, posterior prob)
            predicted_class, probability = knn_func(prox_map[i], k)

            if predicted_class == test_set[i]['class']:
                positives += 1

        print('k={}\t{}'.format(k, round(positives / test_len, 5)))


def knn_classifier(
    training_df: DataFrame,
    test_df: DataFrame,
    k: int,
    prox_func,
    knn_func
) -> list:
    """kNN classifier"""
    training_set = [x for i, x in training_df.iterrows()]
    test_set = [x for i, x in test_df.iterrows()]

    training_len = len(training_set)
    test_len = len(test_set)

    prox_map = [[] for _ in range(test_len)] # Preallocate indices
    classification_map = []

    # Generate proximities between everything in the
    # test set against everything in the training set
    for i in range(0, test_len):
        left = test_set[i]
        for j in range(0, training_len):
            right = training_set[j]

            p = prox_func(training_df, left, right)
            prox_map[i].append((p, right['class']))

        # Calculate class and add to classification map as:
        # (test ID, actual class, predicted class, posterior prob)
        predicted_class, probability = knn_func(prox_map[i], k)
        classification_map.append((
            left['ID'],
            left['class'],
            predicted_class,
            probability
        ))

    return classification_map


def print_knn_classifier_header() -> str:
    return 'ID\tActual\tPredicted\tPosterior Probability\n'


def print_knn_classification(classification: tuple) -> str:
    """

    classification: tuple in the form of (ID, actual class, predicted class, probability)
    """
    return '{}\t{}\t{}\t{}\n'.format(*classification)


def print_classification_map(classifications: list) -> str:

    # Print headers
    out = print_knn_classifier_header()

    for classification in classifications:
        out += print_knn_classification(classification)

    return out


def print_knn_accuracy(classifications: list):
    """dump an accuracy report to console"""

    correct = 0
    for classification in classifications:
        if classification[1] == classification[2]:
            correct += 1

    print(str(round(correct / len(classifications), 5)) + ' match to test classes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Income proximities'
    )

    parser.add_argument(
        '--k',
        default=4,
        type=int,
        help='K-value'
    )
    parser.add_argument(
        '--limit',
        default=0,
        type=int,
        help='Record limit to analyze'
    )
    parser.add_argument(
        '--alt',
        action='store_true',
        help='Use alternate proximity algorithm'
    )
    parser.add_argument(
        '--sig',
        default=3,
        type=int,
        help='Number of sigfigs in proximity table'
    )
    parser.add_argument(
        '--output',
        help='Output CSV filename'
    )
    parser.add_argument(
        '--knn',
        action='store_true',
        help='Run kNN classifier'
    )
    parser.add_argument(
        '--weighted',
        action='store_true',
        help='Use weighted distances for kNN classifier'
    )
    parser.add_argument(
        '--test',
        help='Test dataset CSV filename'
    )
    parser.add_argument('filename')

    args = parser.parse_args()

    # Load training file
    training_df = pandas.read_csv(args.filename)

    if args.test:
        test_df = pandas.read_csv(args.test)

    # Get a subset of training rows, if requested
    if args.limit:
        training_df = training_df[:args.limit]

    # Apply initial transformations
    now = time()

    apply_base_transformations(training_df)
    if args.test:
        apply_base_transformations(test_df)

    transformation_time = time() - now

    now = time()

    # Load either the main or alternative proximity function
    prox_func = proximity
    if args.alt:
        prox_func = alt_proximity

    # Load either majority vote or weighted kNN function
    knn_func = knn_majority_vote
    if args.weighted:
        knn_func = knn_weighted

    # Calculate proximities of everything in the training set
    if not args.knn:
        prox_map = fast_calculate(
            training_df,
            args.k,
            prox_func
        )

        calculate_time = time() - now

        # Generate a proxmity map
        now = time()
        results = print_prox_map(
            training_df,
            prox_map,
            args.k,
            args.sig
        )

    # Run a kNN classifier between the training and test sets
    else:
        # knn_classifier_k_sweep(
        #     training_df,
        #     test_df,
        #     60,
        #     prox_func
        # )

        classification_map = knn_classifier(
            training_df,
            test_df,
            args.k,
            prox_func,
            knn_func
        )

        calculate_time = time() - now

        # Generate classifier test results table
        now = time()
        results = print_classification_map(
            classification_map
        )

        print_knn_accuracy(classification_map)

    # Write to either the output file or to console
    if args.output:
        with open(args.output, 'w') as f:
            f.write(results.replace('\t', ','))
        print('Wrote results to {}'.format(args.output))
    else:
        print(results)

    print_time = time() - now

    # Spit out some basic runtime statistics
    print('Transformed in {:.3f} seconds'.format(transformation_time))
    print('Processed in {:.3f} seconds'.format(calculate_time))
    print('Printed in {:.3f} seconds'.format(print_time))
