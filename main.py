# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 20:55:15 2017

@author: Chase
"""

import math
import pandas
from pandas.core.frame import DataFrame
from pandas.core.series import Series

# CSV_FILE = "C:\\Users\\Chase\\Downloads\\income_tr.csv"
CSV_FILE  = "C:\\Users\\Chase\\Documents\\Projects\\data mining\\income_small.csv"

# Significant figures of similarity values to output
SIGFIGS = 3

def dot(left: tuple, right: tuple) -> float:
    """generic dot product between two arbitrary vectors

    left: tuple vector
    right: tuple vector
    """
    return sum(x[0] * x[1] for x in zip(left, right))


def count_binary_vectors(left: Series, right: Series, cols: tuple) -> tuple:
    m01 = m10 = m00 = m11 = 0
    for c in cols:
        if left[c] > right[c]:
            m10 += 1
        elif left[c] < right[c]:
            m01 += 1
        elif left[c] == 0:
            m00 += 1
        else:
            m11 += 1
    return (m01, m10, m00, m11)


def smc(left: Series, right: Series, cols: tuple) -> float:
    """perform simple matching with the specified columns
    left: row to compare
    right: row to compare
    cols: tuple of column names to include
    """
    m01, m10, m00, m11 = count_binary_vectors(left, right, cols)
    return (m11 + m00) / (m01 + m10 + m11 + m00)


def jaccard(left: Series, right: Series, cols: tuple) -> float:
    """perform Jaccard comparison with the specified columns
    left: row to compare
    right: row to compare
    cols: tuple of column names to include
    """
    m01, m10, m00, m11 = count_binary_vectors(left, right, cols)
    return m11 / (m01 + m10 + m11)


def cosine_similarity(a: tuple, b: tuple) -> float:
    """calculate the cosine similarity between two vectors 
    """
    # (dot(a, b) / (||a|| * ||b||))
    
    dab = dot(a, b)
    mag_a = math.sqrt(dot(a, a))
    mag_b = math.sqrt(dot(b, b))

    return dab / (mag_a * mag_b)

"""
    Naive algorithm:
    
    -- generate prox map
    prox_map <- empty set
    for each row L:
        Lt <- transform(L)
        for each row R not L:
            Rt <- transform(R)
            P <- proximity(Lt, Rt)
            prox_map[L] <- (P, R.id)

    -- output
    print headers up to k
    for each row L:
        print L.id

        sort(prox_map[L]) on first tuple value
        for each prox_map[L] to k:
            print prox_map[L]

    Possible optimizations:
    * Currently at least O(n)
    * We know proximity(Lt, Rt) so
        we shouldn't need to calculate
        proximity(Rt, Lt)
    * Pre-transform of the complete
        dataset in bulk might be faster?
    * Storing a map of ALL proximities
        and then sorting on each during
        output is speed+storage intensive.
        We would only need to store k 
        proximities and insert IFF 
        P < all P in prox_map[L] OR 
        len(prox_map[L]) < k
"""

def print_header(k: int) -> None:
    """
    k: number of proximity columns to print
    """
    # Number to ordinal prettifier
    # code golf credit to: https://stackoverflow.com/a/20007730
    ordinal = lambda n: '%d%s' % (n,'tsnrhtdd'[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])

    cols = ['\t' + ordinal(x+1) + '\t' + str(x+1) + '-prox' for x in range(k)]
    print('ID', *cols)


def print_proximities(id: str, p_arr: list) -> None:
    """
    id: id value to print
    p_array: proximity list to print (list of tuples)
    """
    cols = ['\t' + str(x[1]) + '\t' + str(round(x[0], SIGFIGS)) for x in p_arr]
    print(id, *cols)


def transform(df: DataFrame, row: Series) -> tuple:
    """perform initial data transformation into a tuple

    Note that dtype isn't reliable for a distinct row,
    so we instead rely on the df's dtype for the column

    df: Dataset
    row: Pandas Series
    """
    
    """
        To be done:
        - handle outliers
        - handle differently scaled attributes
            ordinal similarity:
                1 - |p-q|/(n-1) | p & q are mapped in 0 -> n - 1
                and n is the number of values
                ordinal being *ranking* (say happiness 1-10)
                not a category
        - handle missing values
        - 
    """
    
    return (
        row['age'], 
        row['education_cat'], 
        row['hour_per_week'],
        1 if row['gender'].strip() == 'Male' else 0
    )


def proximity(left: tuple, right: tuple) -> float:
    """calculate proximity between two records

    left: Pandas series
    right: Pandas series
    """
    
    """
        TODO: 
        Not a cosine similarity of the entire tuple.
        Instead run different prox algos per record
        (or vectors of records?) and combine for 
        a single similarity value
    """
    return cosine_similarity(left, right)


def calculate(df: DataFrame, k: int) -> None:
    """naive implementation of applying transformations,
        proximity calculations, and final output
        
        -- generate prox map
        prox_map <- empty set
        for each row L:
            Lt <- transform(L)
            for each row R not L:
                Rt <- transform(R)
                P <- proximity(Lt, Rt)
                prox_map[L] <- (P, R.id)
    
        -- output
        print headers up to k
        for each row L:
            print L.id
    
            sort(prox_map[L]) on first tuple value
            for each prox_map[L] to k:
                print prox_map[L]
    """
    prox_map = {}
    for i, left in df.iterrows():
        prox_map[i] = []

        for j, right in df.iterrows():
            if i != j:
                leftT = transform(df, left)
                rightT = transform(df, right)
                prox_map[i].append(
                    (proximity(leftT, rightT), right['ID'])
                )

    # Print headers
    print_header(k)

    for i, left in df.iterrows():
        prox_map[i].sort(key = lambda x: x[0], reverse = True)
        print_proximities(left['ID'], prox_map[i][:k])


def fast_calculate(df: DataFrame, k: int) -> None:
    """optimized variation over calculate()"""
    # https://stackoverflow.com/questions/7837722/what-is-the-most-efficient-way-to-loop-through-dataframes-with-pandas
    pass

def create_dummy_variables(df: DataFrame, cat: str):
    """expand categorical attribute into dummy variables"""
    uniques = [x for x in df[cat].unique()]
    for u in uniques:
        df[u] = df['race'].apply(lambda x: x == u)
    

if __name__ == '__main__':
    df = pandas.read_csv(CSV_FILE)
    k = 4
  
    # df['foo'] = df[['race', 'age']].apply(lambda x: x[0] + str(x[1]), axis = 1)
        
    create_dummy_variables(df, 'race')
    create_dummy_variables(df, 'workclass')
    print(df)
    
    #calculate(df, k)
