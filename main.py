# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 20:55:15 2017

@author: Chase
"""

import math
import pandas
from time import time
from pandas.core.frame import DataFrame
from pandas.core.series import Series

# CSV_FILE = "C:\\Users\\Chase\\Downloads\\income_tr.csv"
CSV_FILE  = "samples/income_tr.csv"

# Significant figures of similarity values to output
SIGFIGS = 3

K = 4

def dot(left: tuple, right: tuple) -> float:
    """generic dot product between two arbitrary vectors

    left: tuple vector
    right: tuple vector
    """
    return sum(x[0] * x[1] for x in zip(left, right))


def count_binary_vectors(left: Series, right: Series, cols: tuple) -> tuple:
    """create an (m01, m10, m00, m11) tuple by counting matches in cols
        
    left: vector
    right: vector
    cols: list of column names to include in the subvector
    """
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


def cosine_similarity(a: tuple, b: tuple) -> float:
    """calculate the cosine similarity between two vectors 
    """
    # (dot(a, b) / (||a|| * ||b||))
    
    dab = dot(a, b)
    mag_a = math.sqrt(dot(a, a))
    mag_b = math.sqrt(dot(b, b))

    return dab / (mag_a * mag_b)


def distance(left: Series, right: Series, col: str) -> float:
    """basic distance calculation of an interval attribute
    
    left: Pandas series
    right: Pandas series
    col: interval attribute name
    """
    return 1 / (1 + abs(left[col] - right[col]))


def print_header(k: int) -> str:
    """
    k: number of proximity columns to print
    """
    # Number to ordinal prettifier
    # code golf credit to: https://stackoverflow.com/a/20007730
    ordinal = lambda n: '%d%s' % (n,'tsnrhtdd'[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])

    cols = [ordinal(x+1) + '\t' + str(x+1) + '-prox' for x in range(k)]
    return 'ID\t' + '\t'.join(cols) + '\n'
        
def print_proximities(id: str, p_arr: list) -> str:
    """
    id: id value to print
    p_array: proximity list to print (list of tuples)
    """
    cols = [str(x[1]) + '\t' + str(round(x[0], SIGFIGS)) for x in p_arr]
    return str(id) + '\t' + '\t'.join(cols) + '\n'

dummy_cache = {}
def get_dummy_variables(df: DataFrame, col: str) -> list:
    """
    """
    # Uses a cache so we don't have to constantly do list comprehension
    if col not in dummy_cache:
        dummy_cache[col] = [x for x in df[col].unique()]
        
    return dummy_cache[col]
    
def proximity(df: DataFrame, left: Series, right: Series) -> float:
    """calculate proximity between two records

    left: Pandas series
    right: Pandas series
    """
    vector = [ 
        # for categoricals we expanded out into dummy variables,
        # we perform SMC (0 relevant) or Jaccard (for long 0 sequences)
        smc(left, right, get_dummy_variables(df, 'gender')),
        jaccard(left, right, get_dummy_variables(df, 'race')),
        jaccard(left, right, get_dummy_variables(df, 'workclass')),
        jaccard(left, right, get_dummy_variables(df, 'marital_status')),
        jaccard(left, right, get_dummy_variables(df, 'relationship')),
        jaccard(left, right, get_dummy_variables(df, 'native_country')),
        
        # For intervals, we do a basic distance calculation 1/(1+|p-q|)
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


def print_prox_map(df: DataFrame, prox_map: dict, k: int) -> str:
    """create a matrix of proximities
    
    """
    # Print headers
    out = print_header(k)

    for i, row in df.iterrows():
        prox_map[i].sort(key = lambda x: x[0], reverse = True)
        out += print_proximities(row['ID'], prox_map[i][:k])

    return out


def fast_calculate(df: DataFrame, k: int) -> None:
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
    
    # Still seeing ~ 4s for 50, 15s for 100, 60s for 200.
    # Iteration time *without* proximity() is 1.5s 
    
    # Implementing dummy_cache gives us:
    # ~1.7s for 50, ~8s for 100, 35s for 200
    for i in range(0, n):
        left = series_set[i]
        l_id = left['ID']
        for j in range(i + 1, n):
            right = series_set[j]
            r_id = right['ID']
            
            p = proximity(df, left, right)
            prox_map[i].append((p, r_id))
            prox_map[j].append((p, l_id))
            
    return prox_map
    

def create_dummy_variables(df: DataFrame, cat: str) -> None:
    """expand a categorical attribute into dummy variables
        named after each unique category
    """
    # some may have " ?" as a category, clean these out first
    # so there aren't collisions later on
    df[cat] = df[cat].apply(lambda x: x if x != ' ?' else cat + '-unk')
    
    uniques = [x for x in df[cat].unique()]
    for u in uniques:
        df[u] = df[cat].apply(lambda x: int(x == u))
    
    
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
    

if __name__ == '__main__':
    df = pandas.read_csv(CSV_FILE)
    
    # 7 seconds for  50 
    # 11 seconds for 60
    # 14 seconds for 70
    
    now = time()
    apply_base_transformations(df)
    transformation_time = time() - now
    
    now = time()
    prox_map = fast_calculate(df[:200], K)
    calculate_time = time() - now
    
    now = time()
    # with open('out.txt', 'w') as f:
    #    f.write(print_prox_map(df, prox_map, K))
    #results = print_prox_map(df[:70], prox_map, K)
    print_time = time() - now
    
    #print(results)
    print(len(prox_map))
    print('Transform', transformation_time)
    print('Calculate', calculate_time)
    print('Print', print_time)
    
    
    