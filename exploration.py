# -*- coding: utf-8 -*-

import pandas
from pandas.plotting import scatter_matrix

import matplotlib
import matplotlib.pyplot as plt

# Adjust figure sizes created by pyplot
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)


def histogram(dataset, attr: str = None) -> None:
    """histogram of an attribute or all attributes
    
    attr: optional attribute name to plot
    """
    if attr:
        dataset[attr].hist()
        plt.title(attr)
    else:
        dataset.hist()
        
    plt.show()

def scatter(dataset, x: str, y: str) -> None:
    """render a scatter plot of two distinct attributes
    
    dataset: Pandas data
    x: X axis label and attribute name
    y: Y axis label and attribute name
    """
    plt.scatter(dataset[x], dataset[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Scatter of {} vs {}'.format(x, y))
    plt.show()

def snapshot(dataset) -> None:
    """high level analysis snapshot of the dataset
    
    dataset: Pandas data
    """
    
    # Quick view of what the data looks like
    print(dataset.head(5))
    
    # Some basic summary statistics
    # means, N percentiles, standard deviations, etc
    print(dataset.describe())
    
    # Box and whisker plots of all properties 
    dataset.plot(
        kind='box', 
        subplots=True, 
        layout=(2,4), 
        sharex=False, 
        sharey=False
    )
    plt.show()
    
    # Matrix of all attributes compared to one another
    # as scatter plots. The cross indices A_(j,j)
    # are histograms of each distinct attribute
    scatter_matrix(dataset)
    plt.show()
    
def load_iris():
    # Load dataset
    url = "C:\\Users\\Chase\\Downloads\\iris.data"
    names = [
        'sepal-length', 
        'sepal-width', 
        'petal-length', 
        'petal-width', 
        'class'
    ]
    
    dataset = pandas.read_csv(url, names=names)
    return dataset

def classify_occupation(record):
    return len(record)
    
def load_income():
    # Load dataset
    url = "C:\\Users\\Chase\\Downloads\\income_tr.csv"
    
    dataset = pandas.read_csv(url)
    return dataset

def report_uniques(df):
    """report uniques from string-based attributes """
    for column in df:
        """ Grab some samples """
        if df[column].dtypes == 'object':
            print(column, len(df[column].unique()), 'unique values')
            print(df[column].unique(),"\n")


if __name__ == '__main__':    
    
    df = load_income()
    
    #report_uniques(df)
    
    for i, row in df.iterrows():
        for j, r in df.iterrows():
            if i == j:
                print(row['education'])
    
    #print(df['relationship'].sample())
    
    
    
    #dataset['occ_class'] = dataset['occupation'].apply(classify_occupation)
    
    #print(dataset.head(5))
    
    #snapshot(dataset)

    