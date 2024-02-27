
"""
Various utilities
"""


import numpy as np
import pandas as pd
from collections import OrderedDict


def load_data(mode='analysis', format=None):
    arg_values = dict(mode=['analysis', 'model', 'predict'],
                      format=[('dataframe', 'df', pd.DataFrame), ('numpy', 'array', 'nd', 'ndarray', 'matrix', np.array, np.ndarray)])
    mode = ([s for s in arg_values['mode'] if s in str(mode).lower()] + [None])[0]
    format = ['dataframe', 'matrix', None][([format in t for t in arg_values['format']] + [True]).index(True)]


    # if the data is needed for prediction/deployment/production -> output ndarray for a sklearn pipeline
    if mode=='predict':
        # substitute with real code
        df = pd.read_csv('data/german.data', delimiter=' ', header=None).iloc[:, :20]  # the first 20 columns
        return df if format=='dataframe' else df.values
    
    # otherwise
    df = pd.read_csv('data/german.data', delimiter=' ', header=None)
    
    if mode=='model':
        df.iloc[:, -1] = df.iloc[:, -1] - 1
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        return (X,y) if format=='dataframe' else (X.values, y.values)
    
    if mode=='analysis':
        df.columns = [
            "status",    #1 (balance)
            "tenure",  #2 (tenure)
            "history",   #3
            "purpose",   #4 (purpose of the loan)
            "amount",    #5 (credit amount requested)
            "savings",   #6 (savings or bonds)
            "employment",#7 (current employment duration)
            "rate",     #8 (Installment rate in percentage of disposable income)
            "personal",  #9 (Personal status and sex)
            "guarantor", #10
            "residence", #11
            "property",  #12
            "age",       #13
            "installments", #14 (Other installment plans)
            "housing",    #15
            "credits",      #16 (Number of existing credits at this bank)
            "job",        #17
            "maintenance", #18  child maintenance / support
            "telephone",  #19
            "foreign",    #20
            "label"
                    ]
        df["sex"] = np.where(df['personal'].isin(['A92', 'A95']), "female", np.where(df['personal'].isin(['A91', 'A93', 'A94']), "male", "?"))

        mapping = {
        0: {
            'A11' : "overdrawn",
            'A12' : "petty",   # up to 200 DM
            'A13' : "salary",
            'A14' : "no account"
            },

        2: OrderedDict([
            ('A30', "no loans"),
            ('A31', "duly paid back"),
            ('A32', "so far so good"),
            ('A33', "delay"),
            ('A34', "critical"),
        ]), 

        3: {
            'A40' : "car",
            'A41' : "used car",
            'A42' : "furniture",
            'A43' : "television",
            'A44' : "appliances",
            'A45' : "repairs",
            'A46' : "education",
            'A47' : "vacation",
            'A48' : "retraining",
            'A49' : "business",
            'A410': "other",
        },

        5: OrderedDict([
            ('A65', "no savings"),  # this item was the last one
            ('A61', pd.Interval(0, 100, closed='left')),   #"[0, 100)"
            ('A62', pd.Interval(100, 500, closed='left')),   #"[100, 500)"
            ('A63', pd.Interval(500, 1000, closed='left')),  #"[500, 1000)"
            ('A64', pd.Interval(1000, np.inf,  closed='left')),  #"[1000, ∞)"
        ]), 

        6: OrderedDict([
            ('A71', "unemployed"),
            ('A72', pd.Interval(0, 1, closed='left')),     #"[0, 1)"
            ('A73', pd.Interval(1, 4, closed='left')),     #"[1, 4)"
            ('A74', pd.Interval(4, 7, closed='left')),     #"[4, 7)"
            ('A75', pd.Interval(7, np.inf, closed='left')),     #"[7, inf)"
        ]),

        8: {
            'A91' : "male divorced/separated",
            'A92' : "female divorced/separated/married",
            'A93' : "male single",
            'A94' : "male married/widowed",
            'A95' : "female single",
        },

        9: {
            'A101' : "none",
            'A102' : "co-applicant",
            'A103' : "guarantor",
        },

        11: OrderedDict([
            ('A121', "real estate"),
            ('A122',  "building savings / life insurance"),
            ('A123',  "car"),
            ('A124',  "none"),
        ]),

        13: {
            'A141' : "loan",    # bank
            'A142' : "consumer",  # stores
            'A143' : "none",    # none
        },

        14: {
            'A151' : "rent",
            'A152' : "ownership",       # own
            'A153' : "without payment",   # for free
        },

        16: OrderedDict([
            ('A171', "unemployed/unskilled non-resident"),
            ('A172', "unskilled"),
            ('A173', "skilled"),
            ('A174', "white color"),
        ]),

        18: {
            'A191' : "none",
            'A192' : "yes",
        },

        19: {
            'A201' : True,
            'A202' : False,
        },

        20: {
            1: 0,  # good
            2: 1,  # bad
        }}

        # Rename the categories
        for j, d in mapping.items():
            if isinstance(d, OrderedDict):
                # convert to an ordered categorical
                df[df.columns[j]] = df.iloc[:, j].map(d).astype('category').cat.as_ordered().cat.reorder_categories(d.values())
            else:
                # otherwise, keep the values as str
                df.iloc[:, j] = df.iloc[:, j].map(d)

        # Rearrange the columns: num, cat, str, label
        df = pd.concat([df.select_dtypes(np.number), df.select_dtypes('category'), df.select_dtypes('object')], axis=1)
        df['personal'] = df.pop('personal')
        df['label'] = df.pop('label')   # move to back
        return df
    
    raise ValueError("unable to output requested data due to arguments not understood")
 



