import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
import seaborn


types = ['INFP', 'ESFJ', 'ESFP', 'ISFJ', 'ISFP', 'ESTJ', 'ESTP', 'ISTJ', 'ISTP', 'INTJ', 'INFJ', 'ENTJ', 'ENTP', 'ENFJ',
         'INTP', 'ENFP']


def read_new():
    df = pd.read_csv("data/mbti.csv")

    plt.figure()
    plt.xticks(fontsize=8)
    seaborn.countplot(data=df, x='type')
    plt.show()

    data = dict([(t, []) for t in types])

    for i, row in df.iterrows():
        data[row.type].extend(row.posts.split('|||'))

    return data
