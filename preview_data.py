import seaborn as sb
import matplotlib.pyplot as plt
import re
import operator
from functools import reduce


def preview(data):
    types_counts = data['type'].value_counts()

    plt.figure(figsize=(12, 4))
    plt.ylabel('Counts')
    plt.xlabel('Types', fontsize=10)
    types_counts.plot.bar()
    plt.show()


def preview_negations(data):
    types_counts = data['type'].value_counts().index.tolist()
    # print(data[data['type'] == 'INFP'])
    temp_data = data.groupby('type')['posts'].apply('|||'.join).reset_index()
    neg_counts = {}
    for index, row in temp_data.iterrows():
        count = get_neut_emojis_count(row[1])
        neg_counts[row[0]] = count

    plt.figure()
    plt.bar(range(len(neg_counts)), neg_counts.values(), align='center', width=0.5)
    plt.xticks(range(len(neg_counts)), neg_counts.keys())
    plt.show()


def get_pos_emojis_count(posts):
    count = reduce(operator.add, (1 for _ in re.finditer(
        re.compile(r'(?i)(\^_\^|\\o/|<3|:happy:|:proud:|:kiss:|crazy:|lol|rofl")|(((O|>)?(:|;)\'?-?|[xX]|=|B-?)(\)+|D+|\*+))'), posts)))
    return count


def get_neg_emojis_count(posts):
    count = reduce(operator.add, (1 for _ in re.finditer(
        re.compile(r'(?i)(:sad:|:crying:|</3|-.-|-_-)|(>?:\'?-?(\(+|S+|\\+))'), posts)))
    return count


def get_neut_emojis_count(posts):
    count = reduce(operator.add, (1 for _ in re.finditer(
        re.compile(r'(?i)(>.>|<.<)|(:-?(O+|P+))'), posts)))
    return count


def get_neg_counts(posts):
    count = reduce(operator.add, (1 for _ in re.finditer(
        re.compile(r'(?i)not|never|didn\'t|did not|don\'t|do not|no'), posts)))
    return count
