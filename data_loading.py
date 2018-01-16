import pandas as pd
import numpy as np

def read():
    df = pd.read_csv("data/mbti.csv")

    types = np.array(df.type.values)
    unique_types = np.unique(types)

    values = [list()] * len(unique_types)
    uniques = dict(zip(list(unique_types), values))
    # print(uniques)

    """
    NA OVAJ NACIN NE RADI
    NPR ENFJ IMA 190 A ENFP 675
    KADA ISPISEM DUZINE U OBA KAO I U OSTALIH 16 LISTI DUZINA JE 190+675
    for i, row in df.iterrows():
        key = row.type
        posts = row.posts.split('|||')
        if key == 'ENFJ':
            uniques['ENFJ'].append(posts)
        if key == 'ENFP':
            uniques['ENFP'].append(posts)

    # print(len(uniques['ENFJ']))
    # print(len(uniques['ENFP']))   
    """
    # ----------------------------------------------------------------
    ENFJ = []
    ENFP = []
    ENTJ = []
    ENTP = []
    ESFJ = []
    ESFP = []
    ESTJ = []
    ESTP = []
    INFJ = []
    INFP = []
    INTJ = []
    INTP = []
    ISFJ = []
    ISFP = []
    ISTJ = []
    ISTP = []

    for i, row in df.iterrows():
        key = row.type
        posts = row.posts.split('|||')
        if key == 'ENFJ': ENFJ.append(posts)
        if key == 'ENFP': ENFP.append(posts)
        if key == 'ENTJ': ENTJ.append(posts)
        if key == 'ENTP': ENTP.append(posts)
        if key == 'ESFJ': ESFJ.append(posts)
        if key == 'ESFP': ESFP.append(posts)
        if key == 'ESTJ': ESTJ.append(posts)
        if key == 'ESTP': ESTP.append(posts)
        if key == 'INFJ': INFJ.append(posts)
        if key == 'INFP': INFP.append(posts)
        if key == 'INTJ': INTJ.append(posts)
        if key == 'INTP': INTP.append(posts)
        if key == 'ISFJ': ISFJ.append(posts)
        if key == 'ISFP': ISFP.append(posts)
        if key == 'ISTJ': ISTJ.append(posts)
        if key == 'ISTP': ISTP.append(posts)
    """
    print(len(ENFJ))
    print(len(ENFP))
    print(len(ENTJ))
    print(len(ENTP))
    print(len(ESFJ))
    print(len(ESFP))
    print(len(ESTJ))
    print(len(ESTP))
    print(len(INFJ))
    print(len(INFP))
    print(len(INTJ))
    print(len(INTP))
    print(len(ISFJ))
    print(len(ISFP))
    print(len(ISTJ))
    print(len(ISTP))
    """
    posts_by_type = []
    posts_by_type.append(ENFJ)
    posts_by_type.append(ENFP)
    posts_by_type.append(ENTJ)
    posts_by_type.append(ENTP)
    posts_by_type.append(ENFJ)
    posts_by_type.append(ESFP)
    posts_by_type.append(ESTJ)
    posts_by_type.append(ESTP)
    posts_by_type.append(INFJ)
    posts_by_type.append(INFP)
    posts_by_type.append(INTJ)
    posts_by_type.append(INTJ)
    posts_by_type.append(INTP)
    posts_by_type.append(ISFJ)
    posts_by_type.append(ISTJ)
    posts_by_type.append(ISTP)

    mbti = dict(zip(list(unique_types), posts_by_type))
    for key, value in mbti.items():
        print(key, ':', len(value))

    return mbti
    # ----------------------------------------------------------------
