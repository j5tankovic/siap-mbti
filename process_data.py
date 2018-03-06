from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.preprocessing import LabelEncoder
from scrape_url import get_yt_tags
import numpy as np
import re

types = ['INFP', 'ESFJ', 'ESFP', 'ISFJ', 'ISFP', 'ESTJ', 'ESTP', 'ISTJ', 'ISTP', 'INTJ', 'INFJ', 'ENTJ', 'ENTP', 'ENFJ',
         'INTP', 'ENFP']
label_encoder = LabelEncoder().fit(types)


# First step - remove all URLs from posts
def replace_non_words(posts):
    posts = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', posts)
    posts = re.sub("[.,:?!+]", " ", posts)
    posts = re.sub("[^a-zA-Z]", " ", posts)
    posts = re.sub(' +', ' ', posts).lower()
    return posts


# Second step - use lemmatization
def lemmatize(lemmatizer, posts):
    return ' '.join([lemmatizer.lemmatize(word) for word in posts.split(' ')])

# Third step - remove stop words
def remove_stop_words(posts, stop_words_set):
    return ' '.join([w for w in posts.split(' ') if w not in stop_words_set])


# Main function for data preprocessing
def process(data):
    _types = []
    posts_for_type = []

    # init lemmatizer
    lemmatizer = WordNetLemmatizer()
    st_words = set(stopwords.words('english'))

    for index, row in data.iterrows():
        posts = row[1]
        posts_temp = replace_non_words(posts)

        posts_temp = lemmatize(lemmatizer, posts_temp)
        posts_temp = remove_stop_words(posts_temp, st_words)
        type_label = label_encoder.transform([row[0]])[0]
        _types.append(type_label)
        posts_for_type.append(posts_temp)

    _types = np.array(_types)
    posts_for_type = np.array(posts_for_type)
    return _types, posts_for_type




