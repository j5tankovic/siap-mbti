import nltk
import sklearn
import re
from data_loading import read, read_new

from nltk.stem import WordNetLemmatizer;
from nltk.corpus import stopwords
from nltk import word_tokenize


def posts_cleaning(dict):

    for key, values in dict.items():
        for posts in values:
            for post in posts:
                post = re.sub("[.,:?!+]", " ", post)
                post = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', post)
                post = re.sub("[^a-zA-Z]", " ", post)
                post = re.sub(' +', ' ', post).lower()

    return dict


def get_processed_words_for_type(sentence, lemmatizer, stop_words_corpus):
    words = word_tokenize(sentence)
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [w for w in words if w not in stop_words_corpus]
    return words


def get_bag_of_words(words):
    return dict([(w, True) for w in words])


if __name__ == '__main__':

    data = read_new()
    #cleaned_dict = posts_cleaning(data)

    post = "Do I hate it when people find me attractive? Hell no. k http://www.politicalcompass.org/facebook/pcgraphpng.php?ec=3.75&soc=-4.72 I wish people did."
    print(post)

    post = re.sub("[.,:?!+]", " ", post)
    post = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', post)
    post = re.sub("[^a-zA-Z]", " ", post)
    post = re.sub(' +', ' ', post).lower()
    print(post)

    lemmatizer = WordNetLemmatizer()
    eng_stopwords = set(stopwords.words('english'))

    words_for_type = {}
    for key, value in data.items():
        for sentence in value:
            tokenized = get_processed_words_for_type(sentence, lemmatizer, eng_stopwords)
            bag = get_bag_of_words(tokenized)
            if key in words_for_type:
                words_for_type[key].update(bag)
            else:
                words_for_type[key] = bag


