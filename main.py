import nltk
import sklearn
import re
from data_loading import read

def posts_cleaning(dict):

    for key, values in dict.items():
        for posts in values:
            for post in posts:
                post = re.sub("[.,:?!+]", " ", post)
                post = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', post)
                post = re.sub("[^a-zA-Z]", " ", post)
                post = re.sub(' +', ' ', post).lower()

    return dict

if __name__ == '__main__':

    dict = read()
    #cleaned_dict = posts_cleaning(dict)

    post = "Do I hate it when people find me attractive? Hell no. k http://www.politicalcompass.org/facebook/pcgraphpng.php?ec=3.75&soc=-4.72 I wish people did."
    print(post)

    post = re.sub("[.,:?!+]", " ", post)
    post = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', post)
    post = re.sub("[^a-zA-Z]", " ", post)
    post = re.sub(' +', ' ', post).lower()
    print(post)
