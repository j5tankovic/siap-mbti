import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet as wn
from sklearn.preprocessing import LabelEncoder
from nltk.tag import RegexpTagger, UnigramTagger, BigramTagger

types = ['INFP', 'ESFJ', 'ESFP', 'ISFJ', 'ISFP', 'ESTJ', 'ESTP', 'ISTJ', 'ISTP', 'INTJ', 'INFJ', 'ENTJ', 'ENTP', 'ENFJ',
         'INTP', 'ENFP']

pos_types = {'NN': wn.NOUN,
             'NNS': wn.NOUN,
             'JJ': wn.ADJ,
             'VBD': wn.VERB,
             'VBG': wn.VERB,
             'RB': wn.ADV}

label_encoder = LabelEncoder().fit(types)


# First step - remove all URLs from posts
def replace_non_words(posts):
    posts = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', posts)
    posts = re.sub(r'(?i)not|never|didn\'t|did not|don\'t|do not|no', 'NEG', posts)
    posts = re.sub(r'(?i)(:sad:|:crying:|</3|-.-|-_-)|(>?:\'?-?(\(+|S+|\\+))', 'NEG', posts)
    posts = re.sub(
        r'(?i)(\^_\^|\\o/|<3|:happy:|:proud:|:kiss:|crazy:|lol|rofl")|(((O|>)?(:|;)\'?-?|[xX]|=|B-?)(\)+|D+|\*+))',
        'POS', posts)
    posts = re.sub(r'(?i)(>.>|<.<|omg)|(:-?(O+|/+|P+))', 'NEU', posts)
    posts = re.sub("[.,:?!+]", " ", posts)
    posts = re.sub("[^a-zA-Z]", " ", posts)
    posts = re.sub(' +', ' ', posts).lower()
    return posts


# Second step - use lemmatization
def lemmatize(lemmatizer, posts):
    return ' '.join([lemmatizer.lemmatize(word[0], pos_types.get(word[1], wn.NOUN))
                     for word in posts])


# Second step - use stemmer
def stemming(stemmer, posts):
    return ' '.join([stemmer.stem(word) for word in posts.split(' ')])


# Third step - remove stop words
def remove_stop_words(posts, stop_words_set):
    return ' '.join([w for w in posts.split(' ') if w not in stop_words_set])


def get_pos_tagger():
    from nltk.corpus import treebank
    regexp_tagger = RegexpTagger(
        [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
         (r'(The|the|A|a|An|an)$', 'AT'),  # articles
         (r'.*able$', 'JJ'),  # adjectives
         (r'.*ness$', 'NN'),  # nouns formed from adjectives
         (r'.*ly$', 'RB'),  # adverbs
         (r'.*s$', 'NNS'),  # plural nouns
         (r'.*ing$', 'VBG'),  # gerunds
         (r'.*ed$', 'VBD'),  # past tense verbs
         (r'.*', 'NN')  # nouns (default)
         ])
    brown_train = treebank.tagged_sents()
    unigram_tagger = UnigramTagger(brown_train, backoff=regexp_tagger)
    bigram_tagger = BigramTagger(brown_train, backoff=unigram_tagger)

    main_tagger = RegexpTagger(
        [(r'(A|a|An|an)$', 'ex_quant'),
         (r'(Every|every|All|all)$', 'univ_quant')
         ], backoff=bigram_tagger)

    return main_tagger


# Main function for data preprocessing
def process(data):
    _types = []
    posts_for_type = []

    # init lemmatizer
    lemmatizer = WordNetLemmatizer()
    st_words = set(stopwords.words('english'))
    pos_tagger = get_pos_tagger()

    for index, row in data.iterrows():
        posts = row[1]
        posts_temp = replace_non_words(posts)
        posts_temp = remove_stop_words(posts_temp, st_words)
        posts_temp = pos_tagger.tag(posts_temp.split())
        posts_temp = lemmatize(lemmatizer, posts_temp)
        type_label = label_encoder.transform([row[0]])[0]
        _types.append(type_label)
        posts_for_type.append(posts_temp)

    _types = np.array(_types)
    posts_for_type = np.array(posts_for_type)
    return _types, posts_for_type
