import sys
from read_data import read
from preview_data import preview
from process_data import process
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

if __name__ == '__main__':
    path = sys.argv[1]
    print('Loading data...')
    data = read(path)
    # preview is showing unbalanced dataset
    preview(data)
    print('Processing data...')
    types, posts = process(data)
    print('Processing data finished')
    count_vectorizer = CountVectorizer(analyzer="word",
                                       max_df=0.7,
                                       min_df=0.1,
                                       lowercase=False)
    multinomial_nb = MultinomialNB()

    num_of_splits = 3
    sss = StratifiedShuffleSplit(n_splits=num_of_splits)
    final_score = 0
    print("Train data...")
    for train, test in sss.split(posts, types):
        X_train, X_test, y_train, y_test = posts[train], posts[test], types[train], types[test]
        X_train = count_vectorizer.fit_transform(X_train)
        X_test = count_vectorizer.transform(X_test)

        multinomial_nb.fit(X_train, y_train)
        predictions = multinomial_nb.predict(X_test)
        score = f1_score(y_test, predictions, average='weighted')
        final_score += score
    print('Train data finished')
    final_score = final_score / num_of_splits
    print(f'Final f1-score: {final_score}')
