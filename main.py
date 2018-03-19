import sys
from read_data import read
from preview_data import preview
from process_data import process
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit

# TODO TfIdf
# Bigram
# Negacija
# Izdvojiti frekvente bigrame
# POS tagovi (naglasene reci)
# Leksikon SentiWordNet, Vader
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
                                       max_df=0.5,
                                       min_df=0.1,
                                       lowercase=False)

    tf_idf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.1, lowercase=False)

    multinomial_nb = MultinomialNB()
    svc = SVC(decision_function_shape='ovo')
    rnf = RandomForestClassifier(n_estimators=10, max_depth=2)

    num_of_splits = 3
    kf = KFold(n_splits=num_of_splits, shuffle=True)

    final_score_nb = 0
    final_score_svc = 0
    final_score_rnf = 0

    print("Train data...")
    for train, test in kf.split(posts, types):
        X_train, X_test, y_train, y_test = posts[train], posts[test], types[train], types[test]
        X_train = count_vectorizer.fit_transform(X_train)
        X_test = count_vectorizer.transform(X_test)

       # print(tf_idf_vectorizer.get_feature_names())

        # Naive Bayes
        # multinomial_nb.fit(X_train, y_train)
        # nb_predictions = multinomial_nb.predict(X_test)
        # score_nb = f1_score(y_test, nb_predictions, average='weighted')
        # final_score_nb += score_nb
        #
        # # SVM
        svc.fit(X_train, y_train)
        svc_predictions = svc.predict(X_test)
        score_svc = f1_score(y_test, svc_predictions, average='weighted')
        final_score_svc += score_svc

        # Random Forest
        # rnf.fit(X_train, y_train)
        # rnf_predictions = rnf.predict(X_test)
        # score_rnf = f1_score(y_test, rnf_predictions, average='weighted')
        # final_score_rnf += score_rnf

    print('Train data finished')
    print('***Scores***')
    final_score_nb = final_score_nb / num_of_splits
    print(f'Final f1-score for Multinomial Bayes: {final_score_nb}')
    final_score_svc = final_score_svc / num_of_splits
    print(f'Final f1-score for SVM: {final_score_svc}')
    final_score_rnf = final_score_rnf / num_of_splits
    print(f'Final f1-score for Random Forest: {final_score_rnf}')
