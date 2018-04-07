import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from preview_data import preview, preview_negations
from process_data import process
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    print('Loading data...')
    data = pd.read_csv("data/mbti.csv")
    # preview is showing unbalanced dataset
    preview(data)
    #preview_negations(data)
    # print('Processing data...')
    # types, posts = process(data)
    # print('Processing data finished')
    # count_vectorizer = CountVectorizer(analyzer="word",
    #                                    max_df=0.7,
    #                                    min_df=0.1,
    #                                    lowercase=True)
    #
    # tf_idf_vectorizer = TfidfVectorizer(lowercase=True,
    #                                     min_df=0.1,
    #                                     max_df=0.7,
    #                                     ngram_range=(1, 2))
    #
    # multinomial_nb = MultinomialNB()
    # svc = SVC(decision_function_shape='ovo')
    # rnf = RandomForestClassifier(n_estimators=10, max_depth=2)
    #
    # num_of_splits = 3
    # kf = KFold(n_splits=num_of_splits, shuffle=True)
    #
    # final_score_ab = 0
    # final_score_logregg = 0
    # final_score_knn = 0
    # final_score_nb = 0
    # final_score_svc = 0
    # final_score_rnf = 0
    #
    # X_train, X_test, y_train, y_test = train_test_split(posts, types, test_size=0.33, random_state=42)
    #
    # print("Train data...")
    # pipeline = Pipeline([
    #     ('tf_vec', tf_idf_vectorizer),
    #     ('clf', LogisticRegression())
    # ])
    #
    # pipeline.fit(X_train, y_train)
    #
    # y_pred = pipeline.predict(X_test)
    # print(f1_score(y_test, y_pred, average='micro'))
    #
    # # for train, test in kf.split(posts, types):
    # #     X_train, X_test, y_train, y_test = posts[train], posts[test], types[train], types[test]
    # #     X_train = tf_idf_vectorizer.fit_transform(X_train)
    # #     X_test = tf_idf_vectorizer.transform(X_test)
    # #
    # #     # X_train = tf_idf_vectorizer.fit_transform(X_train)
    # #     # X_test = tf_idf_vectorizer.transform(X_test)
    # #
    # #     # X_train = count_vectorizer.fit_transform(X_train)
    # #     # X_test = count_vectorizer.transform(X_test)
    # #     # X_train = tfizer.fit_transform(X_train).toarray()
    # #     # X_test = tfizer.transform(X_test).toarray()
    # #
    # #     # Logistic Regression
    # #     logregg = LogisticRegression()
    # #     logregg.fit(X_train, y_train)
    # #     logregg_predictions = logregg.predict(X_test)
    # #     score_logregg = f1_score(y_test, logregg_predictions, average='weighted')
    # #     final_score_logregg += score_logregg
    # #
    # #     # # KNN
    # #     # knn = KNeighborsClassifier(n_neighbors=3)
    # #     # knn.fit(X_train, y_train)
    # #     # knn_predictions = knn.predict(X_test)
    # #     # score_knn = f1_score(y_test, knn_predictions, average='weighted')
    # #     # final_score_knn += score_knn
    # #     #
    # #     # # Naive Bayes
    # #     # multinomial_nb.fit(X_train, y_train)
    # #     # nb_predictions = multinomial_nb.predict(X_test)
    # #     # score_nb = f1_score(y_test, nb_predictions, average='weighted')
    # #     # final_score_nb += score_nb
    # #     #
    # #     # # SVM
    # #     # svc.fit(X_train, y_train)
    # #     # svc_predictions = svc.predict(X_test)
    # #     # score_svc = f1_score(y_test, svc_predictions, average='weighted')
    # #     # final_score_svc += score_svc
    # #     #
    # #     # # Random Forest
    # #     # rnf.fit(X_train, y_train)
    # #     # rnf_predictions = rnf.predict(X_test)
    # #     # score_rnf = f1_score(y_test, rnf_predictions, average='weighted')
    # #     # final_score_rnf += score_rnf
    #
    # # print('Train data finished')
    # # print('***Scores***')
    # # final_score_ab = final_score_ab / num_of_splits
    # # print(f'Final f1-score for AdaBoostClassifier: {final_score_ab}')
    # # final_score_logregg = final_score_logregg / num_of_splits
    # # print(f'Final f1-score for Logistic Regression: {final_score_logregg}')
    # # final_score_knn = final_score_knn / num_of_splits
    # # print(f'Final f1-score for KNN: {final_score_knn}')
    # # final_score_nb = final_score_nb / num_of_splits
    # # print(f'Final f1-score for Multinomial Bayes: {final_score_nb}')
    # # final_score_svc = final_score_svc / num_of_splits
    # # print(f'Final f1-score for SVM: {final_score_svc}')
    # # final_score_rnf = final_score_rnf / num_of_splits
    # # print(f'Final f1-score for Random Forest: {final_score_rnf}')