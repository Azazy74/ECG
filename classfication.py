from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from joblib import dump

def train_and_evaluate_models(features, classes):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.2, shuffle=True)

    # Logistic Regression
    param_grid_logistic = {'C': [0.1, 1 ,100], 'penalty': ['l1', 'l2']}
    grid_search_logistic = GridSearchCV(LogisticRegression(solver="liblinear"), param_grid_logistic)
    grid_search_logistic.fit(X_train, y_train)
    best_logistic = grid_search_logistic.best_estimator_

    # SVM
    param_grid_svm = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear']}
    grid_search_svm = GridSearchCV(SVC(), param_grid_svm)
    grid_search_svm.fit(X_train, y_train)
    best_svm = grid_search_svm.best_estimator_

    # LDA
    LDA_classifier = LinearDiscriminantAnalysis()
    LDA_classifier.fit(X_train, y_train)

    # Calculate accuracies
    y_pred_logistic = best_logistic.predict(X_test)
    acc_logistic = metrics.accuracy_score(y_test, y_pred_logistic) * 100

    y_pred_svm = best_svm.predict(X_test)
    acc_svm = metrics.accuracy_score(y_test, y_pred_svm) * 100

    y_pred_lda = LDA_classifier.predict(X_test)
    acc_lda = metrics.accuracy_score(y_test, y_pred_lda) * 100

    # Print accuracies
    print("Accuracy of Logistic Regression = ", acc_logistic)
    print("Accuracy of SVM = ", acc_svm)
    print("Accuracy of LDA = ", acc_lda)

    # Print the best model for each classifier
    print("Best Logistic Regression Model:", best_logistic)
    print("Best SVM Model:", best_svm)

    # Save models
    dump(grid_search_svm, 'Models/SVM.joblib')
    dump(grid_search_logistic, 'Models/LogisticRegression.joblib')
    dump(LDA_classifier, 'Models/LDA.joblib')

