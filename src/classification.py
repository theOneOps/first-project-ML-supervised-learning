from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures

import numpy as np



class Classification:
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Constructeur pour l'entraînement des modèles et validation.
        :param X_train: Données d'entraînement (features).
        :param X_test: Données de test (features).
        :param y_train: Labels d'entraînement.
        :param y_test: Labels de test.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = [

            {
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'min_samples_split': [2, 4, 6],
                    'criterion': ["gini", "entropy"],
                    'max_depth': [None, 10, 20, 30]
                }
            },
            {
                'model': DecisionTreeClassifier(),
                'params': {
                    'criterion': ["gini", "entropy"],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 4, 6]
                }
            },
            {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            {
                'model': LogisticRegression(),
                'params': {
                    'C': np.logspace(-3, 2, 6),
                    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'saga'],
                    'penalty': ['l1', 'l2', 'elasticnet', 'none']
                }
            },
            {
                'model': BaggingClassifier(),
                'params': {
                    'n_estimators': [10, 20, 50],
                    'max_samples': [0.5, 0.7, 1.0],
                    'bootstrap': [True, False],
                    'bootstrap_features': [True, False]
                }
            },
            {
                'model': MLPClassifier(),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'solver': ['lbfgs', 'sgd', 'adam'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'invscaling', 'adaptive']
                }
            },
            {
                'model': AdaBoostClassifier(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1, 10]
                }
            }
        ]

    def train_and_evaluate(self, n_iter=10):
        """
        Entraîne les modèles et retourne les meilleurs résultats.
        :param n_iter: Nombre d'itérations pour RandomizedSearchCV.
        """
        best_results = {}
        for model_info in self.models:
            model = model_info['model']
            params = model_info['params']
            print(model)
            # RandomizedSearchCV pour chaque modèle
            random_search = RandomizedSearchCV(model, params, cv=5, scoring='accuracy', n_iter=n_iter, n_jobs=-1,random_state=42)
            random_search.fit(self.X_train, self.y_train)

            # Meilleur modèle et évaluation
            best_model = random_search.best_estimator_
            best_score = random_search.best_score_
            y_pred = best_model.predict(self.X_test)

            report = classification_report(self.y_test, y_pred, output_dict=True)
            best_results[model.__class__.__name__] = {
                'best_model': best_model,
                'best_score': best_score,
                'predictions': y_pred,
                'classification_report': report
            }

        return best_results
