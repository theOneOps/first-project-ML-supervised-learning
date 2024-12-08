from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score



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

        ## Les modèles après optimisation des hyperparamètres (avec GridSearchCV)
        self.models = {
            "Logistic Regression": LogisticRegression(C=10, solver='lbfgs', max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5),
            "Random Forest": RandomForestClassifier(max_depth=30, min_samples_split=5, n_estimators=100),
            "SVC": SVC(C=10, gamma='scale', kernel='poly', probability=True),
            "AdaBoost": AdaBoostClassifier(learning_rate=0.01, n_estimators=50),
            "KNN": KNeighborsClassifier(n_neighbors=5, p=1, weights='distance')
        }

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
    def print_results(self):
        """
        Print the comparison of results for all models.
        """
        print("\nComparaison des résultats des modèles :")
        for model_name, result in self.best_results.items():
            print(f"\n{model_name} :")
            print(f"  Best Score: {result['best_score']}")
            print(f"\n{model_name} Classification Report:")

            # Print the header for the table
            print(f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
            print("-" * 50)

            # Print the rows for each class
            for label, metrics in result['classification_report'].items():
                if label != 'accuracy' and label != 'macro avg' and label != 'weighted avg':
                    precision = metrics['precision']
                    recall = metrics['recall']
                    f1_score = metrics['f1-score']
                    print(f"{label:<10}{precision:<12.4f}{recall:<12.4f}{f1_score:<12.4f}")

            # Print averages
            print("-" * 50)
            print(f"{'Macro avg':<10}{result['classification_report']['macro avg']['precision']:<12.4f}{result['classification_report']['macro avg']['recall']:<12.4f}{result['classification_report']['macro avg']['f1-score']:<12.4f}")
            print(f"{'Weighted avg':<10}{result['classification_report']['weighted avg']['precision']:<12.4f}{result['classification_report']['weighted avg']['recall']:<12.4f}{result['classification_report']['weighted avg']['f1-score']:<12.4f}")
    def verify_roc(self):
        results = {}
        for name, clf in self.models.items():
            clf.fit(self.X_train, self.y_train)
            y_score = clf.predict_proba(self.X_test)
            
            # Compute the ROC AUC score
            roc_auc = roc_auc_score(self.y_test, y_score, multi_class='ovo')
            results[name] = roc_auc
            print(f"{name} - ROC AUC Score: {roc_auc}")
        return results