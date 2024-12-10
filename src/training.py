import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Training:
    def __init__(self, X_Train, X_Test, Y_Train, Y_Test):
        self.results = {}
        self.models = {}
        self.df = {}
        self.X_Test = X_Test
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.Y_Test = Y_Test
        self.errors = {}
        self.initialize_all_models()

        self.classifiers = [
            {
                "model": LogisticRegression(max_iter=200),
                "params": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear", "lbfgs", "newton-cg"],
                },
            },
            {
                "model": DecisionTreeClassifier(),
                "params": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                },
            },
            {
                "model": RandomForestClassifier(),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [10, 20, 30, None],
                    "min_samples_split": [2, 5, 10],
                },
            },
            {
                "model": SVC(),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"],
                },
            },
            {
                "model": AdaBoostClassifier(),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1],
                },
            },
            {
                "model": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                },
            },
            {
                "model": MLPClassifier(max_iter=200),
                "params": {
                    "hidden_layer_sizes": [(50,), (100,), (100, 50), (150, 100, 50)],
                    "activation": ["relu", "tanh", "logistic"],
                    "solver": ["adam", "sgd", "lbfgs"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate": ["constant", "adaptive"],
                },
            },
        ]

    def initialize_all_models(self):
        self.models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(),
            "Polynomial Regression": Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=2)),
                    ("logistic", LogisticRegression()),
                ]
            ),
            "Neural Network": MLPClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Bagging": BaggingClassifier(),
        }

    def train_model(self, model_name):
        """
        here we train a given model and record its performance.
        :param model_name : the name of the model
        """
        model = self.models[model_name]
        model.fit(self.X_Train, self.Y_Train)
        y_predictions = model.predict(self.X_Test)

        # The performance of the models is evaluated
        accuracy = accuracy_score(self.Y_Test, y_predictions)

        # Saving model results to self.results
        self.results[model_name] = {
            "accuracy": accuracy,
        }
        print(f"Accuracy for {model_name}: {accuracy:.4f}")

        # Calculate error (1 - precision) and add to self.errors
        error = 1 - accuracy
        self.errors[model_name] = error

    def train_all_without_cv(self):
        """
        here we train all models and display their performance.
        """
        for model_name in self.models.keys():
            self.train_model(model_name)

    def train_with_cross_validation(self, model_name, cv=5):
        """
        here we train a given model with cross-validation.
        :param model_name : the name of the model
        :param cv : value of the fold
        """
        model = self.models[model_name]
        print(f"Training model: {model_name} with {cv}-fold cross-validation...")

        # cross_validation
        scores = cross_val_score(
            model, self.X_Train, self.Y_Train, cv=cv, scoring="accuracy"
        )
        mean_accuracy = scores.mean()

        # Train on the entire train set for the final assessment
        model.fit(self.X_Train, self.Y_Train)
        y_predictions = model.predict(self.X_Test)

        # Evaluate on the test set
        test_accuracy = accuracy_score(self.Y_Test, y_predictions)

        # Store the results
        self.results[model_name] = {
            "cv_accuracy_mean": mean_accuracy,
            "test_accuracy": test_accuracy,
        }

        error = 1 - test_accuracy
        self.errors[model_name] = error

        print(f"Cross-validated accuracy for {model_name}: {mean_accuracy:.4f}")

    def train_all_models_with_cv(self, cv):
        """
        here we train all models and display their performance.
        :param cv : value of the fold
        """
        for model_name in self.models.keys():
            self.train_with_cross_validation(model_name, cv)

    def plot_confusion_matrix(self, model_name):
        """
        here we display the confusion matrix for a given model.
        :param model_name : the name of the model
        """
        model = self.models[model_name]
        y_predictions = model.predict(self.X_Test)
        ConfusionMatrixDisplay.from_predictions(self.Y_Test, y_predictions)
        plt.title(f"Confusion Matrix for {model_name}")
        plt.show()

    def plot_confusion_matrix_all_models(self):
        """
        here we train all models and display their performance.
        """
        for model_name in self.models.keys():
            self.plot_confusion_matrix(model_name)

    def plot_error_curves(self):
        """
        here we plot the error curves for each model.
        """
        # Plotting error curves
        plt.figure(figsize=(10, 8))
        plt.bar(self.models.keys(), self.errors.values(), color="skyblue")
        plt.xlabel("Models")
        plt.ylabel("Error (1 - precision)")
        plt.title("Model error curves")
        plt.xticks(rotation=45)
        plt.show()

    def plot_learning_curve(self, model, title="Learning Curve"):
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            self.X_Test,
            self.Y_Test,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy",
            n_jobs=-1,
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label="Train Score", color="blue")
        plt.plot(train_sizes, test_mean, label="Validation Score", color="green")
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            color="blue",
            alpha=0.2,
        )
        plt.fill_between(
            train_sizes,
            test_mean - test_std,
            test_mean + test_std,
            color="green",
            alpha=0.2,
        )

        plt.title(title)
        plt.xlabel("Training Size")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_learning_curve_all_models(self):
        """
        here we train all models and display their performance.
        """
        for model_name in self.models.keys():
            self.plot_learning_curve(self.models[model_name])

    def verify_roc(self):
        results = {}
        for name, clf in self.models.items():
            clf.fit(self.X_Train, self.Y_Train)
            y_score = clf.predict_proba(self.X_Test)

            # Compute the ROC AUC score
            roc_auc = roc_auc_score(self.Y_Test, y_score, multi_class="ovo")
            results[name] = roc_auc
            print(f"{name} - ROC AUC Score: {roc_auc}")
        return results

    def addParametersForModels(self):
        self.models = self.models = {
            "Logistic Regression": LogisticRegression(
                C=10, solver="lbfgs", max_iter=1000
            ),
            "Decision Tree": DecisionTreeClassifier(
                criterion="entropy", max_depth=10, min_samples_split=5
            ),
            "Random Forest": RandomForestClassifier(
                max_depth=30, min_samples_split=5, n_estimators=100
            ),
            "SVC": SVC(C=10, gamma="scale", kernel="poly", probability=True),
            "AdaBoost": AdaBoostClassifier(learning_rate=0.01, n_estimators=50),
            "KNN": KNeighborsClassifier(n_neighbors=5, p=1, weights="distance"),
        }

    def train_and_evaluate_grid_search_cv(self):
        """
        Trains the models with hyperparameter's search with gridSearchCV and
        returns the best results.
        """
        # Now we add parameters for all models
        for classifier in self.classifiers:
            model = classifier["model"]
            params = classifier["params"]

            grid_search = GridSearchCV(
                model, param_grid=params, cv=5, n_jobs=-1, scoring="accuracy"
            )
            grid_search.fit(self.X_Train, self.Y_Train)

            print(
                f"Meilleurs paramètres pour {model.__class__.__name__}: {grid_search.best_params_}"
            )

            y_pred = grid_search.predict(self.X_Test)

            evaluation = self.evaluate_model(self.Y_Test, y_pred)
            print(f"Évaluation du modèle {model.__class__.__name__}:")
            self.print_evaluation(evaluation)
            print("-" * 50)

    def print_evaluation(self, evaluation):
        print("Exactitude du modèle : {:.2f}%".format(evaluation["Accuracy"] * 100))
        print("Précision du modèle : {:.2f}%".format(evaluation["Precision"] * 100))
        print("Recall du modèle : {:.2f}%".format(evaluation["Recall"] * 100))
        print("F1_score du modèle : {:.2f}%".format(evaluation["F1_Score"] * 100))

    def evaluate_model(self, y_true, y_pred):
        evaluation = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1_Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        return evaluation
