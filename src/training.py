import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, learning_curve, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
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
        self.models = [
            {
                "model": RandomForestClassifier(),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "min_samples_split": [2, 4, 6],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                },
            },
            {
                "model": DecisionTreeClassifier(),
                "params": {
                    "criterion": ["gini", "entropy"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 4, 6],
                },
            },
            {
                "model": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [3, 5, 7, 10],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"],
                },
            },
            {
                "model": BaggingClassifier(),
                "params": {
                    "n_estimators": [10, 20, 50],
                    "max_samples": [0.5, 0.7, 1.0],
                    "bootstrap": [True, False],
                    "bootstrap_features": [True, False],
                },
            },
            {
                "model": MLPClassifier(),
                "params": {
                    "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                    "activation": ["identity", "logistic", "tanh", "relu"],
                    "solver": ["lbfgs", "sgd", "adam"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate": ["constant", "invscaling", "adaptive"],
                },
            },
            {
                "model": AdaBoostClassifier(),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1, 10],
                },
            },
            {
                "model": Pipeline(
                    [
                        ("poly", PolynomialFeatures()),  # Polynomial transformation
                        ("logistic", LogisticRegression()),  # Logistic
                        # Regression
                    ]
                ),
                "params": {
                    "poly__degree": [2, 3, 4],  # Degree of polynomials
                    "logistic__C": np.logspace(-3, 2, 6),  # Regularization
                    "logistic__solver": ["liblinear", "newton-cg", "lbfgs", "saga"],
                    "logistic__penalty": ["l1", "l2", "none"],  # Regularization
                },
            },
        ]

    def train_and_evaluate_grid_search_cv(self, n_iter=10):
        """
        Trains the models and returns the best results.
        :param n_iter: Number of iterations for RandomizedSearchCV.
        """
        # Now we add parameters for all models
        self.addParametersForModels()

        best_results = {}
        for model_info in self.models:
            model = model_info["model"]
            params = model_info["params"]
            # print(model)

            # RandomizedSearchCV for each model
            random_search = RandomizedSearchCV(
                model,
                params,
                cv=5,
                scoring="accuracy",
                n_iter=n_iter,
                n_jobs=-1,
                random_state=42,
            )
            random_search.fit(self.X_Train, self.Y_Train)

            # Best model and evaluation
            best_model = random_search.best_estimator_
            best_score = random_search.best_score_
            y_pred = best_model.predict(self.X_Test)

            report = classification_report(self.Y_Test, y_pred, output_dict=True)
            best_results[model.__class__.__name__] = {
                "best_model": best_model,
                "best_score": best_score,
                "predictions": y_pred,
                "classification_report": report,
            }

        return best_results
