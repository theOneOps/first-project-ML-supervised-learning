from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score
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
