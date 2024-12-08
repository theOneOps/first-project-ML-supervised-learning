import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Visualize:
    def __init__(self, preprocessing):
        self.plt = plt
        self.sns = sns
        self.np = np
        self.preprocessing = preprocessing

    # Interpretation:
    # Coefficients close to 1 or -1 indicate a strong correlation.
    # If two variables are strongly correlated, one can be removed.
    # def showCorrelationMatrix(self):
    #     # Afficher les types de toutes les colonnes pour vérifier celles qui sont des objets
    #     print(self.preprocessing.df.dtypes)
    #
    #     # Sélectionner uniquement les colonnes numériques et exclure les objets
    #     numeric_df = self.preprocessing.df.select_dtypes(include=["number"])
    #
    #     self.plt.figure(figsize=(10, 8))
    #     self.sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    #     self.plt.show()

    def showCorrelationMatrixForSelectedFeatures(self, selected_features):
        # Filtrer le DataFrame pour ne garder que les colonnes sélectionnées
        subset_df = self.preprocessing.df[selected_features]

        # Calculer la matrice de corrélation
        correlation_matrix = subset_df.corr()

        # Afficher la matrice de corrélation sous forme de heatmap
        self.plt.figure(figsize=(10, 8))
        self.sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",  # Limiter à deux décimales
            square=True,  # Assurer que la heatmap est carrée
            cbar_kws={"shrink": 0.8},  # Réduire la taille de la barre de couleur
        )
        self.plt.title("Matrice de corrélation pour les features sélectionnées")
        self.plt.show()

    def showHistogram(self):
        print("Print the histogram of the dataframe \n")
        self.preprocessing.df.hist(figsize=(12, 10))
        self.plt.show()

    def checkDependenciesRelationDist(self, features):
        for feature in features:
            self.sns.kdeplot(
                self.np.log1p(self.preprocessing.df[feature]), label=feature, fill=True
            )
            self.plt.legend()
            plt.title(f"KDE Plot pour le feature {feature}")
            plt.show()  # Affiche chaque plot séparément
