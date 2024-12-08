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

    def plot_outliers_percentage(self):
        """
        Calcule le pourcentage d'outliers pour chaque variable numérique dans le DataFrame et les affiche sous forme de graphique à barres.
        """
        # Sélectionne un sous-ensemble de colonnes numériques du DataFrame
        df_num = self.df.drop(columns=['class'])
        variables = df_num.columns  # Liste des variables à analyser
        
        # Initialise les listes pour stocker les pourcentages d'outliers et les indices des outliers
        percentage_outliers = []
        outliers_indices = []

        # Parcourt chaque variable pour calculer les outliers
        for variable in variables:
            # Calcule le premier et le troisième quartile
            q1 = df_num[variable].quantile(0.25)
            q3 = df_num[variable].quantile(0.75)
            
            # Calcule l'écart interquartile (IQR)
            iqr = q3 - q1
            
            # Détermine les bornes inférieure et supérieure pour détecter les outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Identifie les valeurs considérées comme des outliers
            outliers = df_num[(df_num[variable] < lower_bound) | (df_num[variable] > upper_bound)]

            # Calcule le pourcentage d'outliers pour la variable
            percentage = (len(outliers) / len(df_num)) * 100
            percentage_outliers.append(percentage)
            
            # Ajoute les indices des outliers à la liste
            outliers_indices.extend(outliers.index)

        # Crée un graphique à barres pour afficher le pourcentage d'outliers par variable
        plt.figure(figsize=(9, 4))
        bars = plt.bar(variables, percentage_outliers, color='slategrey')

        plt.xlabel('Variable')
        plt.ylabel('Pourcentage d\'Outliers')
        plt.title('Pourcentage d\'Outliers par Variable')

        plt.xticks(rotation=90)
        
        plt.show()
    
    def plot_correlation_matrix(self, excluded_columns=[]):
        """
        Trace une matrice de corrélation de Pearson pour un DataFrame donné,
        en excluant les colonnes spécifiées.

        Args:
            df (DataFrame): Le DataFrame contenant les données à analyser.
            excluded_columns (list, optional): Liste des colonnes à exclure de l'analyse de corrélation. Par défaut, aucune.

        Returns:
            None
        """
        # Exclut les colonnes spécifiées pour créer un sous-ensemble de données numériques
        df_numeric = self.df.drop(columns=excluded_columns)
        correlation_matrix = df_numeric.corr(method='pearson')

        # Crée un masque pour afficher uniquement une partie supérieure de la matrice
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".3f", mask=mask)

        plt.title('Matrice de Corrélation de Pearson')
        plt.show()
    def checkDependenciesRelationDist(self, features):
        for feature in features:
            self.sns.kdeplot(
                self.np.log1p(self.preprocessing.df[feature]), label=feature, fill=True
            )
            self.plt.legend()
            plt.title(f"KDE Plot pour le feature {feature}")
            plt.show()  # Affiche chaque plot séparément
