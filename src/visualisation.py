import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


class Visualize:
    def __init__(self, preprocessing):
        self.plt = plt
        self.sns = sns
        self.np = np
        self.preprocessing = preprocessing

    def showCorrelationMatrixForSelectedFeatures(self, selected_features):
        # We filter the DataFrame to keep only the selected columns
        subset_df = self.preprocessing.df[selected_features]

        # we calculate the correlation matrix
        correlation_matrix = subset_df.corr()

        # we display the correlation matrix as a heatmap
        self.plt.figure(figsize=(10, 8))
        self.sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",  # we limit to two decimal places
            square=True,  # Ensure the heatmap is square
            cbar_kws={"shrink": 0.8},  # we reduce the size of the color bar
        )
        self.plt.title("Matrice de corrélation pour les features sélectionnées")
        self.plt.show()

    def plot_histograms(self, columns=None, bins=30):
        """
        Displays histograms for the specified columns with a particular layout:
        - First row: 4 figures
        - Subsequent rows: 4 figures each

        :param columns: List of columns to plot. If None, all numeric columns will be plotted.
        :param bins: Number of bins for histograms.
        """
        if columns is None:
            columns = self.preprocessing.df.select_dtypes(include="number").columns

        # Déterminer le nombre total de colonnes et de lignes nécessaires
        first_row_count = 4
        other_row_count = 4
        remaining = len(columns) - first_row_count
        n_rows = 1 + math.ceil(
            max(0, remaining) / other_row_count
        )  # 1 pour la première ligne

        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4 * n_rows))
        axes = (
            axes.flatten()
        )  # Transforme les axes en une liste pour faciliter l'indexation

        for i, col in enumerate(columns):
            sns.histplot(
                self.preprocessing.df[col],
                bins=bins,
                kde=True,
                ax=axes[i],
                color="rosybrown",
            )
            axes[i].set_title(f"Histogramme de {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Fréquence")

        for j in range(len(columns), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_outliers_percentage(self, df):
        """
        Calcule le pourcentage d'outliers pour chaque variable numérique dans le DataFrame et les affiche sous forme de graphique à barres.
        """
        # Sélectionne un sous-ensemble de colonnes numériques du DataFrame
        df_num = df.drop(columns=["class"])
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
            outliers = df_num[
                (df_num[variable] < lower_bound) | (df_num[variable] > upper_bound)
            ]

            # Calcule le pourcentage d'outliers pour la variable
            percentage = (len(outliers) / len(df_num)) * 100
            percentage_outliers.append(percentage)

            # Ajoute les indices des outliers à la liste
            outliers_indices.extend(outliers.index)

        # Crée un graphique à barres pour afficher le pourcentage d'outliers par variable
        plt.figure(figsize=(9, 4))
        bars = plt.bar(variables, percentage_outliers, color="slategrey")

        plt.xlabel("Variable")
        plt.ylabel("Pourcentage d'Outliers")
        plt.title("Pourcentage d'Outliers par Variable")

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
        df_numeric = self.preprocessing.df.drop(columns=excluded_columns)
        correlation_matrix = df_numeric.corr(method="pearson")

        # Crée un masque pour afficher uniquement une partie supérieure de la matrice
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".3f", mask=mask)

        plt.title("Matrice de Corrélation de Pearson")
        plt.show()

    def checkDependenciesRelationDist(self, features):
        for feature in features:
            self.sns.kdeplot(
                self.np.log1p(self.preprocessing.df[feature]), label=feature, fill=True
            )
            self.plt.legend()
            plt.title(f"KDE Plot pour le feature {feature}")
            plt.show()  # Affiche chaque plot séparément

    def plot_boxplots(self, df, columns=None):
        """
        Affiche des boxplots pour les colonnes spécifiées avec une disposition particulière :
        - Première ligne : 4 figures
        - Lignes suivantes : 4 figures chacune

        :param columns: Liste des colonnes à tracer. Si None, toutes les colonnes numériques seront tracées.
        """
        if columns is None:
            columns = df.select_dtypes(include="number").columns

        # Déterminer le nombre total de colonnes et de lignes nécessaires
        first_row_count = 4
        other_row_count = 4
        remaining = len(columns) - first_row_count
        n_rows = 1 + math.ceil(
            max(0, remaining) / other_row_count
        )  # 1 pour la première ligne

        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4 * n_rows))
        axes = (
            axes.flatten()
        )  # Transforme les axes en une liste pour faciliter l'indexation

        for i, col in enumerate(columns):
            sns.boxplot(
                data=df,
                y=col,
                ax=axes[i],
                color="white",
                width=0.2,
                flierprops=dict(markerfacecolor="red", marker="D"),
            )
            axes[i].set_title(f"Boxplot de {col}")
            axes[i].set_ylabel(col)

        # Supprimer les axes inutilisés
        for j in range(len(columns), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
