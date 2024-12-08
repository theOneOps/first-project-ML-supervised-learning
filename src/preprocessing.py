import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Preprocessing:
    def __init__(self, dataFilePath):
        self.label_encoders = {}
        self.duplicatedSum = None
        self.nullSum = None
        self.df = None
        self.dataFilePath = dataFilePath

    def read_file(self):
        self.df = pd.read_csv(self.dataFilePath)

    def printFileData(self):
        print(self.df.head())

    def describeData(self):
        # Display the data
        print(self.df.info())
        # A brief description of the data
        print(self.df.describe())
    
    def extract_categorical_columns(self):
        """
        Identifie les colonnes catégorielles dans le DataFrame, affiche leurs valeurs uniques avec leurs fréquences 
        et crée un DataFrame contenant uniquement ces colonnes.
        Returns:
        DataFrame: Un DataFrame contenant uniquement les colonnes catégorielles.
        """
        def is_categorical(array):
            """
            Vérifie si une colonne est catégorielle en se basant sur son type de données.
            Args:
            array (Series): Une colonne du DataFrame.
            Returns:
            bool: True si la colonne est de type 'object', sinon False.
            """
            return array.dtype.name == 'object'
        
        categorical_columns = []
        
        # Parcourt toutes les colonnes du DataFrame                                        
        for col in self.df.columns:
            # Vérifie si la colonne est catégorielle
            if is_categorical(self.df[col]):
                categorical_columns.append(col)
                # Affiche le nom de la colonne et la répartition des valeurs uniques
                print(col)
                print(self.df[col].value_counts())
                print("\n")
        df_categorical = self.df[categorical_columns].copy()

        return df_categorical

    def plot_target_distribution(self):
        """
        Visualise la répartition de la variable cible 'satisfied' sous forme de diagramme circulaire.
    
        La méthode affiche un graphique avec les proportions de chaque catégorie de la variable cible.
        """
        # Compte les occurrences de chaque catégorie dans la variable 'satisfied'
        counts = self.df['class'].value_counts()
    
        # Calcule les pourcentages pour chaque catégorie
        percentages = (counts / counts.sum()) * 100
    
        colors = ('rosybrown', 'lightgray','slategrey')  
    
        plt.figure(figsize=(3, 2.5))
    
        plt.pie(
            percentages,
            labels=counts.index,
            colors=colors,
            shadow=True,
            autopct='%1.1f%%',
            )
    
        plt.title('Répartition de la variable Class')
    
        plt.show()

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
    def encodingFeatures(self, features):
        for col in features:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[
                col
            ] = le  # Conserver les encoders si besoin de retrouver les
            # valeurs d'origine

    def checkNullValues(self):
        self.nullSum = self.df.isnull().sum()
        self.duplicatedSum = self.df.duplicated().sum()
        print(f"Quantity of null values: {self.nullSum} \n")
        print(f"Quantity of duplicated values: {self.duplicatedSum} \n")

    def removeAndCleanDataframe(self):
        if self.duplicatedSum.any():
            self.df.drop_duplicates(inplace=True)
        if self.nullSum.any():
            self.df.dropna(inplace=True)

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

    def normalize_data(self, df, columns):
        """
        Applique une normalisation aux colonnes spécifiées en utilisant MinMaxScaler.

        Args:
            df (DataFrame): Le DataFrame contenant les données à normaliser.
            columns (list): Liste des colonnes à normaliser.

        Returns:
            DataFrame: Un nouveau DataFrame avec les colonnes normalisées.
        """
        scaler = MinMaxScaler()
        
        normalized_data = scaler.fit_transform(df)
        
        df_normalized = pd.DataFrame(normalized_data, columns=columns)
        
        return df_normalized

    # getters
    def getDataFrame(self):
        return self.df
