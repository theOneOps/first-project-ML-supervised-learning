import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


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
        
        # Affiche le graphique
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

    # getters
    def getDataFrame(self):
        return self.df
