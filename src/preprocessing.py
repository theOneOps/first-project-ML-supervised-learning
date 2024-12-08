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
    
        # Définit les étiquettes et couleurs pour le graphique
        colors = ('rosybrown', 'lightgray','slategrey')  # Couleurs pour les sections du diagramme
    
        # Configure la taille de la figure
        plt.figure(figsize=(3, 2.5))
    
        # Trace un diagramme circulaire
        plt.pie(
            percentages,
            labels=counts.index,
            colors=colors,
            shadow=True,
            autopct='%1.1f%%',
            )
    
        # Ajoute un titre au graphique
        plt.title('Répartition de la variable Class')
    
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
