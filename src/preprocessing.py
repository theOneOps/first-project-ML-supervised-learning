import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE


class Preprocessing:
    def __init__(self, dataFilePath):
        self.label_encoders = {}
        self.duplicatedSum = None
        self.nullSum = None
        self.df = None
        self.dataFilePath = dataFilePath

    def read_file(self):
        self.df = pd.read_csv(self.dataFilePath)
        return self.df

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
            return array.dtype.name == "object"

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
        counts = self.df["class"].value_counts()

        # Calcule les pourcentages pour chaque catégorie
        percentages = (counts / counts.sum()) * 100

        colors = ("rosybrown", "lightgray", "slategrey")

        plt.figure(figsize=(3, 2.5))

        plt.pie(
            percentages,
            labels=counts.index,
            colors=colors,
            shadow=True,
            autopct="%1.1f%%",
        )

        plt.title("Répartition de la variable Class")

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

    def identify_and_remove_strongly_correlated_pairs(self, threshold=0.7):
        """
        Identifie et supprime les colonnes fortement corrélées dans un DataFrame.

        Args:
            df (DataFrame): Le DataFrame contenant les données à analyser.
            threshold (float): Seuil de corrélation pour déterminer les paires fortement corrélées.

        Returns:
            DataFrame: Le DataFrame après suppression des colonnes fortement corrélées.
        """
        df_numeric = self.df.drop(columns=["class"])

        correlation_matrix = df_numeric.corr()

        strongly_correlated_pairs = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                correlation_value = correlation_matrix.iloc[i, j]
                if abs(correlation_value) > threshold:
                    strongly_correlated_pairs.append(
                        (
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            correlation_value,
                        )
                    )

        # Affiche les paires fortement corrélées
        if strongly_correlated_pairs:
            print("Les paires de variables fortement corrélées sont :")
            for pair in strongly_correlated_pairs:
                print(f"{pair[0]} et {pair[1]} avec une corrélation de {pair[2]:.2f}")
        else:
            print("Aucune paire de variables fortement corrélée n'a été trouvée.")

        # Extraction des colonnes à supprimer
        correlated_columns = {pair[0] for pair in strongly_correlated_pairs}

        # Suppression des colonnes fortement corrélées du DataFrame
        df_cleaned = self.df.drop(
            columns=correlated_columns.intersection(self.df.columns), axis=1
        )

        return df_cleaned

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

    def split_data(self, df, target):
        """
        Divise les données en ensembles d'entraînement et de test.

        Args:
            df (DataFrame): Le DataFrame contenant toutes les caractéristiques.
            target (str): Le nom de la colonne cible.

        Returns:
            tuple: Les ensembles d'entraînement et de test pour les caractéristiques (X_train, X_test)
                et pour la cible (y_train, y_test).
        """
        features = df.drop(columns=[target])
        targets = df[target]

        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.20, random_state=42, stratify=targets
        )

        return X_train, X_test, y_train, y_test

    # getters
    def getDataFrame(self):
        return self.df

    def resample_data(self, X, y, strategy='auto'):
        """
        Rééquilibre les classes dans les données en utilisant SMOTE.
        
        Args:
            X (DataFrame): Caractéristiques des données.
            y (Series): Cible.
            strategy (str, dict, optional): Stratégie de rééchantillonnage, peut être 'auto' ou un dictionnaire spécifiant 
                le nombre d'échantillons souhaité pour chaque classe. Par défaut 'auto'.
                
        Returns:
            tuple: Les nouvelles caractéristiques et cibles après le rééchantillonnage (X_resampled, y_resampled).
        """
        # Initialisation de SMOTE pour le sur-échantillonnage
        smote = SMOTE(sampling_strategy=strategy, random_state=42)
        
        # Application de SMOTE
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Affichage des nouvelles proportions de classes
        print("Distribution des classes après rééchantillonnage :")
        print(pd.Series(y_resampled).value_counts())
        
        return X_resampled, y_resampled
