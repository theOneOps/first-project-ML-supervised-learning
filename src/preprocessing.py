import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
