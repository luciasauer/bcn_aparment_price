# %%
from abc import ABC, abstractmethod
import pandas as pd

class FeatureTransformer(ABC):
# Abstract base class for feature transformation
    @abstractmethod
    def transform(self, df):
        pass

class BinaryTransformer(FeatureTransformer):
# Converts gender into binary (M/F)
    def transform(self, df, column_name='gender', true_label:str='M', false_label:str="F"):
        '''
        Converts a Binary categorical into binary values (1/0).
        :param colun_name: Name of the Binary column you want to transform.
        :param true_label: Categorie that corresponds to 1.
        :param false_label: Categorie that corresponds to 2.
        '''
        self.column_name = column_name
        df[self.column_name] = df[self.column_name].map({true_label: 1, false_label: 0})
        return df

class HotEncoder(FeatureTransformer):
    def transform(self, df, column_name='ethnicity'):
        '''
        Performs one hot encoding on a categorical column.
        '''
        return pd.get_dummies(df, column_name, drop_first=True)


