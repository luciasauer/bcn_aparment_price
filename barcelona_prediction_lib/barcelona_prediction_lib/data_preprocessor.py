import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class NaNRemover:
    def __init__(self,columns_with_nan = ["age", "gender", "ethnicity"]):
        self.columns_with_nan = columns_with_nan
    # Constructor that initializes the NaNRemover class with a list of columns we want to treat
    # Takes as argumant a list of columns from which we want to remove rows with NaN values.
        '''
        :Param: a list of columns from which we want to remove NaN values columns "age", "gender", "ethnicity" by default.
        '''
    def remove_nan(self, df):
        '''
        Removes rows with NaN values in specific columns.
        
        :param df: DataFrame to clean.
        
        Return: Cleaned DataFrame with rows removed.
        '''
        return df.dropna(subset=self.columns_with_nan)

class NaNFiller:
    def __init__(self, columns_to_fill = ['height', 'weight']):
    # Constructor that initializes the NaNFiller class with columns to fill NaN values with the mean
        '''
        :Param: a list of columns from which we want to fill NaN values with the mean columns 'height', 'weight' by default.
        '''
        self.columns_to_fill = columns_to_fill
    
    def fill_nan_mean(self, df):
        '''
        Fills NaN values in specific columns with the mean.
        
        :param: DataFrame to fill NaN values in.
        
        Returns a DataFrame with NaN values filled with the mean.
        '''
        for column in self.columns_to_fill:
            df[column].fillna(df[column].mean(), inplace=True)
        return df
    
    def categorical_random_filler(self, df, column_name:str, categories_list:list, probs_list:list):
        '''
        Fills NaN values in specific categoric columns using a random dist.
        
        :param df: DataFrame to fill NaN values in.
        :param column_name: Name of the categorical column to imput.
        :param categories_list: List of categories.
        :param probs: List of Probabilities of each category.
        
        Returns a DataFrame with NaN values filled with the mean.
        '''
        missing_values = df[column_name].isnull().sum()
        randomizer_vector = np.random.choice(categories_list, size=missing_values, p=probs_list)
        df.loc[df[column_name].isna(), column_name] = randomizer_vector

class MeanOperations_ByColumn:
    def __init__(self, mean_column, df):
        
        '''
        Class that perform calculations and fillings based on the mean of a column.
        
        :param mean_column: Column name where you want to perform the operations.
        :param df: Input Dataframe
        '''
        self.mean_column = mean_column
        self.df = df
        
    def grouped_mean_filler(self,  group_by_column:str, round=None):
        '''
        Fills the nan with the mean based in a group_by.
        
        :param group_by_column: Column to Group.
        :param round: number of decimals, rounds the mean value.
        
        '''
        self.group_by_column = group_by_column
        self.column_mean_gb = self.df.groupby(self.group_by_column)[self.mean_column].mean()

        if round != None:
            self.column_mean_gb = self.column_mean_gb.round(round)
            self.df[self.mean_column] = self.df[self.mean_column].fillna(self.df[self.group_by_column].map(self.column_mean_gb))
            return print(f'The nulls remaining in the column {self.mean_column} are {self.df[self.mean_column].isnull().sum()}')
        else:
            self.df[self.mean_column] = self.df[self.mean_column].fillna(self.df[self.group_by_column].map(self.column_mean_gb))
            return print(f'The nulls remaining in the column {self.mean_column} are {self.df[self.mean_column].isnull().sum()}')
    
    
    def simple_mean_filler(self, round=None):
        
        '''
        Fills the nan with the mean value of the column.
        
        :param round: Boolean, rounds the mean value.
        '''
        mean = self.df[self.mean_column].mean()
        if round != None:
            self.df[self.mean_column] = self.df[self.mean_column].fillna(mean.round(round))
            return print(f'The nulls remaining in the column {self.mean_column} are {self.df[self.mean_column].isnull().sum()}')
        
        else:
            self.df[self.mean_column] = self.df[self.mean_column].fillna(mean)
        return print(f'The nulls remaining in the column {self.mean_column} are {self.df[self.mean_column].isnull().sum()}')

class Closest_Mean_Filler_numeric:
    def __init__(self, filler_column, column_to_fill, df, round=None):
        self.round = round
        self.column_to_fill = column_to_fill
        self.filler_column = filler_column
        self.df = df

    def fill_closest(self):
        # Step 1: Calculate the mean of column_to_fill based on filler_column groups
        column_to_fill_avg_dict = self.df.groupby(self.filler_column)[self.column_to_fill].mean()
        
        # Apply rounding if specified
        if self.round is not None:
            column_to_fill_avg_dict = column_to_fill_avg_dict.round(self.round)
        
        # Convert to a dictionary for lookup
        column_to_fill_avg_dict = column_to_fill_avg_dict.to_dict()

        # Step 2: Fill NaNs in column_to_fill based on filler_column's average values
        self.df[self.column_to_fill] = self.df.apply(
            lambda row: column_to_fill_avg_dict[row[self.filler_column]]
            if pd.isna(row[self.column_to_fill]) and row[self.filler_column] in column_to_fill_avg_dict
            else row[self.column_to_fill],
            axis=1
        )

        # Step 3: Fill any remaining NaNs by finding the closest value in column_to_fill_avg_dict
        for index, row in self.df.iterrows():
            if pd.isna(row[self.column_to_fill]):
                # Get the current filler column value for the row
                filler_value = row[self.filler_column]

                # Find the closest available mean in column_to_fill_avg_dict
                closest_mean_value = min(
                    column_to_fill_avg_dict.values(),
                    key=lambda mean: abs(mean - filler_value)
                )
                
                # Fill with the closest mean value
                self.df.at[index, self.column_to_fill] = closest_mean_value

        # Print the remaining NaNs count for verification
        print(f"Remaining NaN values in {self.column_to_fill}: {self.df[self.column_to_fill].isnull().sum()}")
        
        return self.df

class Closest_Mean_Filler_categorical:
    def __init__(self, filler_column, column_to_fill, df, round=None):
        self.round = round
        self.column_to_fill = column_to_fill
        self.filler_column = filler_column
        self.df = df

    def fill_closest(self):
        # Step 1: Calculate the mean of filler_column (e.g., 'num_crimes') for each unique value in column_to_fill (e.g., 'neighborhood')
        column_to_fill_avg_dict = self.df.groupby(self.column_to_fill)[self.filler_column].mean()
        
        # Apply rounding if specified
        if self.round is not None:
            column_to_fill_avg_dict = column_to_fill_avg_dict.round(self.round)
        
        # Convert to a dictionary for lookup
        column_to_fill_avg_dict = column_to_fill_avg_dict.to_dict()

        # Step 2: Fill NaNs in column_to_fill based on the closest average value of filler_column
        for index, row in self.df.iterrows():
            if pd.isna(row[self.column_to_fill]):  # If column_to_fill (e.g., 'neighborhood') is NaN
                # Get the current value in filler_column (e.g., 'num_crimes')
                filler_value = row[self.filler_column]
                
                # Find the closest neighborhood based on the average num_crimes
                closest_filler_value = min(
                    column_to_fill_avg_dict.keys(),
                    key=lambda nh: abs(column_to_fill_avg_dict[nh] - filler_value)
                )
                
                # Fill the missing neighborhood with the closest match
                self.df.at[index, self.column_to_fill] = closest_filler_value

        # Print the remaining NaNs count for verification
        print(f"Remaining NaN values in {self.column_to_fill}: {self.df[self.column_to_fill].isnull().sum()}")
        
        return self.df