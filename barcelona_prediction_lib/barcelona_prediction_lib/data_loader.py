# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# DataLoader class will facilitate loading the data, splitting it into training and testing sets
class DataLoaderSpliter:
    def __init__(self, file_path):
        # Constructor that takes the path to the CSV file as an argument
        self.file_path = file_path
    
    def load_and_split(self, test_size=0.2, random_state=42):
        '''
        Method that loads the data and splits it into training and testing sets
        
        :param test_size: Proportion of data to use for testing. 0.2 by default.
        :param random_state: Random state for reproducibility. 42 by default.
        
        Returns: 2 Dataframes: Training and testing.
        '''
        
        df = pd.read_csv(self.file_path)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, test_df

# %%



