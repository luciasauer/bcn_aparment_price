# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


class Model:
    def __init__(self, feature_columns, target_column, model=None):
        """
        Initializes the model class with feature columns, target column, and optional hyperparameters.
        
        :param feature_columns: List of column names to be used as features.
        :param target_column: Name of the target column.
        :param model: Model object from sklearn. It must be compatible with .predict_proba
        """
        self.__feature_columns = feature_columns
        self.__target_column = target_column
        self.model = model
        
    def train(self, train_df):
        """
        Trains the model using the provided dataframe.
        """
        self.X_train = train_df[self.__feature_columns]
        self.y_train = train_df[self.__target_column]
        self.trained_model = self.model.fit(self.X_train, self.y_train)

    def predict(self, prediction_df):
        """
        Uses the trained model to make predictions on new data.
        
        :param prediction_df: The input DataFrame containing features for prediction.
        :return: Predictions pd.Series.
        :return: Probabilities pd.DataFrame.
        """
        if self.trained_model is None:
            raise ValueError("The model is not trained yet. Please call the `train` method first.")
        
        X_pred_df = prediction_df[self.__feature_columns]
        
        # Perform prediction and maintain the index of the input DataFrame
        self.predictions = pd.Series(self.trained_model.predict(X_pred_df), index=prediction_df.index, name='Predictions_DB')
        
        # Perform probability prediction, also maintaining the index
        self.probs = pd.DataFrame(self.trained_model.predict_proba(X_pred_df), index=prediction_df.index, columns=['Prob_no_DM', 'Prob_has_DM'])
        
        return self.predictions, self.probs
    


