# Random Forest Classifier
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer

class RandomForestModel:
    """
    A class to train and evaluate a Random Forest model for stock market prediction.

    Attributes:
    -----------
    X_train : DataFrame
        Training feature data.
    X_test : DataFrame
        Test feature data.
    y_train : Series
        Training target data.
    y_test : Series
        Test target data.
    model : RandomForestClassifier
        The Random Forest model instance.
    tscv : TimeSeriesSplit
        The time series cross-validator instance.
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize the RandomForestModel with training and test data.

        Parameters:
        -----------
        X_train : DataFrame
            Training feature data.
        X_test : DataFrame
            Test feature data.
        y_train : Series
            Training target data.
        y_test : Series
            Test target data.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.build_model()

    def build_model(self):
        """
        Build the Random Forest model with specified parameters.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 2, 1: 1, 2: 2})
        self.tscv = TimeSeriesSplit(n_splits=5)

    def train(self):
        """
        Train the Random Forest model using cross-validation and fit it on the training data.

        Prints:
        -------
        Cross-Validation Accuracy Scores and Mean Cross-Validation Accuracy.
        """
        accuracy_scorer = make_scorer(accuracy_score)
        X = pd.concat([self.X_train, self.X_test])
        y = pd.concat([self.y_train, self.y_test])
        cross_val_scores = cross_val_score(self.model, X, y, cv=self.tscv, scoring=accuracy_scorer)
        # st.write(f"Cross-Validation Accuracy Scores: {cross_val_scores}")
        # st.write(f"Mean Cross-Validation Accuracy: {np.mean(cross_val_scores)}")
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluate the trained Random Forest model on the test data.

        Returns:
        --------
        accuracy : float
            Accuracy of the model on the test data.
        report : str
            Classification report of the model's performance.
        cm : ndarray of shape (n_classes, n_classes)
            Confusion matrix of the model's predictions.
        """
        y_pred = self.model.predict(self.X_test)
        y_test_decoded = self.decode(self.y_test)
        y_pred_decoded = self.decode(y_pred)
        accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
        report = classification_report(y_test_decoded, y_pred_decoded, target_names=['Sell', 'Hold', 'Buy'])
        cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=[-1, 0, 1])
        return accuracy, report, cm



    def predict(self, X_new):
        """
        Predict the target for new data.

        Parameters:
        -----------
        X_new : DataFrame
            New feature data for prediction.

        Returns:
        --------
        y_pred : ndarray
            Predicted target values.
        """
        y_pred_encoded = self.model.predict(X_new)
        return self.decode(y_pred_encoded)
    
    def decode(self, y):
        """
        Decode the encoded target values back to original labels.

        Parameters:
        -----------
        y : ndarray
            Encoded target values.

        Returns:
        --------
        y_decoded : ndarray
            Decoded target values.
        """
        decode_map = {0: -1, 1: 0, 2: 1}
        return np.array([decode_map[val] for val in y])
    
    
    
 
# Model Training and Evalution

class ModelDevelopment:
    """
    A class to run and evaluate different models for stock market prediction.

    Attributes:
    -----------
    stock_data : pd.DataFrame
        The DataFrame containing stock market data.
    encode_map : dict
        A mapping dictionary to encode 'Buy_Sell_Indication' into numeric labels.
    features : pd.DataFrame
        Features for model training.
    target : pd.Series
        Target variable for model training.
    """

    def __init__(self, stock_data):
        """
        Initialize the ModelSelection instance with stock_data.

        Parameters:
        -----------
        stock_data : pd.DataFrame
            The DataFrame containing stock market data.
        """
        self.stock_data = stock_data
        self.encode_map = {-1: 0, 0: 1, 1: 2}
        self.scaler = StandardScaler()
        self.features, self.target = self.prepare_data()

    def prepare_data(self):
        """
        Prepare features and target variables from stock_data.

        Returns:
        --------
        features : pd.DataFrame
            Processed features for model training.
        target : pd.Series
            Processed target variable for model training.
        """
        features = self.stock_data.drop(columns=['Buy_Sell_Indication'])
        target = self.stock_data['Buy_Sell_Indication'].replace(self.encode_map)
        features_scaled = self.scaler.fit_transform(features)
        return pd.DataFrame(features_scaled, columns=features.columns), target

    def time_series_split(self, test_size=0.2):
        """
        Perform time series split for training and test sets.

        Parameters:
        -----------
        test_size : float, optional
            Size of the test set (default is 0.2).

        Returns:
        --------
        X_train : pd.DataFrame
            Training features.
        X_test : pd.DataFrame
            Test features.
        y_train : pd.Series
            Training target.
        y_test : pd.Series
            Test target.
        """
        split_index = int(len(self.features) * (1 - test_size))
        X_train = self.features.iloc[:split_index]
        X_test = self.features.iloc[split_index:]
        y_train = self.target.iloc[:split_index]
        y_test = self.target.iloc[split_index:]
        # Ensure test set has all classes
        while len(set(y_test)) < 3:
            split_index -= 1
            X_train = self.features.iloc[:split_index]
            X_test = self.features.iloc[split_index:]
            y_train = self.target.iloc[:split_index]
            y_test = self.target.iloc[split_index:]

        return X_train, X_test, y_train, y_test
    

    def run_model(self, model_class, **kwargs):
        """
        Instantiate and train a specified model class, then evaluate its performance.

        Parameters:
        -----------
        model_class : class
            The class of the model to instantiate and train.
        **kwargs : additional keyword arguments
            Additional arguments to pass to the model constructor.

        Returns:
        --------
        accuracy : float
            Accuracy of the model on the test data.
        report : str
            Classification report of the model's performance.
        cm : ndarray of shape (n_classes, n_classes)
            Confusion matrix of the model's predictions.
        """
        X_train, X_test, y_train, y_test = self.time_series_split()
        model = model_class(X_train, X_test, y_train, y_test, **kwargs)
        model.train()
        rf_accuracy, rf_report, rf_cm = model.evaluate()
        self.model = model
        results = {
            'Accuracy': rf_accuracy,
            'Classification Report': rf_report,
            'Confusion Matrix': rf_cm
        }
        st.write("Debug: Results - ", results)
        return results

    def run(self):
        results = self.run_model(RandomForestModel)
        st.subheader("Model Evaluation Results:")
        st.info(f"Accuracy: {results['Accuracy'] * 100:.2f}%")
        st.info("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(results['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Sell', 'Hold', 'Buy'], yticklabels=['Sell', 'Hold', 'Buy'])
        st.pyplot(fig)

        last_element = self.features.iloc[-1:]
        last_element_scaled = self.scaler.transform(last_element)
        prediction = self.model.predict(last_element_scaled)
        indication = int(prediction[0])  # Ensure indication is a Python integer
        indication_label = 'Buy' if indication == 1 else 'Hold' if indication == 0 else 'Sell'
        return indication_label