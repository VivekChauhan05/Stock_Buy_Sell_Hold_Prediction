import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Data Preprocessing

class DataProcessor:
    def __init__(self, file_path=None, data=None):
        """

        Initializes the DataProcessor with the provided file path.

        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing the stock data.
        """

        if file_path:
            self.stock_data = pd.read_csv(file_path)
        elif data is not None:
            self.stock_data = data
        else:
            raise ValueError("Either file_path or data must be provided.")

        self.clean_data()
        self.visualize_data()     


    def clean_data(self):
        """
        Cleans the loaded stock data by handling missing values and ensuring correct data types.
        """
        missing_data = self.stock_data.isnull().sum()

        # Fill the missing values with 0
        self.stock_data.fillna(0, inplace=True)

        # Ensure data types are correct
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])

        return self.stock_data

    def visualize_data(self):
        """
        Visualizes the stock data using a candlestick chart with Plotly.
        """
        # Create a Plotly figure
        fig = go.Figure()

        # Add candlestick trace
        fig.add_trace(go.Candlestick(x=self.stock_data['Date'],
                                     open=self.stock_data['Open'],
                                     high=self.stock_data['High'],
                                     low=self.stock_data['Low'],
                                     close=self.stock_data['Close'],
                                     increasing=dict(line=dict(color='green')),
                                     decreasing=dict(line=dict(color='red')),
                                     name='Stock Price'))

        # Update layout
        fig.update_layout(title='Candlestick plot of Stock Price',
                          xaxis_title='Date',
                          yaxis_title='Price (in US Dollars)',
                          xaxis_rangeslider_visible=True,  # Enable rangeslider
                          xaxis_rangeslider_thickness=0.1,  # Set slider thickness
                          xaxis_rangeslider_bgcolor='rgba(0,0,0,0.1)',  # Set slider background color
                          hovermode='x unified',  # Enable hover mode
                          xaxis=dict(
                              rangeselector=dict(
                                  buttons=list([
                                      dict(count=1, label='1m', step='month', stepmode='backward'),
                                      dict(count=3, label='3m', step='month', stepmode='backward'),
                                      dict(count=6, label='6m', step='month', stepmode='backward'),
                                      dict(count=8, label='8m', step='month', stepmode='backward'),
                                      dict(count=1, label='YTD', step='year', stepmode='todate'),
                                      dict(count=1, label='1y', step='year', stepmode='backward'),
                                      dict(step='all')
                                  ])
                              ),
                              rangeslider=dict(visible=True),  # Make rangeslider visible by default
                              type='date'  # Set x-axis type to date
                          )
        )

        # Display the plot using Streamlit
        st.plotly_chart(fig)
        
        
 
# Technical Indicators        
        
# SMA 

class SMA:
    """
    A class to represent the Simple Moving Average (SMA) calculations and visualizations.
    """

    def __init__(self, stock_data):
        """
        Initializes the SMA class with stock data.

        Parameters:
        stock_data (pd.DataFrame): DataFrame containing stock data.
        """
        self.stock_data = stock_data

    def calculate_sma(self, window):
        """
        Calculate Simple Moving Average (SMA).

        Parameters:
        window (int): The window period for calculating the SMA.

        Returns:
        pd.Series: A pandas Series containing the SMA values.
        """
        return self.stock_data['Adj Close'].rolling(window=window).mean()

    def add_sma_columns(self, sma_days):
        """
        Adds SMA columns to the stock_data DataFrame.

        Parameters:
        sma_days (list): List of integers representing the SMA periods to calculate.
        """
        for day in sma_days:
            self.stock_data[f'SMA_{day}'] = self.calculate_sma(day)
        self.stock_data.fillna(0, inplace=True)

    def plot_sma(self, sma_days):
        """
        Plot SMA values along with the adjusted close prices.

        Parameters:
        sma_days (list): List of integers representing the SMA periods to plot.
        """
        fig = go.Figure()

        colors = ['blue', 'red', 'green', 'purple', 'orange', 'deepskyblue']  # Define a list of colors for SMAs
        color_index = 0  # Initialize color index

        # Add trace for Adjusted Close Price
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data['Adj Close'], mode='lines', name='Adjusted Close Price', line=dict(color='pink')))

        # Add traces for each SMA
        for day in sma_days:
            fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data[f'SMA_{day}'], mode='lines', name=f'{day}-Day SMA', line=dict(color=colors[color_index])))
            color_index = (color_index + 1) % len(colors)  # Cycle through the colors

        # Update layout
        fig.update_layout(title='Stock Prices and SMAs (Simple Moving Averages)',
                          xaxis_title='Date',
                          yaxis_title='Price (in US Dollars)',
                          legend_title='Legend',
                          hovermode='x unified',
                          xaxis=dict(
                              rangeselector=dict(
                                  buttons=list([
                                      dict(count=1, label='1m', step='month', stepmode='backward'),
                                      dict(count=3, label='3m', step='month', stepmode='backward'),
                                      dict(count=6, label='6m', step='month', stepmode='backward'),
                                      dict(count=8, label='8m', step='month', stepmode='backward'),
                                      dict(count=1, label='YTD', step='year', stepmode='todate'),
                                      dict(count=1, label='1y', step='year', stepmode='backward'),
                                      dict(step='all')
                                  ])
                              ),
                              rangeslider=dict(visible=True),
                              type='date'
                          ))

        
        # Display the plot using Streamlit
        st.plotly_chart(fig)

    def calculate_sma_crossovers(self, crossover_periods):
        """
        Calculate SMA crossovers and add them to the stock_data DataFrame.

        Parameters:
        crossover_periods (list of tuples): List of tuples containing short and long period pairs.
        """
        crossovers = []

        for short, long in crossover_periods:
            crossover_column = f'SMA_Crossover_{short}_{long}'
            self.stock_data[crossover_column] = 'Neutral'
            for i in range(1, len(self.stock_data)):  # Start from 1 to avoid index error
                if (self.stock_data[f'SMA_{short}'].iloc[i] > self.stock_data[f'SMA_{long}'].iloc[i] and
                        self.stock_data[f'SMA_{short}'].iloc[i-1] <= self.stock_data[f'SMA_{long}'].iloc[i-1]):
                    crossovers.append((self.stock_data['Date'].iloc[i], f'{short}&{long} SMA', 'Golden Cross(Bullish)'))
                    self.stock_data.at[i, crossover_column] = 'Golden Cross(Bullish)'
                elif (self.stock_data[f'SMA_{short}'].iloc[i] < self.stock_data[f'SMA_{long}'].iloc[i] and
                      self.stock_data[f'SMA_{short}'].iloc[i-1] >= self.stock_data[f'SMA_{long}'].iloc[i-1]):
                    crossovers.append((self.stock_data['Date'].iloc[i], f'{short}&{long} SMA', 'Death Cross(Bearish)'))
                    self.stock_data.at[i, crossover_column] = 'Death Cross(Bearish)'
                else:
                    crossovers.append((self.stock_data['Date'].iloc[i], f'{short}&{long} SMA', 'Neutral'))
                    self.stock_data.at[i, crossover_column] = 'Neutral'

        crossover_df = pd.DataFrame(crossovers, columns=['Date', 'Moving Average Crossover', 'Technical Indication'])
        crossover_data = crossover_df.pivot(index='Date', columns='Moving Average Crossover', values='Technical Indication')
        crossover_data.columns = [col for col in crossover_data.columns]
        crossover_data.reset_index(inplace=True)

        desired_order = ['Date'] + [f'{short}&{long} SMA' for short, long in crossover_periods]
        crossover_data = crossover_data[desired_order]

        return crossover_data

    # def save_sma_crossover_data(self, filename = 'sma_crossover_data.csv'):
    #     """
    #     Save the sMA Crossover data to a CSV file.

    #     Parameters:
    #     filename (str): The name of the CSV file to save the sMA Crossover data. Defaults to 'sma_crossover_data.csv'.
    #     """
    #     crossover_data.to_csv(filename, index=False)

    def plot_sma_crossovers(self, crossover_data, short_sma, long_sma):
        """
        Plot the SMA crossovers on a chart.

        Parameters:
        crossover_data (pd.DataFrame): DataFrame containing crossover data.
        short_sma (int): Short period for SMA crossover.
        long_sma (int): Long period for SMA crossover.
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces for specified SMAs
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data[f'SMA_{short_sma}'], mode='lines', name=f'{short_sma}-Day SMA', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data[f'SMA_{long_sma}'], mode='lines', name=f'{long_sma}-Day SMA', line=dict(color='red')))

        # Add trace for Adjusted Close Price
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data['Adj Close'], mode='lines', name='Adjusted Close Price', line=dict(color='green')))


        # Define colors and symbols for different crossover indicators
        crossover_colors = {
            'Golden Cross(Bullish)': 'green',
            'Death Cross(Bearish)': 'red'
        }
        crossover_symbols = {
            'Golden Cross(Bullish)': 'circle',
            'Death Cross(Bearish)': 'x'
        }

        # Add markers for each crossover indicator
        for crossover, color in crossover_colors.items():
            filtered_data = crossover_data[crossover_data[f'{short_sma}&{long_sma} SMA'] == crossover]
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=[self.stock_data.loc[self.stock_data['Date'] == date, 'Adj Close'].values[0] for date in filtered_data['Date']],
                mode='markers',
                marker=dict(color=color, symbol=crossover_symbols[crossover], size=10),
                showlegend=False,
                hovertext=crossover,
                hoverinfo='text'
            ))



        # Get the last date in the data
        last_date = self.stock_data['Date'].max()

        # Update layout to include date range selectors and range slider
        fig.update_layout(
            title=f'Stock Prices and {short_sma}-Day & {long_sma}-Day SMA with Crossovers',
            xaxis_title='Date',
            yaxis_title='Price (in US Dollars)',
            legend_title='Legend',
            hovermode='x unified',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label='1m', step='month', stepmode='backward'),
                        dict(count=3, label='3m', step='month', stepmode='backward'),
                        dict(count=6, label='6m', step='month', stepmode='backward'),
                        dict(count=8, label='8m', step='month', stepmode='backward'),
                        dict(count=1, label='YTD', step='year', stepmode='todate'),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step='all')
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date',
                range=[last_date - pd.DateOffset(months=8), last_date]
            )
        )


        # Display the plot using Streamlit
        st.plotly_chart(fig)
        
        
         
# EMA

class EMA:
    """
    A class to represent the Exponential Moving Average (EMA) calculations and visualizations.
    """

    def __init__(self, stock_data):
        """
        Initializes the EMA class with stock data.

        Parameters:
        stock_data (pd.DataFrame): DataFrame containing stock data.
        """
        self.stock_data = stock_data

    def calculate_ema(self, window):
        """
        Calculate Exponential Moving Average (EMA).

        Parameters:
        window (int): The window period for calculating the EMA.

        Returns:
        pd.Series: A pandas Series containing the EMA values.
        """
        return self.stock_data['Adj Close'].ewm(span=window, adjust=False).mean()

    def add_ema_columns(self, ema_days):
        """
        Adds EMA columns to the stock_data DataFrame.

        Parameters:
        ema_days (list): List of integers representing the EMA periods to calculate.
        """
        for day in ema_days:
            self.stock_data[f'EMA_{day}'] = self.calculate_ema(day)
        self.stock_data.fillna(0, inplace=True)

    def plot_ema(self, ema_days):
        """
        Plot EMA values along with the adjusted close prices.

        Parameters:
        ema_days (list): List of integers representing the EMA periods to plot.
        """
        fig = go.Figure()

        colors = ['blue', 'red', 'green', 'purple', 'orange']  # Define a list of colors for SMAs
        color_index = 0  # Initialize color index

        # Add trace for Adjusted Close Price
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data['Adj Close'], mode='lines', name='Adjusted Close Price', line=dict(color='deepskyblue')))

        # Add traces for each EMA
        for day in ema_days:
            fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data[f'EMA_{day}'], mode='lines', name=f'{day}-Day EMA', line=dict(color=colors[color_index])))
            color_index = (color_index + 1) % len(colors)  # Cycle through the colors



        # Update layout
        fig.update_layout(title='Stock Prices and EMAs (Exponential Moving Averages)',
                          xaxis_title='Date',
                          yaxis_title='Price (in US Dollars)',
                          hovermode='x unified',
                          legend_title='Legend',
                          xaxis=dict(
                              rangeselector=dict(
                                  buttons=list([
                                      dict(count=1, label='1m', step='month', stepmode='backward'),
                                      dict(count=3, label='3m', step='month', stepmode='backward'),
                                      dict(count=6, label='6m', step='month', stepmode='backward'),
                                      dict(count=8, label='8m', step='month', stepmode='backward'),
                                      dict(count=1, label='YTD', step='year', stepmode='todate'),
                                      dict(count=1, label='1y', step='year', stepmode='backward'),
                                      dict(step='all')
                                  ])
                              ),
                              rangeslider=dict(visible=True),
                              type='date'
                          ))

        
        # Display the plot using Streamlit
        st.plotly_chart(fig)

    def calculate_ema_crossovers(self, ema_crossover_periods):
        """
        Calculate EMA crossovers and add them to the stock_data DataFrame.

        Parameters:
        ema_crossover_periods (list of tuples): List of tuples containing short and long period pairs.
        """
        crossovers = []

        for short, long in ema_crossover_periods:
            crossover_column = f'EMA_Crossover_{short}_{long}'
            self.stock_data[crossover_column] = 'Neutral'
            for i in range(1, len(self.stock_data)):
                if (self.stock_data[f'EMA_{short}'].iloc[i] > self.stock_data[f'EMA_{long}'].iloc[i] and
                        self.stock_data[f'EMA_{short}'].iloc[i-1] <= self.stock_data[f'EMA_{long}'].iloc[i-1]):
                    crossovers.append((self.stock_data['Date'].iloc[i], f'{short}&{long} EMA', 'Golden Cross(Bullish)'))
                    self.stock_data.at[i, crossover_column] = 'Golden Cross(Bullish)'
                elif (self.stock_data[f'EMA_{short}'].iloc[i] < self.stock_data[f'EMA_{long}'].iloc[i] and
                      self.stock_data[f'EMA_{short}'].iloc[i-1] >= self.stock_data[f'EMA_{long}'].iloc[i-1]):
                    crossovers.append((self.stock_data['Date'].iloc[i], f'{short}&{long} EMA', 'Death Cross(Bearish)'))
                    self.stock_data.at[i, crossover_column] = 'Death Cross(Bearish)'
                else:
                    crossovers.append((self.stock_data['Date'].iloc[i], f'{short}&{long} EMA', 'Neutral'))
                    self.stock_data.at[i, crossover_column] = 'Neutral'

        crossover_df = pd.DataFrame(crossovers, columns=['Date', 'Moving Average Crossover', 'Technical Indication'])
        crossover_data = crossover_df.pivot(index='Date', columns='Moving Average Crossover', values='Technical Indication')
        crossover_data.columns = [col for col in crossover_data.columns]
        crossover_data.reset_index(inplace=True)

        desired_order = ['Date'] + [f'{short}&{long} EMA' for short, long in ema_crossover_periods]
        crossover_data = crossover_data[desired_order]

        return crossover_data


    # def save_ema_crossover_data(self, filename = 'ema_crossover_data.csv'):
    #     """
    #     Save the EMA Crossover data to a CSV file.

    #     Parameters:
    #     filename (str): The name of the CSV file to save the EMA Crossover data. Defaults to 'ema_crossover_data.csv'.
    #     """
    #     crossover_data.to_csv(filename, index=False)


    def plot_ema_crossovers(self, crossover_data, short_ema, long_ema):
        """
        Plot the EMA crossovers on a chart.

        Parameters:
        crossover_data (pd.DataFrame): DataFrame containing crossover data.
        short_ema (int): Short period for EMA crossover.
        long_ema (int): Long period for EMA crossover.
        """
        fig = go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces for specified EMAs
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data[f'EMA_{short_ema}'], mode='lines', name=f'{short_ema}-Day EMA', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data[f'EMA_{long_ema}'], mode='lines', name=f'{long_ema}-Day EMA', line=dict(color='red')))

        # Add trace for Adjusted Close Price
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data['Adj Close'], mode='lines', name='Adjusted Close Price', line=dict(color='green')))


        # Define colors and symbols for different crossover indicators
        crossover_colors = {
            'Golden Cross(Bullish)': 'green',
            'Death Cross(Bearish)': 'red'
        }
        crossover_symbols = {
            'Golden Cross(Bullish)': 'circle',
            'Death Cross(Bearish)': 'x'
        }

        # Add markers for each crossover indicator
        for crossover, color in crossover_colors.items():
            filtered_data = crossover_data[crossover_data[f'{short_ema}&{long_ema} EMA'] == crossover]
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=[self.stock_data.loc[self.stock_data['Date'] == date, 'Adj Close'].values[0] for date in filtered_data['Date']],
                mode='markers',
                marker=dict(color=color, symbol=crossover_symbols[crossover], size=10),
                showlegend=False,
                hovertext=crossover,
                hoverinfo='text'
            ))



        # Get the last date in the data
        last_date = self.stock_data['Date'].max()

        # Update layout to include date range selectors and range slider
        fig.update_layout(
            title=f'Stock Prices and {short_ema}-Day & {long_ema}-Day EMA with Crossovers',
            xaxis_title='Date',
            yaxis_title='Price (in US Dollars)',
            legend_title='Legend',
            hovermode='x unified',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label='1m', step='month', stepmode='backward'),
                        dict(count=3, label='3m', step='month', stepmode='backward'),
                        dict(count=6, label='6m', step='month', stepmode='backward'),
                        dict(count=8, label='8m', step='month', stepmode='backward'),
                        dict(count=1, label='YTD', step='year', stepmode='todate'),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step='all')
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date',
                range=[last_date - pd.DateOffset(months=8), last_date]
            )
        )


        # Display the plot using Streamlit
        st.plotly_chart(fig)
        
        
        
# RSI

class RSI:
    """
    A class to represent the Relative Strength Index (RSI) calculations and visualizations.
    """

    def __init__(self, stock_data):
        """
        Initializes the RSI class with stock data.

        Parameters:
        stock_data (pd.DataFrame): DataFrame containing stock data.
        """
        self.stock_data = stock_data

    def calculate_rsi(self, data, window):
        """
        Calculate the Relative Strength Index (RSI).

        Parameters:
        data (pd.Series): Series containing the stock prices.
        window (int): The window period for calculating the RSI.

        Returns:
        pd.Series: A pandas Series containing the RSI values.
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def rsi_technical_indication(self, rsi):
        """
        Determine the technical indication based on the RSI value.

        Parameters:
        rsi (float): The RSI value.

        Returns:
        str: The technical indication ('Oversold', 'Bearish', 'Neutral', 'Bullish', 'Overbought').
        """
        if rsi < 25:
            return 'Oversold'
        elif 25 <= rsi < 45:
            return 'Bearish'
        elif 45 <= rsi < 55:
            return 'Neutral'
        elif 55 <= rsi < 75:
            return 'Bullish'
        else:
            return 'Overbought'

    def add_rsi_column(self, window):
        """
        Add the RSI column to the stock_data DataFrame.

        Parameters:
        window (int): The window period for calculating the RSI.
        """
        self.stock_data['RSI_14'] = self.calculate_rsi(self.stock_data['Close'], window)
        self.stock_data['RSI_Technical_Indication'] = self.stock_data['RSI_14'].apply(self.rsi_technical_indication)
        self.stock_data.at[0, 'RSI_14'] = 0  # Handling the first value if needed

    # def save_rsi_data(self, filename='rsi_data.csv'):
    #     """
    #     Save the RSI data to a CSV file.

    #     Parameters:
    #     filename (str): The name of the CSV file to save the RSI data. Defaults to 'rsi_data.csv'.
    #     """
    #     rsi_data = self.stock_data[['Date', 'RSI_14', 'RSI_Technical_Indication']].copy()
    #     rsi_data.to_csv(filename, index=False)

    def plot_rsi(self):
        """
        Plot the RSI values along with technical indications.
        """
        rsi_data = self.stock_data[['Date', 'RSI_14', 'RSI_Technical_Indication']]

        # Plot RSI values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rsi_data['Date'], y=rsi_data['RSI_14'], mode='lines', name='RSI 14', line=dict(color='cyan')))

        # Define colors and symbols for different technical indications
        indication_colors = {
            'Overbought': 'red',
            'Bearish': 'blue',
            'Bullish': 'orange',
            'Oversold': 'green'
        }
        indication_symbols = {
            'Overbought': 'triangle-up',
            'Bearish': 'circle',
            'Bullish': 'x',
            'Oversold': 'triangle-down'
        }

        # Add markers for each technical indication
        for indication, color in indication_colors.items():
            filtered_data = rsi_data[rsi_data['RSI_Technical_Indication'] == indication]
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['RSI_14'],
                mode='markers',
                marker=dict(color=color, symbol=indication_symbols[indication], size=7),
                name=indication
            ))

        # Get the last date in the data
        last_date = rsi_data['Date'].max()

        fig.update_layout(title='RSI (Relative Strength Index)',
                          xaxis_title='Date',
                          yaxis_title='RSI',
                          hovermode='x unified',
                          legend_title='Legend',
                          xaxis=dict(
                              rangeselector=dict(
                                  buttons=list([
                                      dict(count=1, label='1m', step='month', stepmode='backward'),
                                      dict(count=3, label='3m', step='month', stepmode='backward'),
                                      dict(count=6, label='6m', step='month', stepmode='backward'),
                                      dict(count=8, label='8m', step='month', stepmode='backward'),
                                      dict(count=1, label='YTD', step='year', stepmode='todate'),
                                      dict(count=1, label="1y", step="year", stepmode="backward"),
                                      dict(step='all')
                                  ])
                              ),
                              rangeslider=dict(visible=True),
                              type='date',
                              range=[last_date - pd.DateOffset(months=8), last_date]
                          ))


        # Display the plot using Streamlit
        st.plotly_chart(fig)



# MACD

class MACD:
    """
    A class to represent the Moving Average Convergence Divergence (MACD) calculations and visualizations.
    """

    def __init__(self, stock_data):
        """
        Initializes the MACD class with stock data.

        Parameters:
        stock_data (pd.DataFrame): DataFrame containing stock data.
        """
        self.stock_data = stock_data

    def calculate_macd(self):
        """
        Calculate the MACD and MACD Signal values.
        """
        self.stock_data['MACD'] = self.stock_data['EMA_12'] - self.stock_data['EMA_26']
        self.stock_data['MACD_Signal'] = self.stock_data['MACD'].ewm(span=9, adjust=False).mean()
        self.macd_data = self.stock_data[['Date', 'MACD', 'MACD_Signal']].copy()
        self.macd_data['Technical_Indication'] = 'Neutral'
        self.stock_data['MACD_Technical_Indication'] = 'Neutral'

        for i in range(1, len(self.macd_data)):  # Start from 1 to avoid index error
            if self.macd_data['MACD'].iloc[i] > self.macd_data['MACD_Signal'].iloc[i] and self.macd_data['MACD'].iloc[i-1] <= self.macd_data['MACD_Signal'].iloc[i-1]:
                self.macd_data.at[i, 'Technical_Indication'] = 'Bullish Crossover'
                self.stock_data.at[i, 'MACD_Technical_Indication'] = 'Bullish'
            elif self.macd_data['MACD'].iloc[i] < self.macd_data['MACD_Signal'].iloc[i] and self.macd_data['MACD'].iloc[i-1] >= self.macd_data['MACD_Signal'].iloc[i-1]:
                self.macd_data.at[i, 'Technical_Indication'] = 'Bearish Crossover'
                self.stock_data.at[i, 'MACD_Technical_Indication'] = 'Bearish'
            elif self.macd_data['MACD'].iloc[i] > 0 and self.macd_data['MACD'].iloc[i-1] <= 0:
                self.macd_data.at[i, 'Technical_Indication'] = 'MACD Above Zero Line'
            elif self.macd_data['MACD'].iloc[i] < 0 and self.macd_data['MACD'].iloc[i-1] >= 0:
                self.macd_data.at[i, 'Technical_Indication'] = 'MACD Below Zero Line'
            else:
                self.stock_data.at[i, 'MACD_Technical_Indication'] = 'Neutral'

    # def save_macd_data(self, filename='macd_data.csv'):
    #     """
    #     Save the MACD data to a CSV file.

    #     Parameters:
    #     filename (str): The name of the CSV file to save the MACD data. Defaults to 'macd_data.csv'.
    #     """
    #     self.macd_data.to_csv(filename, index=False)

    def plot_macd(self):
        """
        Plot the MACD values along with technical indications.
        """
        self.macd_data['MACD_Histogram'] = self.macd_data['MACD'] - self.macd_data['MACD_Signal']
        fig = go.Figure()
        # Create subplots with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.3,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
        )

        # Add closing price and EMAs to the first subplot
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data['Adj Close'], mode='lines', name='Adjusted Close Price', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data['EMA_12'], mode='lines', name='12-period EMA', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data['EMA_26'], mode='lines', name='26-period EMA', line=dict(color='green')), row=1, col=1)

        # Add MACD and Signal line to the second subplot
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data['MACD'], mode='lines', name='MACD', line=dict(color='violet')), row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=self.stock_data['Date'], y=self.stock_data['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='orange')), row=2, col=1, secondary_y=False)

        # Create a list of colors based on the value of MACD_Histogram
        colors = ['green' if val >= 0 else 'red' for val in self.macd_data['MACD_Histogram']]

        # Add MACD histogram to the second subplot with conditional coloring
        fig.add_trace(go.Bar(x=self.stock_data['Date'], y=self.macd_data['MACD_Histogram'], name='MACD Histogram', marker_color=colors), row=2, col=1, secondary_y=True)


        # Define colors and symbols for different technical indications
        indication_colors = {
            'Bullish Crossover': 'green',
            'Bearish Crossover': 'red',
            'MACD Above Zero Line': 'blue',
            'MACD Below Zero Line': 'orange'
        }
        indication_symbols = {
            'Bullish Crossover': 'triangle-up',
            'Bearish Crossover': 'triangle-down',
            'MACD Above Zero Line': 'diamond',
            'MACD Below Zero Line': 'diamond'
        }

        # Add markers for each technical indication
        for indication, color in indication_colors.items():
            filtered_data = self.macd_data[self.macd_data['Technical_Indication'] == indication]
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['MACD'],
                mode='markers',
                marker=dict(color=color, symbol=indication_symbols[indication], size=7),
                name=indication,
                showlegend=False
            ), row=2, col=1)


        # Get the last date in the data
        last_date = self.stock_data['Date'].max()

        
        # Update layout for better appearance
        fig.update_layout(
            title='Stock Price and MACD Indicator (Moving Average Convergence Divergence)',
            xaxis_title='Date',
            yaxis_title='Price (in US Dollars)',
            legend_title='Legend',
            height=800,
            width=1200,
            hovermode='x unified',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label='1m', step='month', stepmode='backward'),
                        dict(count=3, label='3m', step='month', stepmode='backward'),
                        dict(count=6, label='6m', step='month', stepmode='backward'),
                        dict(count=8, label='8m', step='month', stepmode='backward'),
                        dict(count=1, label='YTD', step='year', stepmode='todate'),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step='all')
                    ])
                ),
                type='date',
                rangeslider=dict(visible=True),
                range=[last_date - pd.DateOffset(months=8), last_date]  # Default range: Last 1 month
            )
        )

        # Update y-axis titles
        fig.update_yaxes(title_text='Price (in US Dollars)', row=1, col=1)
        fig.update_yaxes(title_text='MACD', row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text='MACD Histogram', row=2, col=1, secondary_y=True)


        # Display the plot using Streamlit
        st.plotly_chart(fig)                        
    


# ADX

class ADX:
    """
    A class to represent the Average Directional Index (ADX) calculations and visualizations.
    """

    def __init__(self, stock_data):
        """
        Initializes the ADX class with stock data.

        Parameters:
        stock_data (pd.DataFrame): DataFrame containing stock data.
        """
        self.stock_data = stock_data

    def calculate_adx(self, n=14):
        """
        Calculate the ADX, +DI, and -DI values.

        Parameters:
        n (int): Number of periods for ADX calculation. Default is 14.
        """
        # Calculate TR (True Range), +DM (Positive Directional Movement), and -DM (Negative Directional Movement)
        self.stock_data['TR'] = self.stock_data[['High', 'Low', 'Close']].max(axis=1) - self.stock_data[['High', 'Low', 'Close']].min(axis=1)
        self.stock_data['+DM'] = np.where((self.stock_data['High'] - self.stock_data['High'].shift(1)) > (self.stock_data['Low'].shift(1) - self.stock_data['Low']),
                                          self.stock_data['High'] - self.stock_data['High'].shift(1), 0)
        self.stock_data['+DM'] = np.where(self.stock_data['+DM'] < 0, 0, self.stock_data['+DM'])
        self.stock_data['-DM'] = np.where((self.stock_data['Low'].shift(1) - self.stock_data['Low']) > (self.stock_data['High'] - self.stock_data['High'].shift(1)),
                                          self.stock_data['Low'].shift(1) - self.stock_data['Low'], 0)
        self.stock_data['-DM'] = np.where(self.stock_data['-DM'] < 0, 0, self.stock_data['-DM'])

        # Calculate smoothed TR, +DI, -DI, and ADX
        self.stock_data['TRn'] = self.stock_data['TR'].rolling(window=n).mean()
        self.stock_data['+DI'] = (self.stock_data['+DM'].rolling(window=n).mean() / self.stock_data['TRn']) * 100
        self.stock_data['-DI'] = (self.stock_data['-DM'].rolling(window=n).mean() / self.stock_data['TRn']) * 100
        self.stock_data['DX'] = (np.abs(self.stock_data['+DI'] - self.stock_data['-DI']) / np.abs(self.stock_data['+DI'] + self.stock_data['-DI'])) * 100
        self.stock_data['ADX'] = self.stock_data['DX'].rolling(window=n).mean()

        # Fill NA values in +DI, -DI, and ADX with their averages
        self.stock_data['+DI'] = self.stock_data['+DI'].fillna(self.stock_data['+DI'].mean())
        self.stock_data['-DI'] = self.stock_data['-DI'].fillna(self.stock_data['-DI'].mean())
        self.stock_data['ADX'] = self.stock_data['ADX'].fillna(self.stock_data['ADX'].mean())

        # Clean up unnecessary columns
        self.stock_data.drop(['TR', '+DM', '-DM', 'TRn', 'DX'], axis=1, inplace=True)

    def plot_adx(self):
        """
        Plot the Candlestick chart with ADX, +DI, and -DI indicators.
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.stock_data['Date'],
                open=self.stock_data['Open'],
                high=self.stock_data['High'],
                low=self.stock_data['Low'],
                close=self.stock_data['Close'],
                name='Candlesticks'
            ),
            row=1, col=1
        )

        # Plot +DI, -DI, and ADX
        fig.add_trace(
            go.Scatter(
                x=self.stock_data['Date'],
                y=self.stock_data['+DI'],
                mode='lines',
                name='+DI',
                line=dict(color='green')
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.stock_data['Date'],
                y=self.stock_data['-DI'],
                mode='lines',
                name='-DI',
                line=dict(color='red')
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.stock_data['Date'],
                y=self.stock_data['ADX'],
                mode='lines',
                name='ADX',
                line=dict(color='blue')
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title='Stock Data with ADX (Average Directional Index), +DI, and -DI Indicators',
            xaxis_title='Date',
            yaxis_title='Price (in US Dollars)',
            legend=dict(x=0.02, y=0.98),
            height=800,
            width=1200,
            hovermode='x unified'
        )

        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=8, label='8m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            type='date'
        )


        # Display the plot using Streamlit
        st.plotly_chart(fig)
        
        
        
# Buy Sell & Hold Calculation and Visualisation

class BuySellHold:
    """
    A class to represent the Buy, Sell, and Hold signals calculations and visualizations.
    """

    def __init__(self, stock_data):
        """
        Initializes the BuySellHold class with stock data.

        Parameters:
        stock_data (pd.DataFrame): DataFrame containing stock data.
        """
        self.stock_data = stock_data

    def generate_signals(self):
        """
        Generate Buy, Sell, and Hold signals based on various indicators and add them to stock_data.
        """
        # Initialize new columns in self.stock_data
        self.stock_data['Signal'] = 0
        self.stock_data['Position'] = 0
        self.stock_data['Buy_Sell_Indication'] = 'Hold'

        # Generate signals based on Moving Averages
        self.stock_data['Signal'] = np.where(self.stock_data['SMA_50'] > self.stock_data['SMA_200'], 1, -1)

        # Generate and refine signals based on RSI
        self.stock_data['Signal'] = np.where(self.stock_data['RSI_14'] < 30, 1, self.stock_data['Signal'])
        self.stock_data['Signal'] = np.where(self.stock_data['RSI_14'] > 70, -1, self.stock_data['Signal'])

        # Generate and refine signals based on RSI
        self.stock_data['Signal'] = np.where((self.stock_data['RSI_14'] >= 30) & (self.stock_data['RSI_14'] <= 70), 0, self.stock_data['Signal'])

        # Generate and refine signals based on MACD
        self.stock_data['Signal'] = np.where(self.stock_data['MACD'] > self.stock_data['MACD_Signal'],
                                             np.where(self.stock_data['Signal'] == -1, 0, 1),
                                             self.stock_data['Signal'])
        self.stock_data['Signal'] = np.where(self.stock_data['MACD'] < self.stock_data['MACD_Signal'],
                                             np.where(self.stock_data['Signal'] == 1, 0, -1),
                                             self.stock_data['Signal'])

        # Generate and refine signals based on ADX and DI crossovers
        self.stock_data['Signal'] = np.where((self.stock_data['+DI'] > self.stock_data['-DI']) & (self.stock_data['ADX'] > 25),
                                             1, self.stock_data['Signal'])  # Buy signal: +DI > -DI and ADX > 25
        self.stock_data['Signal'] = np.where((self.stock_data['-DI'] > self.stock_data['+DI']) & (self.stock_data['ADX'] > 25),
                                             -1, self.stock_data['Signal'])  # Sell signal: -DI > +DI and ADX > 25
        self.stock_data['Signal'] = np.where(self.stock_data['ADX'] < 25, 0, self.stock_data['Signal'])  # Hold signal: ADX < 25

        # Initialize 'Position' column with zeros
        self.stock_data['Position'] = 0

        # Loop through the DataFrame to generate positions
        position = 0  # 0 for hold, 1 for buy, -1 for sell
        for i in range(1, len(self.stock_data)):
            if self.stock_data['Signal'].iloc[i] == 1:
                position = 1
            elif self.stock_data['Signal'].iloc[i] == -1:
                position = -1
            else:
                position = 0

            # Assign position to 'Position' column
            self.stock_data.at[self.stock_data.index[i], 'Position'] = position

        # Handle initial position at index 0
        self.stock_data.at[self.stock_data.index[0], 'Position'] = 0

        # Map positions to Buy_Sell_Indication
        self.stock_data['Buy_Sell_Indication'] = np.where(self.stock_data['Position'] == 1, 'Buy',
                                                          np.where(self.stock_data['Position'] == -1, 'Sell', 'Hold'))

    def visualize_signals(self):
        """
        Visualize Buy, Sell, and Hold signals along with other indicators.
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add the price data to the plot
        fig.add_trace(
            go.Scatter(x=self.stock_data['Date'], y=self.stock_data['Adj Close'], name='Adjacent Close Price', line=dict(color='blue')),
            secondary_y=False,
        )

        # Add the buy signals
        fig.add_trace(
            go.Scatter(x=self.stock_data[self.stock_data['Buy_Sell_Indication'] == 'Buy']['Date'],
                       y=self.stock_data[self.stock_data['Buy_Sell_Indication'] == 'Buy']['Adj Close'],
                       mode='markers', marker=dict(symbol='triangle-up', color='green', size=7),
                       name='Buy Signal'),
            secondary_y=False,
        )

        # Add the sell signals
        fig.add_trace(
            go.Scatter(x=self.stock_data[self.stock_data['Buy_Sell_Indication'] == 'Sell']['Date'],
                       y=self.stock_data[self.stock_data['Buy_Sell_Indication'] == 'Sell']['Adj Close'],
                       mode='markers', marker=dict(symbol='triangle-down', color='red', size=7),
                       name='Sell Signal'),
            secondary_y=False,
        )

        # Add the Moving Averages
        fig.add_trace(
            go.Scatter(x=self.stock_data['Date'], y=self.stock_data['SMA_50'], name='SMA 50', line=dict(color='violet')),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=self.stock_data['Date'], y=self.stock_data['SMA_200'], name='SMA 200', line=dict(color='orange')),
            secondary_y=False,
        )

        # Add RSI to the secondary y-axis
        fig.add_trace(
            go.Scatter(x=self.stock_data['Date'], y=self.stock_data['RSI_14'], name='RSI 14', line=dict(color='cyan', dash='dot')),
            secondary_y=True,
        )

        # Add -DI
        fig.add_trace(
            go.Scatter(x=self.stock_data['Date'], y=self.stock_data['-DI'], name='-DI', line=dict(color='red')),
            secondary_y=True,
        )

        # Add +DI
        fig.add_trace(
            go.Scatter(x=self.stock_data['Date'], y=self.stock_data['+DI'], name='+DI', line=dict(color='green')),
            secondary_y=True,
        )

        # Add ADX
        fig.add_trace(
            go.Scatter(x=self.stock_data['Date'], y=self.stock_data['ADX'], name='ADX', line=dict(color='purple')),
            secondary_y=True,
        )

        # Update the layout
        fig.update_layout(
            title='Stock Buy and Sell Signals',
            xaxis_title='Date',
            yaxis_title='Adjusted Close Price (in US Dollars)',
            legend_title='Legend',
            height=600,
            width=1300,
            hovermode='x unified'
        )

        # Get the last date in the data
        last_date = self.stock_data['Date'].max()


        # Adding date range selector buttons
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=8, label='8m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            type='date',
            range=[last_date - pd.DateOffset(months=8), last_date]  # Default range: Last 5 years
        )


        # Display the plot using Streamlit
        st.plotly_chart(fig)     
        
        
        
# Feature Scaling 

class FeatureScaling:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.encode_categorical_columns()
        self.scale_features()

    def encode_categorical_columns(self):
        """
        Encode categorical columns based on predefined mapping.
        """
        encode_map = {
            'SMA_Crossover_5_20': {'Neutral': 0, 'Death Cross(Bearish)': -1, 'Golden Cross(Bullish)': 1},
            'SMA_Crossover_20_50': {'Neutral': 0, 'Death Cross(Bearish)': -1, 'Golden Cross(Bullish)': 1},
            'SMA_Crossover_50_200': {'Neutral': 0, 'Death Cross(Bearish)': -1, 'Golden Cross(Bullish)': 1},
            'EMA_Crossover_9_12': {'Neutral': 0, 'Death Cross(Bearish)': -1, 'Golden Cross(Bullish)': 1},
            'EMA_Crossover_12_26': {'Neutral': 0, 'Death Cross(Bearish)': -1, 'Golden Cross(Bullish)': 1},
            'EMA_Crossover_50_200': {'Neutral': 0, 'Death Cross(Bearish)': -1, 'Golden Cross(Bullish)': 1},
            'RSI_Technical_Indication': {'Overbought': 2, 'Bullish': 1, 'Neutral': 0, 'Bearish': -1, 'Oversold': -2},
            'MACD_Technical_Indication': {'Bullish': 1, 'Bearish': -1, 'Neutral': 0},
            'Buy_Sell_Indication': {'Hold': 0, 'Buy': 1, 'Sell': -1}
        }
        self.stock_data.replace(encode_map, inplace=True)

    def scale_features(self):
        """
        Scale numeric features using StandardScaler.
        """
        # Separate features and target
        features = self.stock_data.drop(columns=['Buy_Sell_Indication', 'Signal', 'Position'])
        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()

        # Initialize the scaler
        scaler = StandardScaler()

        # Apply the scaler to the numeric features
        features[numeric_features] = scaler.fit_transform(features[numeric_features])

        # Update the original stock_data DataFrame with the scaled features
        self.stock_data.update(features)

    # def summary_statistics(self):
    #     """
    #     Print summary statistics of the stock data after encoding and scaling.
    #     """
    #     print(self.stock_data.describe())
        
        
 
 # Feature Selection

class FeatureSelection:
    def __init__(self, stock_data):
        stock_data.set_index('Date', inplace=True)
        self.stock_data = stock_data
        self.features = self.stock_data.drop(columns=['Buy_Sell_Indication', 'Signal', 'Position'])
        self.target = self.stock_data['Buy_Sell_Indication']
        self.select_features()

    def feature_importance_from_trees(self):
        """
        Perform feature selection based on feature importance from Random Forest.
        """
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.features, self.target)
        importances = model.feature_importances_
        feature_importances = pd.DataFrame({'Feature': self.features.columns, 'Importance': importances})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
        selected_features_rf = feature_importances['Feature'].iloc[:12].tolist()
        return selected_features_rf

    def univariate_feature_selection(self):
        """
        Perform univariate feature selection using ANOVA F-value for classification.
        """
        selector = SelectKBest(score_func=f_classif, k=12)
        selector.fit(self.features, self.target)
        selected_feature_names = self.features.columns[selector.get_support()].tolist()
        return selected_feature_names

    def feature_selection_rfe(self):
        """
        Perform feature selection using Recursive Feature Elimination (RFE) with Random Forest classifier.
        """
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=12)
        selector.fit(self.features, self.target)
        selected_features_rfe = self.features.columns[selector.support_].tolist()
        return selected_features_rfe

    def most_frequent_features(self):
        """
        Combine selected features from all methods and count their frequencies.
        """
        selected_features_rf = self.feature_importance_from_trees()
        selected_feature_names = self.univariate_feature_selection()
        selected_features_rfe = self.feature_selection_rfe()

        all_selected_features = selected_features_rf + selected_feature_names + selected_features_rfe
        feature_counts = Counter(all_selected_features)
        feature_counts_df = pd.DataFrame.from_dict(feature_counts, orient='index', columns=['Frequency'])
        feature_counts_df = feature_counts_df.sort_values(by='Frequency', ascending=False)

        threshold = 2  # Select features chosen by at least 2 methods
        final_selected_features = feature_counts_df[feature_counts_df['Frequency'] >= threshold].index.tolist()

        return final_selected_features

    def select_features(self):
        """
        Select final features and update the stock_data with selected features and target variable.
        """
        final_selected_features = self.most_frequent_features()
        # Determine the features to drop
        features_to_drop = [col for col in self.stock_data.columns if col not in final_selected_features + ['Buy_Sell_Indication']]

        # Drop the unselected features
        self.stock_data.drop(columns=features_to_drop, inplace=True)
        selected_features = ', '.join(final_selected_features)
        # self.save_updated_stock_data(self.stock_data)
        return selected_features

    # def save_updated_stock_data(self, updated_stock_data):
    #     """
    #     Save the updated stock_data to a CSV file.
    #     """
    #     updated_stock_data.to_csv('updated_stock_data.csv', index=True)
                             