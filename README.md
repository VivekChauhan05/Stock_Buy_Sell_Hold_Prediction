# 📈 Stock Market Buy, Sell, and Hold Classification Project 📉

## Project Description

Welcome to the Stock Market Buy, Sell, and Hold Classification Project! This comprehensive project aims to leverage the power of machine learning and technical analysis to assist in making informed stock market decisions. Here's a detailed breakdown of the key components and functionalities of the project:

### 1. Data Processing
Data is the foundation of any analytical project. In this project, we ensure the stock data is clean, consistent, and ready for analysis. This involves:
- **Loading Data**: Reading stock market data from CSV files.
- **Cleaning Data**: Handling missing values, ensuring data types are correct, and formatting the 'Date' column.
- **Visualizing Data**: Creating initial plots to understand the stock price movements over time.

### 2. Technical Indicators
Technical indicators are mathematical calculations based on historical price, volume, or open interest information. They are used to predict future price movements. In this project, we calculate and visualize several key indicators:
- **Simple Moving Average (SMA)**: An average of stock prices over a specific period.
- **Exponential Moving Average (EMA)**: A weighted average of stock prices that gives more importance to recent prices.
- **Relative Strength Index (RSI)**: A momentum oscillator that measures the speed and change of price movements.
- **Moving Average Convergence Divergence (MACD)**: A trend-following momentum indicator that shows the relationship between two moving averages.
- **Average Directional Index (ADX)**: An indicator used to quantify the strength of a trend.

### 3. Buy, Sell, and Hold Calculations
Based on the calculated technical indicators, we generate actionable signals:
- **Buy Signals**: Indications to buy stocks when certain conditions are met (e.g., when the price crosses above the SMA).
- **Sell Signals**: Indications to sell stocks when other conditions are met (e.g., when the price crosses below the SMA).
- **Hold Signals**: Indications to hold the current stock positions if neither buy nor sell conditions are met.

### 4. Feature Scaling
Feature scaling is crucial for preparing the data for machine learning models. It involves:
- **Encoding Categorical Variables**: Converting categorical columns into numerical values.
- **Scaling Numerical Features**: Standardizing the range of independent variables to have a mean of 0 and a standard deviation of 1.

### 5. Feature Selection
Feature selection helps in identifying the most significant features that influence the target variable. Techniques used in this project include:
- **Feature Importance from Trees**: Using tree-based models to estimate the importance of each feature.
- **Univariate Feature Selection**: Selecting the best features based on univariate statistical tests.
- **Recursive Feature Elimination (RFE)**: Iteratively removing the least important features and building the model with the remaining features.

### 6. Model Development and Training
This is the core of the project, where we build and train machine learning models to predict stock actions. Steps involved:
- **Model Building**: Creating a Random Forest model with specified parameters.
- **Cross-Validation**: Evaluating the model using time series cross-validation to ensure robustness.
- **Training**: Fitting the model on the training data.
- **Evaluation**: Testing the model on unseen data and generating performance metrics like accuracy, classification report, and confusion matrix.

### 7. Streamlit Application
The project includes an interactive web application built with Streamlit, enabling users to:
- **Upload Stock Data**: Users can upload their own stock data CSV files.
- **Run Analysis**: Perform data processing, calculate technical indicators, generate signals, scale features, select features, and develop models.
- **View Predictions**: Get actionable buy, sell, or hold indications based on the trained model.

## Usage

### How to Use the Project Locally

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/stock-market-classification.git
    ```

2. **Navigate to the Project Directory**:
    ```bash
    cd stock-market-classification
    ```

3. **Install the Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```

5. **Open Your Web Browser**: Go to `http://localhost:8501` to view the application.

### Steps to Use the Application

1. **Upload Your Stock Data CSV File**: Ensure the file contains at least 5 years of data with a `Date` column.
2. **Run Analysis**:
    - Click the "Run Analysis" button to perform the following steps:
        - **Add Technical Indicators**: Calculate and plot various technical indicators.
        - **Generate Signals**: Generate buy, sell, and hold signals.
        - **Perform Feature Scaling**: Encode categorical columns and scale numeric features.
        - **Perform Feature Selection**: Select the most important features.
        - **Model Development and Training**: Develop and train the machine learning model.
3. **View Results**: The final indication (Buy, Sell, or Hold) will be displayed on the screen.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m 'Add some feature'
    ```
4. Push to the branch:
    ```bash
    git push origin feature/your-feature-name
    ```
5. Create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Enjoy using the Stock Market Buy, Sell, and Hold Classification Project! 📊🚀