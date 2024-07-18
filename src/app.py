import streamlit as st
import pandas as pd
from src.main import DataProcessor, SMA, EMA, RSI, MACD, ADX, BuySellHold, FeatureScaling, FeatureSelection
from src.model import  ModelDevelopment
from streamlit_extras.let_it_rain import rain
class StockMarketAnalysis:
    def __init__(self, file_path=None, data=None):
        self.data_processor = DataProcessor(file_path, data)
        self.stock_data = self.data_processor.stock_data
        self.sma_days = [5, 10, 20, 50, 100, 200]
        self.sma_crossover_periods = [(5, 20), (20, 50), (50, 200)]
        self.ema_days = [9, 12, 26, 50, 200]
        self.ema_crossover_periods = [(9, 12), (12, 26), (50, 200)]

    def sma(self):
        sma = SMA(self.stock_data)
        sma.add_sma_columns(self.sma_days)
        sma.plot_sma(self.sma_days)
        crossover_data = sma.calculate_sma_crossovers(self.sma_crossover_periods)
        sma.plot_sma_crossovers(crossover_data, 5, 20)
        sma.plot_sma_crossovers(crossover_data, 20, 50)
        sma.plot_sma_crossovers(crossover_data, 50, 200)


    def ema(self):
        ema = EMA(self.stock_data)
        ema.add_ema_columns(self.ema_days)
        ema.plot_ema(self.ema_days)
        crossover_data = ema.calculate_ema_crossovers(self.ema_crossover_periods)
        ema.plot_ema_crossovers(crossover_data, 9, 12)
        ema.plot_ema_crossovers(crossover_data, 12, 26)
        ema.plot_ema_crossovers(crossover_data, 50, 200)


    def rsi(self):
        rsi = RSI(self.stock_data)
        rsi.add_rsi_column(14)
        rsi.plot_rsi()

    
    def macd(self):
        macd = MACD(self.stock_data)
        macd.calculate_macd()
        macd.plot_macd()



    def adx(self):
        adx = ADX(self.stock_data)
        adx.calculate_adx()
        adx.plot_adx()

        
    def add_technical_indicators(self):
        self.sma()
        self.ema()
        self.rsi()
        self.macd()
        self.adx()

    def generate_signals(self):
        self.signals = BuySellHold(self.stock_data)
        self.signals.generate_signals()
        self.signals.visualize_signals()

    def perform_feature_scaling(self):
        self.feature_scaling = FeatureScaling(self.stock_data)

    def feature_selection(self):
        self.feature_selection = FeatureSelection(self.stock_data)

    def model_development(self):
        self.model_development = ModelDevelopment(self.stock_data)
        self.model_development.run()
        
    
def main():
    st.title("Stock Market Buy/Sell/Hold Prediction")
    st.info(' Please upload your stock data CSV file containing at least 5 years of data.', icon="⚠️")
    
    uploaded_file = st.file_uploader("Upload your stock data CSV file", type=["csv"])

    if uploaded_file is not None:
        # Check file size
        file_size_kb = uploaded_file.size / 1024
        if file_size_kb < 85:
            st.error('File size must be greater than 85 KB. Please upload a larger file.', icon="🚨")
            return

        # Read the data
        stock_data = pd.read_csv(uploaded_file)

        # Check if the data contains at least 5 years
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        min_date = stock_data['Date'].min()
        max_date = stock_data['Date'].max()
        date_diff = (max_date - min_date).days / 365

        if date_diff < 5:
            st.error("The data must contain at least 5 years of stock prices. Please upload a valid dataset.")
            return
        
        st.subheader("Uploaded Data:")
        st.write(stock_data)

        # Perform analysis
        analysis = StockMarketAnalysis(data=stock_data)

        # Button to run analysis
        if st.button("Run Analysis"):
            with st.spinner("Adding technical indicators..."):
                analysis.add_technical_indicators()
                st.success("Technical Indicators added successfully.")

            with st.spinner("Generating signals..."):
                analysis.generate_signals()
                st.success("Signals generated successfully.")

            with st.spinner("Performing feature scaling..."):
                analysis.perform_feature_scaling()
                st.success("Feature scaling completed.")

            with st.spinner("Performing feature selection..."):
                analysis.feature_selection()
                st.success("Feature selection completed.")

            with st.spinner("Developing and training the model..."):
                model_dev = ModelDevelopment(stock_data)
                indication = model_dev.run()
                # # Store the indication in session state
                # st.session_state['indication'] = indication
            st.divider()
            if indication == "Buy":    
               st.info(f"Indication: {indication}", icon="💹") #  Buy: 📈 or 💹 or 🟢     Sell: 📉 or 🔻 or 🔴     Hold: ⏸️ or 🤚 or 🟡
            elif indication == "Sell":
               st.info(f"Indication: {indication}", icon="📉")
            else:
                st.info(f"Indication: {indication}", icon="⏸️")   
            rain(
                emoji= f"🎈",  
                font_size=54,
                falling_speed=6,
                animation_length="5",
            )
            st.success("Model Development and Training completed.")

if __name__ == "__main__":
    main()