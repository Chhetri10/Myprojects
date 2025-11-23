import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import datetime
import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
# The following code is based on the logic developed in the Jupyter Notebook cells.
# We will load the data, perform feature engineering, and train the model.
# This entire process is encapsulated here for a self-contained Streamlit app.


# --- Page Configuration and CSS ---
st.set_page_config(
    page_title="NEPSE Stock Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
        /* Custom Font (Google Fonts) */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        /* Background with gradient */
        .main {
            background: linear-gradient(135deg, #e0f7fa, #fce4ec);
            padding: 10px;
        }

        /* Sidebar Styling */
        .st-emotion-cache-16txtl3 {
            background-color: #ffffff;
            border-right: 1px solid #ddd;
        }

        /* Sidebar Button Styling */
        section[data-testid="stSidebar"] .stButton button {
            background-color: #f4f4f4;
            border: 2px solid transparent;
            border-radius: 12px;
            padding: 12px 20px;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
            transition: 0.3s ease;
        }

        /* Hover effect for buttons */
        section[data-testid="stSidebar"] .stButton button:hover {
            background-color: #dbeafe;
            color: #0d47a1;
            border: 2px solid #90caf9;
        }

        /* Active button style */
        section[data-testid="stSidebar"] .stButton button.active {
            background-color: #2196f3;
            color: #fff;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        /* Metric cards */
        .st-emotion-cache-1f1figzo, .st-emotion-cache-ocqkz7 {
            background: white;
            border-radius: 16px;
            padding: 24px;
            border: 1px solid #ccc;
            box-shadow: 2px 6px 15px rgba(0,0,0,0.05);
        }

        /* Chart Styling */
        .stPlotlyChart {
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_preprocess_training_data():
    """Loads and preprocesses the full merged training dataset."""
    try:
        df = pd.read_csv('C:/My projects/data/merged_stock_with_index1.csv')
    except FileNotFoundError:
        st.error("Error: The file 'merged_stock_with_index1.csv' was not found. Please upload it.")
        st.stop()
        return None

    # Clean NEPSE Index (remove commas and convert to float)
    df['NEPSE Index'] = df['NEPSE Index'].astype(str).str.replace(',', '', regex=False)
    df['NEPSE Index'] = pd.to_numeric(df['NEPSE Index'], errors='coerce')

    # Convert date
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

    # Drop unwanted column
    df.drop(columns=['Conf.'], errors='ignore', inplace=True)

    # Sort and fill missing values within each symbol
    df.sort_values(by=['Symbol', 'date'], inplace=True)
    df = df.groupby('Symbol').apply(lambda x: x.ffill().bfill()).reset_index(drop=True)

    # Feature Engineering
    df['Prev_Close'] = df.groupby('Symbol')['Close'].shift(1)
    df['Prev_VWAP'] = df.groupby('Symbol')['VWAP'].shift(1)
    df['MA5'] = df.groupby('Symbol')['Close'].rolling(window=5).mean().reset_index(level=0, drop=True)
    df['MA10'] = df.groupby('Symbol')['Close'].rolling(window=10).mean().reset_index(level=0, drop=True)
    df['Close_Change'] = df['Close'] - df['Prev_Close']
    df['Close_Change_%'] = (df['Close_Change'] / df['Prev_Close']) * 100

    # Drop rows where any critical feature is missing
    required_cols = ['Prev_Close', 'Prev_VWAP', 'MA5', 'MA10', 'Close_Change', 'Close_Change_%', 'NEPSE Index']
    df.dropna(subset=required_cols, inplace=True)

    # Rename columns for consistency
    df.rename(columns={'Vol': 'Volume'}, inplace=True)
    df['Date'] = df['date']  # duplicate column for plotting

    # Final debug print
    print("✅ Final processed data shape:", df.shape)
    print("✅ Sample data:\n", df[required_cols + ['Symbol', 'date']].head())

    if df.empty:
        st.error("The preprocessed training data is empty. Please check the data file.")
        st.stop()

    return df


@st.cache_data

def load_latest_prediction_data():
    """Loads the most recent stock CSV file from the 'data' directory for prediction."""
    data_dir = 'data'
    if not os.path.isdir(data_dir):
        st.error(f"Error: The directory '{data_dir}' was not found. Please create it and add your daily CSV files.")
        st.stop()
        
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        st.error(f"No CSV files found in the '{data_dir}' directory. Please add your daily CSV files.")
        st.stop()
    
    latest_file = max(csv_files, key=os.path.getmtime)
    
    try:
        df = pd.read_csv(latest_file)
        df['Symbol'] = df['Symbol'].str.strip()
        filename = os.path.basename(latest_file)
        date_str = filename.replace('.csv', '')  # "2025_08_03"
        date_obj = datetime.datetime.strptime(date_str, "%Y_%m_%d").date()
        print(date_obj)
        # ✅ Add 'date' column to all rows
        df['date'] = pd.to_datetime(date_obj)

        # Optional: clean NEPSE Index if present
        if 'NEPSE Index' in df.columns:
            df['NEPSE Index'] = df['NEPSE Index'].astype(str).str.replace(',', '', regex=False).astype(float)

        return df, latest_file, date_obj
    except Exception as e:
        st.error(f"Failed to load or process the latest data file: {latest_file}. Error: {e}")
        st.stop()


@st.cache_resource
def train_model(df):
    """
    Trains and returns an XGBoost model with hyperparameter tuning.
    Also returns the fitted scaler.
    """
    # Define features and target
    features_to_use = [
        'Prev_Close', 'Prev_VWAP', 'MA5', 'MA10',
        'Close_Change', 'Close_Change_%', 'NEPSE Index'
    ]
    target_column = 'Close'

    X = df[features_to_use]
    y = df[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Hyperparameter tuning with RandomizedSearchC
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.2),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.5),
        }

        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            **params
        )

        score = cross_val_score(model, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=3)
        return score.mean()

    # Run the study
    study = optuna.create_study(direction='maximize')  # since neg_mean_squared_error is maximized for better results
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Train best model
    best_params = study.best_params
    xgb_regressor = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        **best_params
    )
    xgb_regressor.fit(X_train_scaled, y_train)
    return xgb_regressor, scaler, features_to_use


# Load and train
df_train = load_and_preprocess_training_data()
xgb_model, scaler, features = train_model(df_train)

# Later when predicting
df_latest, file_used, data_date = load_latest_prediction_data()

# --- Main App Logic ---
df_full = load_and_preprocess_training_data()
if df_full is not None:
    # print("df_full.head()", df_full.head())
    # print("df_full.tail()", df_full.tail())
    model, scaler, features_to_use = train_model(df_full)
    
    st.sidebar.title("NEPSE Predictor")
    pages = {
        "Home": "🏠",
        "Prediction": "🔮",
        "Charts": "📈"
    }
    # Dark mode toggle (using session state)
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False



    # Apply dark mode class to the app
    if st.session_state.dark_mode:
        st.markdown("<body class='dark-mode'>", unsafe_allow_html=True)
    else:
        st.markdown("<body>", unsafe_allow_html=True)


    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Home"

    for page, icon in pages.items():
        if st.sidebar.button(f"{icon} {page}", key=page):
            st.session_state.selected_page = page

    selected_page = st.session_state.selected_page

    # --- JavaScript to highlight active button ---
    st.markdown(f"""
        <script>
            var buttons = window.parent.document.querySelectorAll('section[data-testid="stSidebar"] .stButton button');
            var selectedPage = "{selected_page}";
            buttons.forEach(function(btn) {{
                var btnText = btn.innerText.split(' ').slice(1).join(' ');
                if (btnText === selectedPage) {{
                    btn.classList.add('active');
                }} else {{
                    btn.classList.remove('active');
                }}
            }});
        </script>
    """, unsafe_allow_html=True)

    # --- Page Content ---
    if selected_page == "Home":
        
        st.markdown(
        "<h1 style='text-align: center;'> Welcome to the NEPSE Stock Predictor</h1>",
        unsafe_allow_html=True
    )
    
        # latest_date = df_full['date'].max().strftime('%d/%m/%Y')
        _,_,latest_date = load_latest_prediction_data()
        st.markdown(f"<h4 style='text-align: center;'>Latest Date: {latest_date}</h4>", unsafe_allow_html=True)

        st.markdown("""
        <div style=" padding: 25px; border-radius: 10px; border: 1px solid #E0E0E0; margin-bottom: 20px;">
            <p>This application provides tools to analyze and predict stock prices from the Nepal Stock Exchange (NEPSE).</p>
            <p>Navigate through the sidebar to explore different features:</p>
            <ul>
                <li><strong>Prediction:</strong> Select a stock to predict its closing price for the next trading day using an XGBoost machine learning model.</li>
                <li><strong>Charts:</strong> Visualize historical price action and the prediction on an interactive candlestick chart.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        df_latest = df_full[df_full['date'] == df_full['date'].max()].copy()
        df_latest['Diff'] = df_latest['Close'] - df_latest['Prev_Close']
        df_latest['Diff %'] = (df_latest['Diff'] / df_latest['Prev_Close']) * 100
        
        features_for_model = ['Prev_Close', 'Prev_VWAP', 'MA5', 'MA10', 'Close_Change', 'Close_Change_%', 'NEPSE Index']
        X_today_scaled = scaler.transform(df_latest[features_for_model])
        st.subheader(f"Top Predicted Movers for the Next Trading Week")
    
    # --- Define Sector Lists ---
        commercial_banks = ['ADBL', 'CZBIL', 'EBL', 'GBIME', 'HBL', 'KBL', 'NABIL', 'NBL', 'NICA', 'NMB', 'PCBL', 'SANIMA', 'SBI', 'SBL', 'SCB', 'PRVU', 'NIMB', 'LSL']
        development_banks = ['CORBL', 'EDBL', 'GRDBL', 'GBBL', 'JBBL', 'KSBBL', 'LBBL', 'MLBL', 'MDB', 'MNBBL', 'NABBC', 'NIDC', 'SADBL', 'SHINE', 'SINDU', 'SAPDBL']
        finance_companies = ['BFC', 'CFCL', 'GFCL', 'GMFIL', 'GUFL', 'HATHY', 'ICFC', 'JFL', 'MFIL', 'MPFL', 'NFS', 'PFL', 'PROFL', 'RLFL', 'SFCL', 'SIFC', 'SWMF', 'SWBBL']
        microfinance = ['ACLBSL', 'ALBSL', 'CBBL', 'CLBSL', 'DDBL', 'FMDBL', 'FOWAD', 'GMFBS', 'GILB', 'GBLBS', 'GLBSL', 'ILBS', 'JSLBB', 'JBLB', 'KMCDB', 'KLBSL', 'LLBS', 'MLBSL', 'MLBBL', 'NADEP', 'NMFBS', 'NMBMF', 'RSDC', 'SAMAJ', 'SMATA', 'SLBSL', 'SKBBL', 'SWBBL', 'SMFBS', 'USLB', 'VLBS', 'WOMI', 'NSLB', 'NUBL']
        life_insurance = ['ALICL', 'LICN', 'NLIC', 'NLICL', 'CLI', 'RNLI', 'ILI', 'SNLI', 'PLI', 'SJLIC', 'SRLI', 'HLI', 'PMLI']
        non_life_insurance = ['AIL', 'EIC', 'GIC', 'HGI', 'IGI', 'LGIL', 'NIL', 'NICL', 'NLG', 'PRIN', 'PIC', 'PICL', 'RBCL', 'SIC', 'SGIC', 'SICL', 'SIL', 'UIC']
        hydropower = ['AKJCL', 'API', 'AKPL', 'AHPC', 'BARUN', 'BPCL', 'CHCL', 'CHL', 'DHPL', 'GHL', 'GLH', 'HDHPC', 'HURJA', 'HPPL', 'JOSHI', 'KKHC', 'LEC', 'MEN', 'MHNL', 'NHPC', 'NHDL', 'NGPL', 'PMHPL', 'PPCL', 'RADHI', 'RHPL', 'RHPC', 'RURU', 'SHPC', 'SJCL', 'SHEL', 'SSHL', 'SPDL', 'UNHPL', 'UMRH', 'UMHL', 'UPCL', 'UPPER', 'CKHL', 'MMKJL', 'DOLTI']
        manufacturing = ['AVU', 'AVYAN', 'BNL', 'BNT', 'HDL', 'SHIVM', 'UNL', 'GCIL', 'MKCL']
        hotel_tourism = ['CGH', 'OHL', 'SHL', 'TRH', 'YHL']
        trading_others = ['BBC', 'STC', 'NTC', 'NRIC', 'HRL', 'NRM', 'NWCL']

        allowed_symbols = (commercial_banks + development_banks + finance_companies +
                        microfinance + life_insurance + non_life_insurance +
                        hydropower + manufacturing + hotel_tourism + trading_others)

    # --- Prediction Logic for Home Page ---
        df_filtered = df_latest[df_latest['Symbol'].isin(allowed_symbols)].copy()
        X_filtered_scaled = scaler.transform(df_filtered[features_for_model])
        df_filtered['Predicted_Close'] = model.predict(X_filtered_scaled)
        df_filtered['Predicted_Diff'] = df_filtered['Predicted_Close'] - df_filtered['Close']
        df_filtered['Predicted_Diff_%'] = (df_filtered['Predicted_Diff'] / df_filtered['Close']) * 100

        # ✅ NEW: Deduplicate by symbol
        df_filtered = df_filtered.sort_values(by='Date', ascending=False)
        df_filtered = df_filtered.drop_duplicates(subset='Symbol', keep='first')

# ... (your previous code is correct) ...

        # Sort for gainers/losers
        top_gainers = df_filtered[df_filtered['Predicted_Diff_%'] > 0] \
            .sort_values(by='Predicted_Diff_%', ascending=False).head(5)

        top_losers = df_filtered[df_filtered['Predicted_Diff_%'] < 0] \
            .sort_values(by='Predicted_Diff_%').head(5)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h5>🚀 Top 5 Gainers</h5>", unsafe_allow_html=True)
            # ✅ CHANGE: Use the PREDICTED difference columns
            gainers_display = top_gainers[['Symbol', 'Close', 'Predicted_Diff', 'Predicted_Diff_%']].rename(
                # ✅ CHANGE: Rename the PREDICTED columns for display
                columns={'Symbol': 'Ticker', 'Close': 'Price', 'Predicted_Diff': 'Change', 'Predicted_Diff_%': '% Change'}
            )
            st.dataframe(
                gainers_display.style.format({
                    'Price': '{:.2f}', 'Change': '{:+.2f}', '% Change': '{:+.2f}%'
                }).map(lambda x: 'color: #2ca02c', subset=['Change', '% Change']),
                use_container_width=True, hide_index=True
            )

        with col2:
            st.markdown("<h5>📉 Top 5 Losers</h5>", unsafe_allow_html=True)
            # ✅ CHANGE: Use the PREDICTED difference columns
            losers_display = top_losers[['Symbol', 'Close', 'Predicted_Diff', 'Predicted_Diff_%']].rename(
                # ✅ CHANGE: Rename the PREDICTED columns for display
                columns={'Symbol': 'Ticker', 'Close': 'Price', 'Predicted_Diff': 'Change', 'Predicted_Diff_%': '% Change'}
            )
            st.dataframe(
                losers_display.style.format({
                    'Price': '{:.2f}', 'Change': '{:+.2f}', '% Change': '{:+.2f}%'
                }).map(lambda x: 'color: #d62728', subset=['Change', '% Change']),
                use_container_width=True, hide_index=True
            )

        st.markdown("""
        <div style=" padding: 25px; border-radius: 10px; border: 1px solid #E0E0E0; margin-top: 20px;">
            <p><strong>Disclaimer:</strong> This is a tool for educational and informational purposes only. Stock market predictions are inherently uncertain. Do not use this as financial advice.</p></div>
        """, unsafe_allow_html=True)

    elif selected_page == "Prediction":
        st.title("Predict Tomorrow's Close Price")
        st.markdown("Select a stock symbol to get the next day's predicted closing price based on the latest available data.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            symbols = df_full['Symbol'].unique()
            selected_symbol = st.selectbox("Select Stock Symbol", symbols, label_visibility="collapsed")
            predict_button = st.button("Predict Price", use_container_width=True, type="primary")

        if predict_button and selected_symbol:
            latest_row = df_full[df_full['Symbol'] == selected_symbol].iloc[-1]
            # print(latest_row)
            latest_features = latest_row[features_to_use].values.reshape(1, -1)
            # print(latest_features)
            # Scale the latest data point
            latest_features_scaled = scaler.transform(latest_features)
            
            # Make the prediction
            prediction = model.predict(latest_features_scaled)[0]
            
            with col2:
                st.metric(
                    label=f"Predicted Close for {selected_symbol}",
                    value=f"NPR {prediction:.2f}",
                    delta=f"{prediction - latest_row['Close']:.2f} vs Latest Close",
                    delta_color="normal"
                )

            st.subheader("Feature Inputs for the Prediction")
            values = latest_row[features_to_use]
            m1, m2, m3 = st.columns(3)
            m1.metric("Previous Close", f"{values['Prev_Close']:.2f}")
            m1.metric("Previous VWAP", f"{values['Prev_VWAP']:.2f}")
            m2.metric("5-Day Moving Avg.", f"{values['MA5']:.2f}")
            m2.metric("10-Day Moving Avg.", f"{values['MA10']:.2f}")
            m3.metric("Price Change", f"{values['Close_Change']:.2f}")
            m3.metric("Price Change %", f"{values['Close_Change_%']:.2f}%")

    elif selected_page == "Charts":
        st.title("Historical Price & Prediction Chart")
        st.markdown("Visualize the historical price movement and see the model's prediction for the next day.")

        symbols = df_full['Symbol'].unique()
        selected_symbol_chart = st.selectbox("Select Symbol for Chart", symbols)

        df_chart = df_full[df_full['Symbol'] == selected_symbol_chart].copy()
        
        if not df_chart.empty:
            # Get the latest row for prediction
            latest_row = df_chart.iloc[-1]
            latest_features = latest_row[features_to_use].values.reshape(1, -1)
            latest_features_scaled = scaler.transform(latest_features)
            prediction = model.predict(latest_features_scaled)[0]

            # --- Custom Candlestick Chart with Dark Mode Theme ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                subplot_titles=(f'Price Chart for {selected_symbol_chart}', 'Volume'),
                                row_heights=[0.7, 0.3])

            fig.add_trace(go.Candlestick(
                x=df_chart['Date'],
                open=df_chart['Open'],
                high=df_chart['High'],
                low=df_chart['Low'],
                close=df_chart['Close'],
                name='Price',
                increasing_line_color='#00ff99',
                decreasing_line_color='#ff4c4c'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df_chart['Date'],
                y=df_chart['Close'],
                mode='lines',
                name='Close',
                line=dict(color='#00b4d8', width=1.5)
            ), row=1, col=1)

            last_date = df_chart['Date'].iloc[-1]
            prediction_date = last_date + pd.Timedelta(days=1)
            fig.add_trace(go.Scatter(
                x=[last_date, prediction_date],
                y=[df_chart['Close'].iloc[-1], prediction],
                mode='lines+markers',
                name='Prediction',
                line=dict(color='#ffc107', dash='dash'),
                marker=dict(symbol='star', size=12, color='#ffc107')
            ), row=1, col=1)

            volume_colors = ['#66ff66' if r['Close'] > r['Open'] else '#ff6666' for _, r in df_chart.iterrows()]
            fig.add_trace(go.Bar(
                x=df_chart['Date'],
                y=df_chart['Volume'],
                name='Volume',
                marker_color=volume_colors
            ), row=2, col=1)

            fig.update_layout(
                xaxis_rangeslider_visible=False,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#ffffff")),
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='#ffffff')
            )

            fig.update_yaxes(title_text="Price (NPR)", row=1, col=1, gridcolor="#2e3a59")
            fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#2e3a59")
            fig.update_xaxes(
                rangebreaks=[dict(bounds=["fri", "sun"])], 
                tickformat='%Y/%m/%d',      
                gridcolor="#2e3a59"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No historical data available for {selected_symbol_chart}.")

