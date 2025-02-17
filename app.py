import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Title
st.title("Electrical Distribution & Energy Efficiency Dashboard")

# Sidebar for file upload
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Energy Data", type=["csv", "xlsx"])

if uploaded_file:
    # Read the uploaded file
    if uploaded_file.name.endswith("csv"):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith("xlsx"):
        data = pd.read_excel(uploaded_file)
    
    st.write("Data Preview:")
    st.write(data.head())

    # Data Preprocessing
    data['Renewable energy share in the total final energy consumption (%)'] = pd.to_numeric(data['Renewable energy share in the total final energy consumption (%)'], errors='coerce')
    data['Electricity from fossil fuels (TWh)'] = pd.to_numeric(data['Electricity from fossil fuels (TWh)'], errors='coerce')
    data['Value_co2_emissions_kt_by_country'] = pd.to_numeric(data['Value_co2_emissions_kt_by_country'], errors='coerce')

    # Handle missing values
    data = data.dropna(subset=['Renewable energy share in the total final energy consumption (%)', 'Electricity from fossil fuels (TWh)', 'Value_co2_emissions_kt_by_country'])

    # Energy Usage Visualization
    if 'Year' in data.columns and 'Renewable energy share in the total final energy consumption (%)' in data.columns:
        st.subheader("Renewable Energy Share Over Time")
        plt.figure(figsize=(10, 5))
        plt.plot(data['Year'], data['Renewable energy share in the total final energy consumption (%)'], label='Renewable Energy Share')
        plt.xlabel("Year")
        plt.ylabel("Renewable Energy Share (%)")
        plt.title("Renewable Energy Share Trend")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Predictive Maintenance Section (Now focusing on CO2 emissions prediction)
    st.subheader("Predictive CO2 Emissions Model")
    if 'Renewable energy share in the total final energy consumption (%)' in data.columns and 'Electricity from fossil fuels (TWh)' in data.columns:
        # Feature selection for CO2 emissions prediction
        X = data[['Renewable energy share in the total final energy consumption (%)', 'Electricity from fossil fuels (TWh)']]  # Example features
        y = data['Value_co2_emissions_kt_by_country']  # Target variable

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Regressor
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Predictions and Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Model Mean Squared Error: {mse:.2f}")

        # Model performance
        st.write(f"Model R-squared: {model.score(X_test, y_test):.2f}")

        # Predict button
        st.subheader("Predict CO2 Emissions for New Data")
        renewable_share = st.number_input("Enter Renewable Energy Share (%)", min_value=0, max_value=100, value=30)
        fossil_fuels = st.number_input("Enter Electricity from Fossil Fuels (TWh)", min_value=0, value=100)

        prediction = model.predict([[renewable_share, fossil_fuels]])
        st.write(f"Predicted CO2 Emissions (kt): {prediction[0]:.2f}")

    # Energy Efficiency Suggestions Section
    st.subheader("Energy Efficiency Recommendations")
    if 'Renewable energy share in the total final energy consumption (%)' in data.columns:
        average_renewable_share = data['Renewable energy share in the total final energy consumption (%)'].mean()
        st.write(f"Average Renewable Energy Share: {average_renewable_share:.2f} %")
        
        # Example of energy saving tips based on renewable energy share
        if average_renewable_share < 25:
            st.write("Suggestion: Increase renewable energy sources to improve sustainability.")
        else:
            st.write("Suggestion: Continue to focus on maintaining or increasing renewable energy sources.")

# Additional Visualization Section
st.sidebar.subheader("Additional Visualizations")
visual_option = st.sidebar.selectbox("Choose Visualization Type", ["Renewable Energy Share Over Time", "CO2 Emissions Analysis"])

if visual_option == "Renewable Energy Share Over Time":
    if 'Year' in data.columns and 'Renewable energy share in the total final energy consumption (%)' in data.columns:
        st.subheader("Renewable Energy Share Over Time")
        plt.figure(figsize=(10, 5))
        sns.lineplot(x='Year', y='Renewable energy share in the total final energy consumption (%)', data=data)
        plt.xticks(rotation=45)
        st.pyplot(plt)
elif visual_option == "CO2 Emissions Analysis":
    if 'Value_co2_emissions_kt_by_country' in data.columns:
        st.subheader("CO2 Emissions Analysis")
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Year', y='Value_co2_emissions_kt_by_country', data=data)
        plt.xlabel("Year")
        plt.ylabel("CO2 Emissions (kt)")
        plt.title("CO2 Emissions Over Time")
        st.pyplot(plt)

# Footer
st.write("---")
st.write("This app is powered by Streamlit. Built for energy distribution optimization and predictive maintenance.")
