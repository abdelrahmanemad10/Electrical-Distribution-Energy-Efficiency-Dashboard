import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

    # Energy Consumption Over Time
    if 'Year' in data.columns and 'Renewable-electricity-generating-capacity-per-capita' in data.columns:
        st.subheader("Renewable Energy Capacity Over Time")
        plt.figure(figsize=(10, 5))
        plt.plot(data['Year'], data['Renewable-electricity-generating-capacity-per-capita'], label='Renewable Energy Capacity per Capita')
        plt.xlabel("Year")
        plt.ylabel("Renewable Energy Capacity (kW/capita)")
        plt.title("Renewable Energy Capacity Trend")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Energy Efficiency Recommendations
    st.subheader("Energy Efficiency Recommendations")
    if 'Renewable energy share in the total final energy consumption (%)' in data.columns:
        renewable_share = data['Renewable energy share in the total final energy consumption (%)'].mean()
        st.write(f"Average Renewable Energy Share: {renewable_share:.2f}%")
        
        if renewable_share < 20:
            st.write("Suggestion: Consider increasing the share of renewable energy sources to improve sustainability.")
        else:
            st.write("Suggestion: Renewable energy share is significant. Continue with monitoring.")

    # Predictive Maintenance Section
    st.subheader("Predictive Maintenance Model for CO2 Emissions")
    if 'Renewable energy share in the total final energy consumption (%)' in data.columns and 'Value_co2_emissions_kt_by_country' in data.columns:
        # Feature selection for predicting CO2 emissions
        X = data[['Renewable energy share in the total final energy consumption (%)', 'Electricity from fossil fuels (TWh)']]  # Example features
        y = data['Value_co2_emissions_kt_by_country']  # Target variable

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Regressor (since we're predicting a continuous value)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Predictions and Accuracy
        st.write(f"Model R-squared: {model.score(X_test, y_test):.2f}")

        # Predict CO2 emissions for new data
        st.subheader("Predict CO2 Emissions for New Data")
        renewable_share_input = st.number_input("Enter Renewable Energy Share (%)", min_value=0, max_value=100, value=renewable_share)
        fossil_fuel_electricity_input = st.number_input("Enter Electricity from Fossil Fuels (TWh)", min_value=0, value=0)

        prediction = model.predict([[renewable_share_input, fossil_fuel_electricity_input]])
        st.write(f"Predicted CO2 Emissions: {prediction[0]:.2f} kt")

    # CO2 Emissions by Country
    if 'Entity' in data.columns and 'Value_co2_emissions_kt_by_country' in data.columns:
        st.subheader("CO2 Emissions by Country")
        co2_emissions = data[['Entity', 'Value_co2_emissions_kt_by_country']].sort_values(by='Value_co2_emissions_kt_by_country', ascending=False).head(10)
        sns.barplot(x='Value_co2_emissions_kt_by_country', y='Entity', data=co2_emissions)
        plt.xlabel("CO2 Emissions (kt)")
        plt.ylabel("Country")
        plt.title("Top 10 Countries by CO2 Emissions")
        st.pyplot(plt)

    # Additional Visualization Section
    st.sidebar.subheader("Additional Visualizations")
    visual_option = st.sidebar.selectbox("Choose Visualization Type", ["Energy Usage Over Time", "CO2 Emissions by Country"])

    if visual_option == "Energy Usage Over Time":
        if 'Year' in data.columns and 'Renewable-electricity-generating-capacity-per-capita' in data.columns:
            st.subheader("Energy Usage Over Time")
            plt.figure(figsize=(10, 5))
            sns.lineplot(x='Year', y='Renewable-electricity-generating-capacity-per-capita', data=data)
            plt.xticks(rotation=45)
            st.pyplot(plt)
    elif visual_option == "CO2 Emissions by Country":
        if 'Entity' in data.columns and 'Value_co2_emissions_kt_by_country' in data.columns:
            st.subheader("CO2 Emissions by Country")
            co2_emissions = data[['Entity', 'Value_co2_emissions_kt_by_country']].sort_values(by='Value_co2_emissions_kt_by_country', ascending=False).head(10)
            sns.barplot(x='Value_co2_emissions_kt_by_country', y='Entity', data=co2_emissions)
            plt.xlabel("CO2 Emissions (kt)")
            plt.ylabel("Country")
            plt.title("Top 10 Countries by CO2 Emissions")
            st.pyplot(plt)

# Footer
st.write("---")
st.write("This app is powered by Streamlit. Built for energy distribution optimization and predictive maintenance.")
