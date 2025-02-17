import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title
st.title("Electrical Distribution & Energy Efficiency Dashboard")

# Sidebar for file upload
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Energy Data", type=["csv", "xlsx"])

data = None  # Initialize data as None

if uploaded_file:
    # Read the uploaded file
    if uploaded_file.name.endswith("csv"):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith("xlsx"):
        data = pd.read_excel(uploaded_file)
    
    st.write("Data Preview:")
    st.write(data.head())

# Ensure that data exists before proceeding with visualizations
if data is not None:
    # Energy Usage Visualization
    if 'timestamp' in data.columns and 'energy_usage' in data.columns:
        st.subheader("Energy Consumption Over Time")
        plt.figure(figsize=(10, 5))
        plt.plot(data['timestamp'], data['energy_usage'], label='Energy Usage')
        plt.xlabel("Timestamp")
        plt.ylabel("Energy Usage (kWh)")
        plt.title("Energy Usage Trend")
        plt.xticks(rotation=45)
        st.pyplot(plt)
    
    # Predictive Maintenance Section
    st.subheader("Predictive Maintenance Model")
    if 'equipment_age' in data.columns and 'failure' in data.columns:
        # Feature selection for predictive maintenance
        X = data[['equipment_age', 'usage_hours', 'previous_failures']]  # Example features
        y = data['failure']  # Target variable

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Classifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Predictions and Accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Predict button
        st.subheader("Predict Maintenance for New Equipment")
        equipment_age = st.number_input("Enter Equipment Age", min_value=0, max_value=50, value=5)
        usage_hours = st.number_input("Enter Usage Hours", min_value=0, max_value=10000, value=500)
        previous_failures = st.number_input("Enter Previous Failures", min_value=0, max_value=10, value=1)

        prediction = model.predict([[equipment_age, usage_hours, previous_failures]])
        if prediction[0] == 1:
            st.write("Prediction: Equipment will fail soon. Maintenance is recommended.")
        else:
            st.write("Prediction: Equipment is functioning well. No immediate maintenance required.")
    
    # Energy Efficiency Suggestions Section
    st.subheader("Energy Efficiency Recommendations")
    if 'energy_usage' in data.columns:
        average_energy_usage = data['energy_usage'].mean()
        st.write(f"Average Energy Usage: {average_energy_usage:.2f} kWh")
        
        # Example of energy saving tips based on usage
        if average_energy_usage > 500:
            st.write("Suggestion: Consider optimizing energy distribution and equipment performance.")
        else:
            st.write("Suggestion: Energy usage is within the expected range. Continue monitoring.")

    # Additional Visualization Section
    st.sidebar.subheader("Additional Visualizations")
    visual_option = st.sidebar.selectbox("Choose Visualization Type", ["Energy Usage Over Time", "Equipment Failure Analysis"])

    if visual_option == "Energy Usage Over Time":
        if 'timestamp' in data.columns and 'energy_usage' in data.columns:
            st.subheader("Energy Usage Over Time")
            plt.figure(figsize=(10, 5))
            sns.lineplot(x='timestamp', y='energy_usage', data=data)
            plt.xticks(rotation=45)
            st.pyplot(plt)
    elif visual_option == "Equipment Failure Analysis":
        if 'failure' in data.columns:
            st.subheader("Failure Count")
            failure_counts = data['failure'].value_counts()
            sns.barplot(x=failure_counts.index, y=failure_counts.values)
            plt.xlabel("Failure Status")
            plt.ylabel("Count")
            plt.title("Equipment Failure Analysis")
            st.pyplot(plt)

else:
    st.warning("Please upload a dataset to proceed with the analysis.")

# Footer
st.write("---")
st.write("This app is powered by Streamlit. Built for energy distribution optimization and predictive maintenance.")
