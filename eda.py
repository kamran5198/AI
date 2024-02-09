# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px

# Set Streamlit app title and description
st.title("Exploratory Data Analysis Web App 📊")
st.sidebar.header("Data Upload")
# Function to allow users to upload their own data or select sample data
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    sample_data = st.sidebar.radio("Select Sample Dataset", ("tips", "titanic"))
    if sample_data == "tips":
        data = sns.load_dataset("tips")
    elif sample_data == "titanic":
        data = sns.load_dataset("titanic")

# Sidebar checkboxes to choose visualizations
visualization_options = st.sidebar.multiselect("Select Visualizations", ["Histogram", "Bar Plot"])

# Sidebar checkbox to perform EDA
if st.sidebar.checkbox("Perform EDA"):
    st.subheader("Exploratory Data Analysis (EDA)")

    # Display summary statistics for numeric columns
    st.write("Summary Statistics:")
    st.write(data.describe())

    # Select columns for analysis
    columns = st.multiselect("Select Columns for Analysis", data.columns)
    selected_data = data[columns]

    # Generate selected visualizations
    if "Histogram" in visualization_options:
        for column in selected_data.select_dtypes(include=["number"]).columns:
            st.subheader(f"Histogram for {column}")
            plt.hist(selected_data[column], bins=20)
            plt.xlabel(column)
            plt.ylabel("Frequency")
            st.pyplot()
    
    if "Bar Plot" in visualization_options:
        for column in selected_data.select_dtypes(exclude=["number"]).columns:
            st.subheader(f"Bar Plot for {column}")
            sns.set_style("whitegrid")
            plt.figure(figsize=(8, 6))
            sns.countplot(x=column, data=selected_data, palette="Set3")
            plt.xticks(rotation=45)
            plt.xlabel(column)
            plt.ylabel("Count")
            st.pyplot()


# Sidebar checkbox to perform machine learning
if st.sidebar.checkbox("Perform Machine Learning"):
    st.subheader("Machine Learning")

    # Select target variable for prediction
    target_variable = st.selectbox("Select Target Variable for Prediction", data.columns)

    # Split the data into features (X) and target variable (y)
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]

    # One-hot encode categorical variables
    X = pd.get_dummies(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose the machine learning model
    model_option = st.selectbox("Select Machine Learning Model", ["Logistic Regression", "Random Forest", "Decision Tree"])

    if model_option == "Logistic Regression":
        # Logistic Regression
        model = LogisticRegression()
        model.fit(X_train, y_train)
         # Display coefficient values
        st.subheader("Coefficient Values")
        coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
        st.table(coefficients)

    elif model_option == "Random Forest":
        # Random Forest
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

    elif model_option == "Decision Tree":
        # Decision Tree
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
    
    elif model_option == "Linear Regression":
        # Linear Regression for multivariate regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Display coefficient values
        st.subheader("Coefficient Values")
        coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
        st.table(coefficients)

    # Make predictions
    predictions = model.predict(X_test)

    # Display model evaluation metrics
    st.subheader("Model Evaluation Metrics")
    st.write("Accuracy Score:", accuracy_score(y_test, predictions))
    st.write("Confusion Matrix:\n", confusion_matrix(y_test, predictions))


    # Display Classification Report in a table
    classification_rep = classification_report(y_test, predictions, output_dict=True)
    st.table(pd.DataFrame(classification_rep).transpose())


# Run the Streamlit app
if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)



