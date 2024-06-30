import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Pima Indians Diabetes Dataset
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv", 
                   header=None,
                   names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                          "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])

# Feature matrix and target variable
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build and train the k-NN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Sidebar for navigation
section = st.sidebar.radio("Navigate", ["Home", "Dataset Info", "Visualization", "Prediction", "EDA"])

# Home section
if section == "Home":
    st.markdown("<h1 style='text-align: center;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>Use the sidebar to navigate between sections for viewing data, visualizations, and predictions.</p>",
        unsafe_allow_html=True,
    )
    image_path = "C:\\Users\\ADMIN\\anaconda3\\diab.jpg"
    st.image(image_path, caption="Welcome to the Diabetes Prediction App", use_column_width=True)

elif section == "Prediction":
    st.title("Predict Diabetes")

    with st.form("prediction_form"):
        cols = st.columns(2)
        user_data = {}
        
        for i, feature in enumerate(X.columns):
            if i % 2 == 0:
                user_data[feature] = cols[0].text_input(f"Enter {feature}")
            else:
                user_data[feature] = cols[1].text_input(f"Enter {feature}")

        submitted = st.form_submit_button("Submit")

        if submitted:
            user_input = np.array([float(val) for val in user_data.values()]).reshape(1, -1)
            user_input_scaled = scaler.transform(user_input)

            prediction = knn.predict(user_input_scaled)

            if prediction[0] == 1:
                st.markdown(
                    "<div style='background-color: red; padding: 10px; color: white; text-align: center;'>Prediction Result: Diabetes</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='background-color: green; padding: 10px; color: white; text-align: center;'>Prediction Result: No Diabetes</div>",
                    unsafe_allow_html=True,
                )

# Visualization section
elif section == "Visualization":
    st.title("Data Visualization and Model Evaluation")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.write("Model Accuracy:", accuracy)
    st.write("Classification Report:")
    st.text(report)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    fig, ax = plt.subplots()
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, alpha=0.5, label="Train")
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, marker="x")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("PCA Visualization")
    st.pyplot(fig)
    plt.close(fig)

# Dataset Info section
elif section == "Dataset Info":
    st.title("Dataset Overview")

    st.markdown(
        """
        **Dataset Description:**
        The Pima Indians Diabetes Dataset contains health metrics and demographic information for Pima Indian women,
        used to predict the risk of diabetes.
        """
    )

    st.write("First Few Rows:")
    st.dataframe(data.head(10))
    st.write("Dataset Shape:", data.shape)
    st.write("Descriptive Statistics:")
    st.write(data.describe())
    st.write("Null Values in the Dataset:")
    st.write(data.isnull().sum())

# EDA section
elif section == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    # Distribution of diabetes risk levels
    st.subheader("Distribution of Diabetes Risk Levels")
    fig = plt.figure(figsize=(8, 6))
    sns.countplot(x='Outcome', data=data, palette='coolwarm')
    plt.title("Count of Diabetes Outcomes")
    plt.xlabel("Outcome (0: No Diabetes, 1: Diabetes)")
    plt.ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)  # Explicitly close the figure

    # Correlations between features and diabetes risk
    st.subheader("Correlations Between Features")
    fig = plt.figure(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, linecolor='white')
    plt.title("Feature Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

    # Trends in patient data
    st.subheader("Trends in Patient Data")
    features_to_explore = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

    for feature in features_to_explore:
        fig = plt.figure(figsize=(8, 6))
        sns.histplot(data[feature], kde=True, color='blue')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(x='Outcome', y=feature, data=data, palette='coolwarm')
        plt.title(f"{feature} vs Outcome")
        plt.xlabel("Outcome (0: No Diabetes, 1: Diabetes)")
        plt.ylabel(feature)
        st.pyplot(fig)
        plt.close(fig)

