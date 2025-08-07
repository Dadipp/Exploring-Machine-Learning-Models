import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_FnUseC_TelcoCustomerChurn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

def preprocess_data(df):
    df = df.drop(['customerID'], axis=1)
    for col in df.select_dtypes(include='object').columns:
        if col != 'Churn':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def main():
    st.title("Telco Customer Churn Prediction")
    st.write("This app uses machine learning to predict customer churn.")

    df = load_data()
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Model Evaluation")
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
