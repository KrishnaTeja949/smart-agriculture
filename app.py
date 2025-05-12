# === app.py ===
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Load models and encoders
crop_model = joblib.load('crop_model.pkl')
yield_model = joblib.load('yield_model.pkl')
fert_model = joblib.load('fert_model.pkl')
le_crop = joblib.load('le_crop.pkl')
le_fert = joblib.load('le_fert.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title("🌾 Smart Agriculture Assistant")

# Session state for input persistence and chart toggle
if 'input_done' not in st.session_state:
    st.session_state.input_done = False
if 'show_charts' not in st.session_state:
    st.session_state.show_charts = False
if 'user_df' not in st.session_state:
    st.session_state.user_df = pd.DataFrame()

if not st.session_state.input_done:
    st.header("🔢 Enter Soil and Weather Parameters")
    user_input = {}
    for col in feature_names:
        user_input[col] = st.number_input(f"Enter value for {col}", value=0.0, key=col)
    user_df = pd.DataFrame([user_input])

    if st.button("🚀 Predict Crop, Yield & Fertilizer"):
        st.session_state.user_df = user_df
        st.session_state.input_done = True
        st.experimental_rerun()
else:
    user_df = st.session_state.user_df
    pred_crop = le_crop.inverse_transform(crop_model.predict(user_df))[0]
    pred_yield = yield_model.predict(user_df)[0]
    pred_fert = le_fert.inverse_transform(fert_model.predict(user_df))[0]

    st.success(f"🧾 **Recommended Crop**: {pred_crop}")
    st.info(f"📈 **Predicted Yield**: {pred_yield:.2f} quintals/hectare")
    st.warning(f"💡 **Recommended Fertilizer**: {pred_fert}")

    if st.button("🔁 Enter New Values"):
        st.session_state.input_done = False
        st.experimental_rerun()

# === Toggle Charts Button ===
if st.button("📊 Toggle Evaluation Graphs"):
    st.session_state.show_charts = not st.session_state.show_charts

if st.session_state.show_charts:
    st.subheader("📈 Model Evaluation Metrics")

    # Hardcoded evaluation results
    hardcoded_results = [
        {"Technique": "RFE", "Model": "Naive Bayes", "Accuracy": 98.82, "Precision": 98.87, "Recall": 98.82, "F1-Score": 98.82},
        {"Technique": "RFE", "Model": "Decision Tree", "Accuracy": 98.82, "Precision": 98.87, "Recall": 98.82, "F1-Score": 98.81},
        {"Technique": "RFE", "Model": "SVM", "Accuracy": 97.23, "Precision": 97.74, "Recall": 97.23, "F1-Score": 97.20},
        {"Technique": "RFE", "Model": "Random Forest", "Accuracy": 99.32, "Precision": 99.36, "Recall": 99.32, "F1-Score": 99.32},
        {"Technique": "RFE", "Model": "KNN", "Accuracy": 97.45, "Precision": 97.64, "Recall": 97.45, "F1-Score": 97.46},

        {"Technique": "Boruta", "Model": "Naive Bayes", "Accuracy": 99.32, "Precision": 99.35, "Recall": 99.32, "F1-Score": 99.32},
        {"Technique": "Boruta", "Model": "Decision Tree", "Accuracy": 98.55, "Precision": 98.63, "Recall": 98.55, "F1-Score": 98.54},
        {"Technique": "Boruta", "Model": "SVM", "Accuracy": 97.50, "Precision": 97.98, "Recall": 97.50, "F1-Score": 97.46},
        {"Technique": "Boruta", "Model": "Random Forest", "Accuracy": 99.32, "Precision": 99.36, "Recall": 99.32, "F1-Score": 99.32},
        {"Technique": "Boruta", "Model": "KNN", "Accuracy": 97.95, "Precision": 98.12, "Recall": 97.95, "F1-Score": 97.95},

        {"Technique": "SMOTE", "Model": "Naive Bayes", "Accuracy": 99.32, "Precision": 99.35, "Recall": 99.32, "F1-Score": 99.32},
        {"Technique": "SMOTE", "Model": "Decision Tree", "Accuracy": 98.45, "Precision": 98.54, "Recall": 98.45, "F1-Score": 98.45},
        {"Technique": "SMOTE", "Model": "SVM", "Accuracy": 97.50, "Precision": 97.98, "Recall": 97.50, "F1-Score": 97.46},
        {"Technique": "SMOTE", "Model": "Random Forest", "Accuracy": 99.27, "Precision": 99.32, "Recall": 99.27, "F1-Score": 99.27},
        {"Technique": "SMOTE", "Model": "KNN", "Accuracy": 98.05, "Precision": 98.20, "Recall": 98.05, "F1-Score": 98.05},

        {"Technique": "ROSE", "Model": "Naive Bayes", "Accuracy": 99.36, "Precision": 99.40, "Recall": 99.36, "F1-Score": 99.36},
        {"Technique": "ROSE", "Model": "Decision Tree", "Accuracy": 98.68, "Precision": 98.74, "Recall": 98.68, "F1-Score": 98.68},
        {"Technique": "ROSE", "Model": "SVM", "Accuracy": 97.50, "Precision": 97.98, "Recall": 97.50, "F1-Score": 97.46},
        {"Technique": "ROSE", "Model": "Random Forest", "Accuracy": 99.36, "Precision": 99.40, "Recall": 99.36, "F1-Score": 99.36},
        {"Technique": "ROSE", "Model": "KNN", "Accuracy": 97.95, "Precision": 98.12, "Recall": 97.95, "F1-Score": 97.95}
    ]

    results_df = pd.DataFrame(hardcoded_results)
    if not all(col in results_df.columns for col in ["Technique", "Model"]):
        st.error("Evaluation data is malformed. Please check the format of hardcoded_results.")
    else:
        melted = results_df.melt(
            id_vars=["Technique", "Model"],
            value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
            var_name="Metric",
            value_name="Score"
        )

        for tech in melted["Technique"].unique():
            tech_data = melted[melted["Technique"] == tech].copy()
            st.subheader(f"📊 {tech} - Metric Comparison")

            fig_bar, ax_bar = plt.subplots(figsize=(14, 8))
            sns.barplot(data=tech_data, x="Model", y="Score", hue="Metric", ax=ax_bar)
            ax_bar.set_title(f"{tech} - Bar Plot")
            st.pyplot(fig_bar)

            for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                fig_line, ax_line = plt.subplots(figsize=(10, 5))
                metric_data = tech_data[tech_data["Metric"] == metric]
                sns.lineplot(data=metric_data, x="Model", y="Score", marker="o", ax=ax_line)
                ax_line.set_title(f"{tech} - {metric} Line Plot")
                st.pyplot(fig_line)

        for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
            pivot_df = results_df.pivot(index="Model", columns="Technique", values=metric)
            sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax_heat)
            ax_heat.set_title(f"Heatmap of {metric} by Model & Technique")
            st.pyplot(fig_heat)
