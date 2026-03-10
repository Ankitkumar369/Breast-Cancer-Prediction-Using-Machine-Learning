import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import traceback


@st.cache_data
def load_data():
    df = pd.read_csv("breast-cancer.csv")
    return df


@st.cache_resource
def load_models():
	with open("model1.pkl", "rb") as f:
		model1 = pickle.load(f)
	with open("model2.pkl", "rb") as f:
		model2 = pickle.load(f)
	return model1, model2


@st.cache_resource
def load_scaler(feature_df):
	scaler_path = "scaler.pkl"
	# If a saved scaler exists, load it. Otherwise fit a new scaler on provided features and return it.
	if os.path.exists(scaler_path):
		with open(scaler_path, "rb") as f:
			scaler = pickle.load(f)
		return scaler

	scaler = StandardScaler()
	scaler.fit(feature_df)
	# save scaler for future runs
	try:
		with open(scaler_path, "wb") as f:
			pickle.dump(scaler, f)
	except Exception:
		pass
	return scaler


def main():
	st.title("Breast Cancer Prediction App ")
	st.write("Simple  app to predict breast cancer (malignant vs benign) using pre-trained models.")

	df = load_data()
	# keep all features used during training (drop only the target column 'diagnosis')
	df_features = df.drop(["diagnosis"], axis=1)

	model1, model2 = load_models()

	# get cached scaler (loads scaler.pkl if present or fits-and-saves a new one)
	scaler = load_scaler(df_features)

	st.sidebar.header("Model & Input Options")
	model_choice = st.sidebar.selectbox("Choose model", ("Logistic Regression (model1)", "SVM (model2)"))
	use_sample = st.sidebar.checkbox("Use example from dataset", value=False)

	# prepare input values
	input_values = {}
	if use_sample:
		idx = st.sidebar.number_input("Sample row index", min_value=0, max_value=len(df_features)-1, value=0)
		sample = df_features.iloc[int(idx)]
		for col in df_features.columns:
			input_values[col] = float(sample[col])
	else:
		st.sidebar.write("Enter feature values")
		for col in df_features.columns:
			default = float(df_features[col].mean())
			val = st.sidebar.number_input(col, value=default, format="%f")
			input_values[col] = float(val)

	if st.sidebar.button("Predict"):
		X = np.array([list(input_values.values())], dtype=float)
		X_scaled = scaler.transform(X)

		if model_choice.startswith("Logistic"):
			model = model1
		else:
			model = model2

		pred = model.predict(X_scaled)[0]
		proba = None
		try:
			proba = model.predict_proba(X_scaled)[0]
		except Exception:
			proba = None

		label = "Malignant (Cancerous)" if int(pred) == 1 else "Benign (Not cancerous)"
		st.subheader("Prediction")
		st.write(f"**Result:** {label}")
		if proba is not None:
			if len(proba) == 2:
				confidence = max(proba) * 100
				st.write(f"**Confidence:** {confidence:.2f}%")
			else:
				st.write(f"**Probabilities:** {proba}")

	st.markdown("---")
	st.subheader("Dataset sample")
	st.dataframe(df.head())


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		tb = traceback.format_exc()
		# when running in Streamlit show the traceback in the app
		try:
			st.error("An error occurred while running the app. See details below:")
			st.text(tb)
		except Exception:
			# fallback to printing
			print(tb)
		raise


