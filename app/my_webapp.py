import streamlit as st
from PIL import Image
import numpy as np
import lime
from lime import lime_tabular
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import plotly.express as px

st.set_page_config(layout="wide")

data = pd.read_csv("diabetes_prediction_dataset.csv")

smoking_history_mapping = {'not current' : 'former', 'ever' : 'current', 'No Info' : 'unrevealed'}
data['smoking_history'] = data['smoking_history'].replace(smoking_history_mapping)
data['smoking_history'] = data['smoking_history'].apply(lambda x: x.split()[0])

#converting bmi of people over 40, because the weight can be in pounds which needs to be converted into kgs for homogenity
bmi_validity = data['bmi'] > 40
data.loc[bmi_validity, 'bmi'] = data.loc[bmi_validity, 'bmi'] / 2.205

df = pd.get_dummies(data)
y= df['diabetes']
feats= df.drop('diabetes', axis=1)
copied_feats=feats.copy()

pickle_in = open("LGBMBoost.pkl", "rb")
LGBMBoost = pickle.load(pickle_in)

tabs = ["Dashboard", "Prediction"]
active_tab = st.selectbox("**Navigation**", tabs)
st.markdown("---")

# Increase the size of the navigation options
# st.sidebar.markdown("<style>div.row-widget.stRadio > div{font-size: 30px !important;}</style>", unsafe_allow_html=True)

if active_tab == "Dashboard":
	st.markdown("<h1 style='text-align: center;'>Analysis of Diabetes Risk Factors</h1>", unsafe_allow_html=True)
	data['age_group'] = pd.cut(data['age'], bins=[0, 19, 40, 60, float('inf')], labels=['under 19', '19-40', '40-60', 'over 60'])
	
	grouped_data = data[data['diabetes'] == 1].groupby(['age_group', 'gender']).size().unstack()

	# Create a stacked bar plot
	fig, axs = plt.subplots(3, 2, figsize=(16, 16))
	fig.patch.set_facecolor('darkgray')

	axs[0, 1].bar(grouped_data.index, grouped_data['Male'], label='Male')
	axs[0, 1].bar(grouped_data.index, grouped_data['Female'],bottom=grouped_data['Male'], label='Female')
	#axs[0, 0].bar(grouped_data.index, grouped_data['Other'], label='Other')
	
	# Set labels and title
	axs[0, 1].set_xlabel('Gender')
	axs[0, 1].set_ylabel('Frequency of People with Diabetes')
	axs[0, 1].set_title('Frequency of People with Diabetes by Gender and Age Group')
	axs[0, 1].legend()

	plt.show()

	# Graph 2 showing relationship between BMI and Blood Glucose Level
	axs[2, 1].scatter(data['bmi'], data['blood_glucose_level'])
	axs[2, 1].set_xlabel('BMI')
	axs[2, 1].set_ylabel('Blood Glucose Level')
	axs[2, 1].set_title('Correlation between BMI and Blood Glucose Level')

	# Show the plots
	plt.show()

	selected_columns = ['heart_disease', 'hypertension', 'diabetes']
	subset_data = data[selected_columns]

	# Calculate the correlation matrix
	correlation_matrix = subset_data.corr()


	# Plot the correlation matrix on the desired subplot
	im = axs[2, 0].imshow(correlation_matrix, cmap='Blues')

	# Add correlation values as text annotations
	for i in range(len(correlation_matrix.columns)):
		for j in range(len(correlation_matrix.columns)):
			axs[2, 0].text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha="center", va="center",
						color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black")

	# Set axis labels
	axs[2, 0].set_xticks(range(len(correlation_matrix.columns)))
	axs[2, 0].set_yticks(range(len(correlation_matrix.columns)))
	axs[2, 0].set_xticklabels(correlation_matrix.columns)
	axs[2, 0].set_yticklabels(correlation_matrix.columns)

	# Set title and colorbar
	axs[2, 0].set_title('Correlation Matrix: Hypertension vs Diabetes')
	cbar = fig.colorbar(im, ax=axs[1, 0])

	# Adjust spacing between subplots
	plt.tight_layout()

	# Display the plot
	plt.show()

	# Group data by smoking history and diabetes status
	grouped_data2 = data.groupby(['smoking_history', 'diabetes']).size().unstack()

	ax = plt.subplot(3, 2, 3)

	# Create the stacked bar chart
	grouped_data2.plot(kind='bar', stacked=False, ax=ax)

	# Set labels and title
	ax.set_xlabel('Smoking History')
	ax.set_ylabel('Frequency')
	ax.set_title('Distribution of Diabetes Status by Smoking History')

	# Select data for individuals with and without diabetes
	diabetes_data = data[data['diabetes'] == 1]
	non_diabetes_data = data[data['diabetes'] == 0]

	# Prepare the data for box plot
	box_plot_data = [diabetes_data['bmi'], non_diabetes_data['bmi']]
	labels = ['Diabetes', 'Non-Diabetes']

	
	# Create the box plot
	axs[1, 1].boxplot(box_plot_data, labels=labels)

	# Set labels and title
	axs[1, 1].set_xlabel('Diabetes Status')
	axs[1, 1].set_ylabel('BMI')
	axs[1, 1].set_title('Distribution of BMI: Diabetes vs Non-Diabetes')

	# Display the plot
	plt.tight_layout()
	plt.show()

	ax = plt.subplot(3, 2, 1)

	# Plot a bar plot or a pie chart
	data['diabetes2'] = data['diabetes'].replace({0: 'Non-Diabetic', 1: 'Diabetic'})
	diabetes_count = data['diabetes2'].value_counts()

	# Pie chart
	diabetes_count.plot(kind='pie', ax=ax, autopct='%1.1f%%')
	ax.set_ylabel('')
	ax.set_title('Prevalence of Diabetes')
	ax.legend(['Non-Diabetic', 'Diabetic'])

	# Show the plot
	plt.tight_layout()
	plt.show()

	# Adjust spacing between subplots
	fig.tight_layout()

	# Display the plot using st.pyplot()
	st.pyplot(fig)




if active_tab == "Prediction":

	age=0
	hypertension=0
	heart_disease=0
	bmi=0
	HbA1c_level=0
	blood_glucose_level=0
	gender_Female=0
	gender_Male=0
	gender_Other=0
	smoking_history_current=0
	smoking_history_former=0
	smoking_history_never=0
	smoking_history_unrevealed=0


	st.title('This form will predict your chances of getting Diabetes')
	st.write('This web app predicts the chances of a patient getting Diabetes. \
			You need to fill in the form below, and then the predictor will perform its prediction')

	age = st.slider(
	label='Age',
	min_value=2,
	max_value=80,
	value=2,
	step=1
	)

	genderrad = st.radio('Gender', ('Male', 'Female', 'Other'))
	if genderrad == 'Male':
		gender_Male = 1
	elif genderrad == 'Female':
		gender_Female = 1
	else:
		gender_Other = 1

	bmi = st.slider(
	label= 'BMI',
	min_value=10.0,
	max_value=45.0,
	value=10.0,
	step=0.1
	)

	smoking_historyrad = st.radio('Do you smoke?', ('Yes', 'Used to', 'No', 'Do not want to share'))
	if smoking_historyrad == 'Yes':
		smoking_history_current = 1
	elif smoking_historyrad == 'Used to':
		smoking_history_former = 1
	elif smoking_historyrad == 'No':
		smoking_history_never = 1
	else:
		smoking_history_unrevealed = 1

	heart_diseaserad = st.radio('Did you suffer from Heart Disease?', ('Yes', 'No'))
	if heart_diseaserad == 'Yes':
		heart_disease = 1
	else:
		heart_disease = 0

	hypertension = st.radio('Do you suffer from Hypertension?', ('Yes', 'No'))
	if hypertension == 'Yes':
		hypertension = 1
	else:
		hypertension = 0

	HbA1c_level = st.slider(
	label='Average Blood Glucose (HbA1c) level',
	min_value=0.0,
	max_value=10.0,
	value=0.0,
	step=0.1
	)

	blood_glucose_level = st.slider(
	label='Blood Glucose Level',
	min_value=80,
	max_value=300,
	value=80,
	step=1
	)

	features = {
	'age': age,
	'hypertension': hypertension,
	'heart_disease': heart_disease,
	'bmi': bmi,
	'HbA1c_level': HbA1c_level,
	'blood_glucose_level': blood_glucose_level,
	'gender_Female': gender_Female,
	'gender_Male': gender_Male,
	'gender_Other': gender_Other,
	'smoking_history_current': smoking_history_current,
	'smoking_history_former': smoking_history_former,
	'smoking_history_never': smoking_history_never,
	'smoking_history_unrevealed': smoking_history_unrevealed
	}

	model = lgb.LGBMClassifier(subsample=1.0, num_leaves=80, min_child_samples=20, max_depth=5, learning_rate=0.1, colsample_bytree=1.0)
	model.fit(feats, y)
	prediction = model.predict([list(features.values())])

	def run_lime_prediction():
		explainer = lime.lime_tabular.LimeTabularExplainer(feats.values, feature_names=feats.columns, class_names=['0', '1'])

		# Select an instance from the test data for explanation
		instance = copied_feats.iloc[len(copied_feats)-1]

		# Explain the prediction for the selected instance
		explanation = explainer.explain_instance(instance.values, model.predict_proba, num_features=len(feats.columns))

		# Plot the Lime prediction graph
		fig = explanation.as_pyplot_figure()

		# Display the Lime plot in Streamlit using matplotlib's figure
		st.subheader('LIME Prediction')
		st.pyplot(fig)

	if st.button("Predict"):
		if prediction == 1:
			st.write('Prediction: You are more likely to have diabetes')
		else:
			st.write('Prediction: You are less likely to have diabetes')
		copied_feats = copied_feats.append(features, ignore_index=True)
		run_lime_prediction()
