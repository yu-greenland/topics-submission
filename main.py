import numpy as np
import pandas as pd
import os
from scipy.stats import variation
from scipy import stats
import conduct_analysis as ca
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
Functions that 
"""
def whitwell_study(data: pd.DataFrame, x_axis: str):
	""" Named after the Whitwell study because it applies the proportion adjustment method on given volumes. 

	Args:
		data: a DataFrame containing brain structure volumes
		x_axis: a string defining what the x-axis for the generated plots should be. Either 'TIV' or 'Age'.
	
	Returns:
		Generates density plots for every brain structure in the given data. Additionally, creates a bar plot of the coefficient of variation for every brain structure. 
	"""

	# conduct proportion analysis with adjustment factor as TIV and VOI as Volume_of_brain
	tbv_cov = ca.conduct_proportion_analysis(data, 28, 11, x_axis)

	# conduct proportion analysis with adjustment factor as "scaling" and VOI as Volume_of_brain
	# ca.conduct_proportion_analysis(data, 5, 11, x_axis)

	# adjustment factor TIV and VOI is Volume_of_thalamus_avg
	thalamus_cov = ca.conduct_proportion_analysis(data, 28, 44, x_axis)

	# adjustment factor TIV and VOI is Volume_of_caudate_avg
	caudate_cov = ca.conduct_proportion_analysis(data, 28, 45, x_axis)

	# adjustment factor TIV and VOI is Volume_of_putamen_avg
	putamen_cov = ca.conduct_proportion_analysis(data, 28, 46, x_axis)

	# adjustment factor TIV and VOI is Volume_of_pallidum_avg
	pallidum_cov = ca.conduct_proportion_analysis(data, 28, 47, x_axis)

	# adjustment factor TIV and VOI is Volume_of_hippocampus_avg
	hippocampus_cov = ca.conduct_proportion_analysis(data, 28, 48, x_axis)

	# adjustment factor TIV and VOI is Volume_of_amygdla_avg
	amygdla_cov = ca.conduct_proportion_analysis(data, 28, 49, x_axis)

	# adjustment factor TIV and VOI is Volume_of_accumbens_avg
	accumbens_cov = ca.conduct_proportion_analysis(data, 28, 50, x_axis)

	# plots the coefficients of variation as a bar plot
	cov_values = [tbv_cov, thalamus_cov, putamen_cov, hippocampus_cov, caudate_cov, pallidum_cov, amygdla_cov, accumbens_cov]
	# cov_values = [0.22, 0.41, 0.11, 0.3, 0.1, 0.45, 0.6, 0.23]
	x = np.arange(8)
	fig, ax = plt.subplots(figsize=(16,5))
	plt.bar(x, cov_values, width=0.8)
	plt.xticks(x, ('TBV', 'Thalamus', 'Putamen', 'Hippocampus', 'Caudate', 'Pallidum', 'Amyglda', 'Accumbens'))
	plt.title("coefficients of variation (proportion adjusted)")
	if x_axis == 'Age':
		plt.savefig("whitwell_age_cov_value_plots.pdf")
	else:
		plt.savefig("whitwell_cov_value_plots.pdf")
	plt.close()

def liu_study(data: pd.DataFrame, x_axis: str):
	"""Named after the Liu study because it applies the power-proportion adjustment method on given volumes. 

	Args:
		data: a DataFrame containing brain structure volumes
		x_axis: a string defining what the x-axis for the generated plots should be. Either 'TIV' or 'Age'.
	
	Returns:
		Generates density plots for every brain structure in the given data. Creates a bar plot of the coefficient of variation for every brain structure. Creates a bar plot of the fitted beta value of every brain structure. 
	"""
	# adjustment factor TIV and VOI is Volume_of_brain
	tbv_beta, tbv_cov = ca.conduct_power_proportion_analysis(data, 28, 11, x_axis)

	# adjustment factor TIV and VOI is Volume_of_thalamus_avg
	thalamus_beta, thalamus_cov = ca.conduct_power_proportion_analysis(data, 28, 44, x_axis)

	# adjustment factor TIV and VOI is Volume_of_caudate_avg
	caudate_beta, caudate_cov = ca.conduct_power_proportion_analysis(data, 28, 45, x_axis)

	# adjustment factor TIV and VOI is Volume_of_putamen_avg
	putamen_beta, putamen_cov = ca.conduct_power_proportion_analysis(data, 28, 46, x_axis)

	# adjustment factor TIV and VOI is Volume_of_pallidum_avg
	pallidum_beta, pallidum_cov = ca.conduct_power_proportion_analysis(data, 28, 47, x_axis)

	# adjustment factor TIV and VOI is Volume_of_hippocampus_avg
	hippocampus_beta, hippocampus_cov = ca.conduct_power_proportion_analysis(data, 28, 48, x_axis)

	# adjustment factor TIV and VOI is Volume_of_amygdla_avg
	amygdla_beta, amygdla_cov = ca.conduct_power_proportion_analysis(data, 28, 49, x_axis)

	# adjustment factor TIV and VOI is Volume_of_accumbens_avg
	accumbens_beta, accumbens_cov = ca.conduct_power_proportion_analysis(data, 28, 50, x_axis)

	# produces two bar plots of the fitted beta values and the value of coefficient of variation
	# brain structures in order of volume: TBV, thalamus, putamen, hippocampus, caudate, pallidum, amygdla, accumbens

	# plots the beta values as a bar plot
	beta_values = [tbv_beta, thalamus_beta, putamen_beta, hippocampus_beta, caudate_beta, pallidum_beta, amygdla_beta, accumbens_beta]
	x = np.arange(8)
	fig, ax = plt.subplots(figsize=(16,5))
	plt.bar(x, beta_values, width=0.8)
	plt.xticks(x, ('TBV', 'Thalamus', 'Putamen', 'Hippocampus', 'Caudate', 'Pallidum', 'Amyglda', 'Accumbens'))
	plt.title("fitted beta value plots (power-proportion adjusted)")
	plt.savefig("liu_beta_value_plots.pdf")
	plt.close()

	# plots the coefficients of variation as a bar plot
	cov_values = [tbv_cov, thalamus_cov, putamen_cov, hippocampus_cov, caudate_cov, pallidum_cov, amygdla_cov, accumbens_cov]
	x = np.arange(8)
	fig, ax = plt.subplots(figsize=(16,5))
	plt.bar(x, cov_values, width=0.8)
	plt.xticks(x, ('TBV', 'Thalamus', 'Putamen', 'Hippocampus', 'Caudate', 'Pallidum', 'Amyglda', 'Accumbens'))
	plt.title("coefficients of variation (power-proportion adjusted)")
	if x_axis == 'Age':
		plt.savefig("liu_age_cov_value_plots.pdf")
	else:
		plt.savefig("liu_cov_value_plots.pdf")
	plt.close()

	# ----------------- plots for left volumes of interest -----------#

	# # adjustment factor TIV and VOI is Volume_of_thalamus_left
	# ca.conduct_power_proportion_analysis(data, 28, 12, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_caudate_left
	# ca.conduct_power_proportion_analysis(data, 28, 14, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_putamen_left
	# ca.conduct_power_proportion_analysis(data, 28, 16, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_pallidum_left
	# ca.conduct_power_proportion_analysis(data, 28, 18, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_hippocampus_left
	# ca.conduct_power_proportion_analysis(data, 28, 20, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_amygdala_left
	# ca.conduct_power_proportion_analysis(data, 28, 22, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_accumbens_left
	# ca.conduct_power_proportion_analysis(data, 28, 24, 'TIV')

	# ----------------- plots for right volumes of interest -----------#

	# # adjustment factor TIV and VOI is Volume_of_thalamus_right
	# ca.conduct_power_proportion_analysis(data, 28, 13, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_caudate_right
	# ca.conduct_power_proportion_analysis(data, 28, 15, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_putamen_right
	# ca.conduct_power_proportion_analysis(data, 28, 17, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_pallidum_right
	# ca.conduct_power_proportion_analysis(data, 28, 19, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_hippocampus_right
	# ca.conduct_power_proportion_analysis(data, 28, 21, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_amygdala_right
	# ca.conduct_power_proportion_analysis(data, 28, 23, 'TIV')

	# # adjustment factor TIV and VOI is Volume_of_accumbens_right
	# ca.conduct_power_proportion_analysis(data, 28, 25, 'TIV')

def glm_study(data: pd.DataFrame, x_axis: str):
	"""Named after the General Linear Model because it applies the glm adjustment method on given volumes. 

	Args:
		data: a DataFrame containing brain structure volumes
		x_axis: a string defining what the x-axis for the generated plots should be. Either 'TIV' or 'Age'.
	
	Returns:
		Generates density plots for every brain structure in the given data. Additionally, creates a bar plot of the coefficient of variation for every brain structure. 
	"""
	# conduct GLM. features are sex, BMI, age and TIV. trying to predict Volume_of_brain
	tbv_cov = ca.conduct_GLM(data, [0, 3, 4, 28], 11, x_axis)

	# conduct GLM. features are sex, BMI, age and TIV. trying to predict Volume_of_thalamus_avg
	thalamus_cov = ca.conduct_GLM(data, [0, 3, 4, 28], 44, x_axis)

	# conduct GLM. features are sex, BMI, age and TIV. trying to predict Volume_of_caudate_avg
	caudate_cov = ca.conduct_GLM(data, [0, 3, 4, 28], 45, x_axis)

	# conduct GLM. features are sex, BMI, age and TIV. trying to predict Volume_of_putamen_avg
	putamen_cov = ca.conduct_GLM(data, [0, 3, 4, 28], 46, x_axis)

	# conduct GLM. features are sex, BMI, age and TIV. trying to predict Volume_of_pallidum_avg
	pallidum_cov = ca.conduct_GLM(data, [0, 3, 4, 28], 47, x_axis)

	# conduct GLM. features are sex, BMI, age and TIV. trying to predict Volume_of_hippocampus_avg
	hippocampus_cov = ca.conduct_GLM(data, [0, 3, 4, 28], 48, x_axis)

	# conduct GLM. features are sex, BMI, age and TIV. trying to predict Volume_of_amygdla_avg
	amygdla_cov = ca.conduct_GLM(data, [0, 3, 4, 28], 49, x_axis)

	# conduct GLM. features are sex, BMI, age and TIV. trying to predict Volume_of_accumbens_avg
	accumbens_cov = ca.conduct_GLM(data, [0, 3, 4, 28], 50, x_axis)

	# plots the coefficients of variation as a bar plot
	cov_values = [tbv_cov, thalamus_cov, putamen_cov, hippocampus_cov, caudate_cov, pallidum_cov, amygdla_cov, accumbens_cov]
	x = np.arange(8)
	fig, ax = plt.subplots(figsize=(16,5))
	plt.bar(x, cov_values, width=0.8)
	plt.xticks(x, ('TBV', 'Thalamus', 'Putamen', 'Hippocampus', 'Caudate', 'Pallidum', 'Amyglda', 'Accumbens'))
	plt.title("coefficients of variation (general linear model adjusted)")
	if x_axis == 'Age': 
		plt.savefig("glm_age_cov_value_plots.pdf")
	else:
		plt.savefig("glm_cov_value_plots.pdf")
	plt.close()

def classifier_test(data: pd.DataFrame):
	"""Runs a classifier test as another way of measuring effectiveness of adjustment methods.

	Args:
		data: data: a DataFrame containing brain structure volumes
	
	Returns: outputs in terminal the accuracy of each adjustment method when passed through a classfier
	"""
	# classifying using Total Brain Volume as the Volume of Interest
	ca.classify_on_Age(data, 11)




"""
Data manipulation functions
"""
def normalise_TIV(data: pd.DataFrame) -> pd.DataFrame:
	"""Normalises the Total Intracranial Volume column. This is so that when conducting proportion and power-proportion adjustments the adjusted volumes are in the same order of magnitude. Since Total Intracranial Volume is normally the adjustment factor used.

	Args:
		data: a DataFrame containing brain structure volumes
	
	Returns:
		a DataFrame with the TIV column normalised
	"""
	# first extract the TIV column
	tiv_column = ca.data_splitter_single_column(data, 28)
	# find the mean of TIV
	col_mean = tiv_column['TIV'].mean()
	# divide each row in the column by the mean, normalising it
	tiv_column = tiv_column.div(col_mean)
	# replace the original TIV column with normalised TIV
	data = data.assign(TIV=tiv_column['TIV'])
	return data

def remove_TIV_outliers(data: pd.DataFrame) -> pd.DataFrame:
	"""Removes any rows where the Total Intracranial Volume is an outlier. Outliers are classifed as being more tha 3 std away from the mean.

	Args:
		data: a DataFrame containing brain structure volumes
	
	Returns:
		a DataFrame with rows where the TIV is an outlier removed
	"""
	z_score = np.abs(stats.zscore(data['TIV']))
	outlier_idx = np.where(z_score > 3)
	outliers = pd.Series(outlier_idx[0]) 

	return data.drop(data.index[[outliers]])

def create_averaged_data(data: pd.DataFrame) -> pd.DataFrame:
	""" Combines the volumes that have a left part and the volumes that have a right part into a single column. Is done for indexes (12, 13), (14,15), (16,17), (18,19), (20,21), (22,23) and (24,25)

	Args:
		data: a DataFrame containing brain structure volumes
	
	Returns:
		the original DataFrame with the averaged volumes appended to the end
	"""
	col = data.loc[: , "Volume_of_thalamus_left":"Volume_of_thalamus_right"]
	data['Volume_of_thalamus_avg'] = col.mean(axis=1)
	col = data.loc[: , "Volume_of_caudate_left":"Volume_of_caudate_right"]
	data['Volume_of_caudate_avg'] = col.mean(axis=1)
	col = data.loc[: , "Volume_of_putamen_left":"Volume_of_putamen_right"]
	data['Volume_of_putamen_avg'] = col.mean(axis=1)
	col = data.loc[: , "Volume_of_pallidum_left":"Volume_of_pallidum_right"]
	data['Volume_of_pallidum_avg'] = col.mean(axis=1)
	col = data.loc[: , "Volume_of_hippocampus_left":"Volume_of_hippocampus_right"]
	data['Volume_of_hippocampus_avg'] = col.mean(axis=1)
	col = data.loc[: , "Volume_of_amygdala_left":"Volume_of_amygdala_right"]
	data['Volume_of_amygdala_avg'] = col.mean(axis=1)
	col = data.loc[: , "Volume_of_accumbens_left":"Volume_of_accumbens_right"]
	data['Volume_of_accumbens_avg'] = col.mean(axis=1)
	return data

def process_data(path_to_file: str) -> pd.DataFrame:
	"""Given the path to a csv file containing brain structure data, processes the data so adjustments to brain structures can be performed.

	Args:
		path_to_file: a string defining the file path to a csv
	
	Returns:
		a pandas DataFrame with fields processed
	"""
	# loads csv into a pandas data frame
	data = pd.read_csv(path_to_file)
	# drop all columns that contain NaN values
	data = data.dropna(axis='columns')
	# creates averaged brain structures
	data = create_averaged_data(data)
	# remove outliers
	data = remove_TIV_outliers(data)
	# normalise the TIV
	data = normalise_TIV(data)
	return data




"""
Main function
"""
def main():

	"""
	To-do list
	"""

	data = process_data("./allvols2.csv")


	# conduct the Whitwell study with ___ as the x_axis
	whitwell_study(data, 'TIV')

	# conduct the Liu study with ___ as the x_axis
	liu_study(data, 'TIV')

	# use glm as adjustment method with ___ as the x_axis
	# defaults to using sex, age, bmi and TIV as features
	glm_study(data, 'TIV')

	# as another measure of adjustment effectiveness
	# classifier accuracy is used
	classifier_test(data)


if __name__ == "__main__":
    main()