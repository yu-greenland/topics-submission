import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileMerger
import os
import seaborn as sns
from scipy.stats import variation
from statistics import variance
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mdutils.mdutils import MdUtils
import markdown
from weasyprint import HTML
from matplotlib.pyplot import figure
from sklearn.preprocessing import StandardScaler
import main
import pickle



"""
Save outputs to binary file for adjust_x.py to access without csv
"""




"""
PDF generation functions
"""
def export_pdf(pdf_name: str, data_frame: pd.DataFrame, adjust_method: str):
	"""Plots data supplied and saves the results as a PDF

	Args:
		pdf_name: a string describing the data that is to be plotted
		data_frame: a DataFrame, should be two columns, the first column representing the x-axis nd the second column representing the y-axis of the data to be plotted
		adjust_method: a string describing what adjustment method has been used, this is used for naming the plot
	
	Returns: a PDF with name described by pdf_string and adjust_method
	"""

	# put data into Series
	x = data_frame.iloc[:,0].to_numpy()
	y = data_frame.iloc[:,1].to_numpy()

	# from my very extensive testing the sci-kit learn linear regression model and the numpy.polyfit function prodcue the same outputs, either can be used
	reg = LinearRegression()
	x1 = x.reshape(-1, 1)
	y1 = y.reshape(-1, 1)

	reg.fit(x1, y1)

	plt.suptitle(pdf_name.replace("_", " "), fontsize=14, fontweight='bold')
	
	sns.kdeplot(x, y, shade=True, label='density plot')			# plot density plot
	m,b = np.polyfit(x, y, 1)									# fit data with one degree of movement
	y = m*x+b													# create the fitted y
	line, = plt.plot(x,y,'b',label='best fit')					# plot line of best fit
	line.set_linestyle((0, (1, 5)))   							# set linstyle to loosely dotted
	var = variation(data_splitter_single_column(data_frame,1))	# calculate variance
	plt.title(adjust_method+" adj, slope of line: "+"{:.2e}".format(m)+", coefficient of variation: " + str(round(var[0],3)), fontsize=12)
	plt.legend()
	x_label = str(data_frame.columns[0]).replace("_", " ")
	plt.xlabel(x_label, fontsize=12)
	plt.ylabel(str(data_frame.columns[1]), fontsize=14)
	plt.savefig(pdf_name+"_"+adjust_method+".pdf")
	plt.close()

def export_pdf_mult(pdf_name: str, data_frame: pd.DataFrame, data_frame2: pd.DataFrame, adjust_method: str, adjust_method2: str):
	"""Creates two density plots using the two given DataFrames. The plots are saved in a single A4 sized PDF for vewing convenience.

	Args:
		pdf_name: a string describing the data that is to be plotted
		data_frame: a DataFrame, should be two columns, the first column representing the x-axis nd the second column representing the y-axis of the data to be plotted
		data_frame2: a DataFrame, should be two columns, the first column representing the x-axis nd the second column representing the y-axis of the data to be plotted
		adjust_method: a string describing what adjustment method has been used, this is used for naming the plot
		adjust_method2: a string describing what adjustment method has been used, this is used for naming the plot
	
	Returns: a PDF with two plots, with name described by the string type parameters

	render as pixel-graphic
	save as png first then save as pdf.
	TEST
	"""
	fig, axs = plt.subplots(2,1)
	fig = plt.gcf()
	# set figure to A4 size
	fig.set_size_inches(11.93, 15.98)

	# -------------------------------- plot first plot ---------------#
	x = data_frame.iloc[:,0].to_numpy()									# put data into Series
	y = data_frame.iloc[:,1].to_numpy()

	# from my very extensive testing the sci-kit learn linear regression model and the numpy.polyfit function prodcue the same outputs, either can be used
	reg = LinearRegression()
	x1 = x.reshape(-1, 1)
	y1 = y.reshape(-1, 1)

	reg.fit(x1, y1)

	fig.suptitle(pdf_name.replace("_", " "), fontsize=20, fontweight='bold')
	
	sns.kdeplot(x, y, shade=True, label='density plot', ax=axs[0])			# plot density plot
	m,b = np.polyfit(x, y, 1)									# fit data with one degree of movement
	y = m*x+b													# create the fitted y
	line, = axs[0].plot(x,y,'b',label='best fit')					# plot line of best fit
	line.set_linestyle((0, (1, 5)))   							# set linstyle to loosely dotted
	var = variation(data_splitter_single_column(data_frame,1))	# calculate variance
	axs[0].set_title(adjust_method+" adj, slope of line: "+"{:.2e}".format(m)+", coefficient of variation: " + str(round(var[0],3)), fontsize=14)
	axs[0].legend()
	x_label = str(data_frame.columns[0]).replace("_", " ")
	axs[0].set_xlabel(x_label, fontsize=14)
	axs[0].set_ylabel(str(data_frame.columns[1]), fontsize=14)


	# -------------------------------- plot second plot ---------------#
	# put data into Series
	x = data_frame2.iloc[:,0].to_numpy()
	y = data_frame2.iloc[:,1].to_numpy()

	# from my very extensive testing the sci-kit learn linear regression model and the numpy.polyfit function prodcue the same outputs, either can be used
	reg = LinearRegression()
	x1 = x.reshape(-1, 1)
	y1 = y.reshape(-1, 1)

	reg.fit(x1, y1)

	fig.suptitle(pdf_name.replace("_", " "), fontsize=14, fontweight='bold')
	
	sns.kdeplot(x, y, shade=True, label='density plot', ax=axs[1])			# plot density plot
	m,b = np.polyfit(x, y, 1)									# fit data with one degree of movement
	y = m*x+b													# create the fitted y
	line, = axs[1].plot(x,y,'b',label='best fit')					# plot line of best fit
	line.set_linestyle((0, (1, 5)))   							# set linstyle to loosely dotted
	var = variation(data_splitter_single_column(data_frame2,1))	# calculate variance
	axs[1].set_title(adjust_method2+" adj, slope of line: "+"{:.2e}".format(m)+", coefficient of variation: " + str(round(var[0],3)), fontsize=14)
	axs[1].legend()
	x_label = str(data_frame2.columns[0]).replace("_", " ")
	axs[1].set_xlabel(x_label, fontsize=14)
	axs[1].set_ylabel(str(data_frame2.columns[1]), fontsize=14)

	# save figure as pdf
	fig.savefig(pdf_name+"_"+adjust_method2+".pdf", dpi=100)

	# closes all plots
	plt.close('all')




"""
proportion approach functions
The propoprtion approach takes the Volume of Interest and divides by Total Intracranial Volume
"""
def porportion_method(adjustment_factor: pd.DataFrame, voi: pd.DataFrame) -> pd.DataFrame:
	"""Performs the proportion adjustment method.

	Args:
		adjustment_factor: a DataFrame that should contain only one column
		voi: a DataFrame that can contain multiple columns, should have the same number of indexes as adjustment_factor
	Returns: 
		adjusted volumes as a DataFrame
	"""
	return voi.div(adjustment_factor[str(adjustment_factor.columns[0])], axis ="index")

def porportion_method_mult(adjustment_factor: pd.DataFrame, voi: pd.DataFrame) -> pd.DataFrame:
	"""Performs the proportion adjustment method. Identical to the proportion_method function, except it multiplies, instead of dividing. This should only be called when the adjustment_factor is "scaling"

	Args:
		adjustment_factor: a DataFrame that should contain only one column
		voi: a DataFrame that can contain multiple columns, should have the same number of indexes as adjustment_factor
	Returns: 
		adjusted volumes as a DataFrame
	"""
	return voi.multiply(adjustment_factor[str(adjustment_factor.columns[0])], axis ="index")

def conduct_proportion_analysis(data: pd.DataFrame, adjustment_factor_index: int, voi_index: int, x_axis: str, return_volumes=False):
	"""A function to conduct analysis on the effect the proportion adjustment method has on a particular brain structure. 
	This includes: performing the proportion method, creating plots to visualise the effect and generating statistics

	Args:
		data: a DataFrame containing brain structure volumes
		adjustment_factor_index: an integer, the index in data that is to be used as the adjustment factor
		voi_index: an integer, the index in data that is to be used as the volume of interest
		x_axis: a string defining what the x-axis for the generated plots should be. Either 'TIV' or 'Age'.
		return_volumes: a bool, default is false, if set to True this function returns the adjusted volumes as a single DataFrame column and does not produce plots
	
	Returns: 
		If return_volumes is set to True, returns the adjusted volumes as a single DataFrame column and does not produce plots. 
		However, if false (default) produces a PDF containing two plots visually showcasing the unadjusted and adjusted volumes and additonal statistics on the two volumes. At the very end returns a measure of variation for the adjusted volume.
	"""

	# ----------------------------------------- start of data extraction and fitting to produce results ---------------------------------#
	
	voi_dataframe = data_splitter_single_column(data, voi_index)
	adjustment_factor_dataframe = data_splitter_single_column(data, adjustment_factor_index)

	# use the mult method if using the scaling adjustment factor
	adjustment_factor = str(adjustment_factor_dataframe.columns[0])
	if adjustment_factor == "scaling":
		adjusted_volumes = porportion_method_mult(adjustment_factor_dataframe, voi_dataframe)
	else:
		adjusted_volumes = porportion_method(adjustment_factor_dataframe, voi_dataframe)
	
	# if all we're looking for is the adjusted volumes
	if return_volumes == True:
		return adjusted_volumes

	# ----------------------------------------- start of combining adjusted data and chosen x-axis in order to plot ---------------------------------#

	# change x_axis to whatever you want
	if x_axis == 'Age':
		col_idx = 4
	elif x_axis == 'TIV':
		col_idx = 28
	else:
		print('???')
	x_axis_column = data_splitter_single_column(data, col_idx)

	x_axis_and_voi = data_joiner(x_axis_column, voi_dataframe)

	x_axis_and_adjusted_voi = data_joiner(x_axis_column, adjusted_volumes)

	# dynamically extracts the name of the two columns and makes it the pdf name
	unadjusted_pdf_name = str(x_axis_and_voi.columns[0]) + "_vs_" + str(x_axis_and_voi.columns[1])
	adjusted_pdf_name = str(x_axis_and_adjusted_voi.columns[0]) + "_vs_adjusted_" + str(x_axis_and_adjusted_voi.columns[1])

	# exports the scatter plots to a pdf
	export_pdf_mult(unadjusted_pdf_name, x_axis_and_voi, x_axis_and_adjusted_voi, "no", "proportion")

	# ----------------------------------------- start of joining together pdfs and creating statistics for final pdf ---------------------------------#

	# append pdf
	merger = PdfFileMerger()
	# merger.append(unadjusted_pdf_name+"_"+"no" + ".pdf")
	merger.append(unadjusted_pdf_name+"_"+"proportion" + ".pdf")

	# delete residual pdfs
	# os.remove(unadjusted_pdf_name+"_"+"no" + ".pdf")
	os.remove(unadjusted_pdf_name+"_"+"proportion" + ".pdf")


	# why not lets create it as a markdown file and make the formatting all pretty
	mdFile = MdUtils(file_name=unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "proportion_adj")

	mdFile.new_paragraph('Adjustment factor: ' + adjustment_factor)
	mdFile.new_paragraph('Adjustment method: proportion approach, where to adjust an individual VOI, that VOI is divided by the same indivudal\'s Total Intracranial Volume or whatever chosen adjustment factor')

	mdFile.new_header(level=3, title='Unadjusted Volume of Interest Statistics')

	# create stats on unadjusted volumes
	stats = voi_dataframe.describe().to_string().splitlines()
	for lines in stats:
		mdFile.new_paragraph(lines)
	mdFile.new_paragraph('coefficient of variation ' + str(variation(voi_dataframe)))

	# create stats on adjusted volumes
	mdFile.new_header(level=3, title='Adjusted Volume of Interest Statistics')
	stats = adjusted_volumes.describe().to_string().splitlines()
	for lines in stats:
		mdFile.new_paragraph(lines)
	mdFile.new_paragraph('coefficient of variation ' + str(variation(adjusted_volumes)))

	# if the x axis is Age, additional statistics is produced
	if x_axis == 'Age':
		age_independent_variation = age_resdiual_variance(x_axis_and_adjusted_voi)
		mdFile.new_paragraph('coefficient of variation age independent: ' + str(age_independent_variation))

	mdFile.create_md_file()


	# create a html file from the markdown file
	markdown.markdownFromFile(input=unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "proportion_adj"+".md", output=unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "proportion_adj"+".html")

	# convert html to pdf
	HTML(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "proportion_adj"+".html").write_pdf(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "proportion_adj_stats"+".pdf")

	# append on the created stats pdf
	merger.append(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "proportion_adj_stats"+".pdf")
	merger.write(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "proportion_adj" + ".pdf")
	merger.close()

	# delete residual md, pdf and html file
	os.remove(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "proportion_adj"+".md")
	os.remove(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "proportion_adj"+".html")
	os.remove(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "proportion_adj_stats"+".pdf")

	# the conduct analysis function has to return these values so that further plots can be made using these retun values
	# if the Age is being analysed, we return the Age independent variation
	if x_axis == 'Age':
		return age_independent_variation
	else:
		return variation(adjusted_volumes)[0]





"""
power proportion method functions here
The power-proportion approach fits the Volume of Interest and Total Intracranial Volume in a power law relationship. Then uses the fit to adjust.
"""
def get_beta_value(voi_idx: int) -> float:
	"""
	given the entire dataset and the volume of interest index
	returns a fitted beta value using the power-proportion method
	Is used in adjust_i.py and adjust_cl.py
	"""
	data = main.process_data("./allvols2.csv")
	# split data into single columns, extracts the needed columns, default adjustment factor is TIV (28)
	voi_dataframe = data_splitter_single_column(data, voi_idx)
	adjustment_factor_dataframe = data_splitter_single_column(data, 28)
	# fit the raw volume of interest with the adjustment factor, adjustment factor should be TIV
	beta = power_proportion_train(adjustment_factor_dataframe, voi_dataframe)
	return beta

def power_proportion_method(adjustment_factor: pd.DataFrame, voi: pd.DataFrame, beta: float) -> pd.DataFrame:
	"""
	perform the power proportion method as described by Liu et al.
	data: the entire array
	adjustment_idx: the column in which the adjustment factor is
	volumes_idx: the index that indicate which column to be adjusted
	return: the power adjusted column as a DataFrame
	"""
	power_applied = adjustment_factor.pow(beta)
	return voi.div(power_applied[str(power_applied.columns[0])], axis ="index")

def power_proportion_train(adjust_fact_column: pd.DataFrame, voi_column: pd.DataFrame) -> float:
	"""
	adjust_fact_column: a pandas DataFrame, should only haev one column and is the adjustment factor column
	VOI_column: a pandas DataFrame, should only have one column and is the volume of interest column
	this function trains and gives the "beta" value, should be expecting this "beta" value to be less than 1
	assumption that data is far away from zero
	"""
	# following this tutorial: https://www.youtube.com/watch?v=wujirumjHxU
	# i need to combine the two DataFrames first, because I am sorting
	# data is in the form of x, y
	data = data_joiner(adjust_fact_column, voi_column)
	# removes all rows that have a zero for either VOI or adjustment factor
	data = data[(data != 0).all(1)]
	# data = data.reset_index(drop=True)

	# the log-i-thied data, now we have to perform linear regression on it
	log_data = np.log(data[[str(data.columns[0]),str(data.columns[1])]])

	# scikit learn linear regression
	reg = LinearRegression()
	# add a column of 1s into X, modelling the mean
	X = log_data.iloc[:,0].to_numpy()
	X = X.reshape(-1, 1)
	y = log_data.iloc[:,1].to_numpy()
	reg.fit(X ,y)

	return reg.coef_[0]

def conduct_power_proportion_analysis(data: pd.DataFrame, adjustment_factor_index: int, voi_idx: int, x_axis: str, return_volumes=False) -> pd.DataFrame:
	"""A function to conduct analysis on the effect the power-proportion adjustment method has on a particular brain structure. 
	This includes: performing the power-proportion method, creating plots to visualise the effect and generating statistics

	Args:
		data: a DataFrame containing brain structure volumes
		adjustment_factor_index: an integer, the index in data that is to be used as the adjustment factor
		voi_idx: an integer, the index in data that is to be used as the volume of interest
		x_axis: a string defining what the x-axis for the generated plots should be. Either 'TIV' or 'Age'.
		return_volumes: a bool, default is false, if set to True this function returns the adjusted volumes as a single DataFrame column and does not produce plots
	
	Returns: 
		If return_volumes is set to True, returns the adjusted volumes as a single DataFrame column and does not produce plots. 
		However, if false (default) produces a PDF containing two plots visually showcasing the unadjusted and adjusted volumes and additonal statistics on the two volumes. At the very end returns a measure of variation for the adjusted volume and the fitted beta value.
	"""

	# ----------------------------------------- start of data extraction and fitting to produce results ---------------------------------#
	
	# split data into single columns, extracts the needed columns
	voi_dataframe = data_splitter_single_column(data, voi_idx)
	adjustment_factor_dataframe = data_splitter_single_column(data, adjustment_factor_index)

	# fit the raw volume of interest with the adjustment factor, adjustment factor should be TIV
	beta = power_proportion_train(adjustment_factor_dataframe, voi_dataframe)
	
	# use the beta value produced from fitting to power proportion correct the VOI
	adjusted_volumes = power_proportion_method(adjustment_factor_dataframe, voi_dataframe, beta)

	if return_volumes == True:
		return adjusted_volumes

	# ----------------------------------------- start of combining adjusted data and chosen x-axis in order to plot ---------------------------------#

	# change x_axis to whatever you want
	if x_axis == 'Age':
		col_idx = 4
	elif x_axis == 'TIV':
		col_idx = 28
	else:
		print('???')
	x_axis_column = data_splitter_single_column(data, col_idx)

	# join the x-axis and adjusted/non-adjusted volumes
	x_axis_and_voi = data_joiner(x_axis_column, voi_dataframe)
	x_axis_and_adjusted_voi = data_joiner(x_axis_column, adjusted_volumes)

	# dynamically extracts the name of the two columns and makes it the pdf name
	unadjusted_pdf_name = str(x_axis_and_voi.columns[0]) + "_vs_" + str(x_axis_and_voi.columns[1])
	adjusted_pdf_name = str(x_axis_and_adjusted_voi.columns[0]) + "_vs_adjusted_" + str(x_axis_and_adjusted_voi.columns[1])

	# exports the scatter plots to a pdf
	export_pdf_mult(unadjusted_pdf_name, x_axis_and_voi, x_axis_and_adjusted_voi, "no", "power-proportion")

	# ----------------------------------------- start of joining together pdfs and creating statistics for final pdf ---------------------------------#

	# append pdf
	merger = PdfFileMerger()
	# merger.append(unadjusted_pdf_name+"_"+"no" + ".pdf")
	merger.append(unadjusted_pdf_name+"_"+"power-proportion" + ".pdf")

	# delete residual pdfs
	# os.remove(unadjusted_pdf_name+"_"+"no" + ".pdf")
	os.remove(unadjusted_pdf_name+"_"+"power-proportion" + ".pdf")

	# this way calls to this function with different adjustment factors will have unique pdf names
	adjustment_factor = str(adjustment_factor_dataframe.columns[0])
	

	# create the Markdown file for displaying statistics
	mdFile = MdUtils(file_name=unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "power-proportion_adj")

	mdFile.new_paragraph('Adjustment factor: ' + adjustment_factor)
	mdFile.new_paragraph('Adjustment method: power proportion approach')
	mdFile.new_paragraph('Fitted beta value: ' + str(beta))

	mdFile.new_header(level=3, title='Unadjusted Volume of Interest Statistics')

	# create stats on unadjusted volumes
	stats = voi_dataframe.describe().to_string().splitlines()
	for lines in stats:
		mdFile.new_paragraph(lines)
	mdFile.new_paragraph('coefficient of variation: ' + str(variation(voi_dataframe)))

	mdFile.new_header(level=3, title='Adjusted Volume of Interest Statistics')
	
	# create stats on adjusted volumes
	stats = adjusted_volumes.describe().to_string().splitlines()
	for lines in stats:
		mdFile.new_paragraph(lines)
	mdFile.new_paragraph('coefficient of variation: ' + str(variation(adjusted_volumes)))

	# if the x axis is Age, additional statistics is produced
	if x_axis == 'Age':
		age_independent_variation = age_resdiual_variance(x_axis_and_adjusted_voi)
		mdFile.new_paragraph('coefficient of variation age independent: ' + str(age_independent_variation))

	mdFile.create_md_file()

	# create a html file
	markdown.markdownFromFile(input=unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "power-proportion_adj"+".md", output=unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "power-proportion_adj"+".html")

	# convert html to pdf
	HTML(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "power-proportion_adj"+".html").write_pdf(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "power-proportion_adj_stats"+".pdf")

	# append on the created stats pdf
	merger.append(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "power-proportion_adj_stats"+".pdf")
	merger.write(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "power-proportion_adj" + ".pdf")
	merger.close()

	# delete residual md, pdf and html file
	os.remove(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "power-proportion_adj"+".md")
	os.remove(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "power-proportion_adj"+".html")
	os.remove(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "power-proportion_adj_stats"+".pdf")

	# the conduct analysis function has to return these values so that further plots can be made using these retun values
	# if the Age is being analysed, we return the Age independent variation
	if x_axis == 'Age':
		return beta, age_independent_variation
	else:
		return beta, variation(adjusted_volumes)[0]





"""
General Linear Model functions here
The GLM finds the linear relationship between covariates and the Volume of Interest and removes it.
"""
def get_model():
	"""
	given the dataset and voi index
	uses standard features to make prediction
	Is used in adjust_i.py and adjust_cl.py

	pickles the linear regression model for all 8 volumes of interest
	"""
	data = main.process_data("./allvols2.csv")

	brain_structures = {11:'Total_Brain_Volume', 44:'Thalamus', 45:'Caudate', 46:'Putamen', 48:'Hippocampus', 49:'Amygdala', 50:'Accumbens', 47:'Pallidum'}
	volume_idx_list = [11, 44, 45, 46, 47, 48, 49, 50]

	for volumes in volume_idx_list:
		print(volumes)
		# extract first, turn into numpy arrays
		X = data_splitter_range(data, [0, 3, 4, 28]).values
		y = data_splitter_single_column(data, volumes).values

		# normalise X features
		X = normalise_features(X)

		# train data
		regressor = LinearRegression()  
		regressor.fit(X, y)

		# save trained model
		pkl_filename = "pickle_model_" + brain_structures[volumes] + ".pkl"
		with open(pkl_filename, 'wb') as file:
			pickle.dump(regressor, file)



def normalise_features(X: np.ndarray) -> np.ndarray:
	"""
	given a numpy array of m features and n rows
	this function will find the mean for each column and subtract this from each row in the column
	normalieses each feature
	"""
	# find the means of each column feature
	col_means = np.mean(X, axis = 0)
	# initialise first row to 0
	X_new = np.array([0]*col_means.size)
	# for every row in X, normalise using column means
	for row in X:
		X_new = np.vstack((X_new, [row - col_means]))
	# remove first row of 0s
	X_new = np.delete(X_new, 0, 0)
	return X_new

def conduct_GLM(data: pd.DataFrame, feature_array, voi_idx: int, x_axis: str, return_volumes=False) -> pd.DataFrame:
	"""
	A function to conduct analysis on the effect the general linear model adjustment method has on a particular brain structure. 
	This includes: performing the general linear model adjustment method, creating plots to visualise the effect and generating statistics

	Args:
		data: a DataFrame containing brain structure volumes
		feature_array: an array of indexes that indicate which columns are to be used as features
		voi_idx: an integer, the index in data that is to be used as the volume of interest
		x_axis: a string defining what the x-axis for the generated plots should be. Either 'TIV' or 'Age'.
		return_volumes: a bool, default is false, if set to True this function returns the adjusted volumes as a single DataFrame column and does not produce plots
	
	Returns: 
		If return_volumes is set to True, returns the adjusted volumes as a single DataFrame column and does not produce plots. 
		However, if false (default) produces a PDF containing two plots visually showcasing the unadjusted and adjusted volumes and additonal statistics on the two volumes. At the very end returns a measure of variation for the adjusted volume.
	"""

	# ----------------------------------------- start of data extraction and fitting to produce results ---------------------------------#

	# extract first, turn into numpy arrays
	X = data_splitter_range(data, feature_array).values
	y = data_splitter_single_column(data, voi_idx).values
	# print(np.mean(y))

	# normalise X features
	X = normalise_features(X)

	# have to have a separate voi dataframe to run the describe method when creating statistics
	voi_df = data_splitter_single_column(data, voi_idx)
	voi_name = str(voi_df.columns[0])
	
	# split into train and test sets
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	# I should be plotting my training data against the training data after an adjustment has been made
	# that's why dataframes have been created, so they can be fed into the plotting function
	X_train_df = pd.DataFrame(data=X) # not used, remove probably
	y_train_df = pd.DataFrame(data=y)

	# train data
	regressor = LinearRegression()  
	regressor.fit(X, y)

	# intercept represents 
	intercept = regressor.intercept_
	# print(intercept)

	# predict data on same train set, adjust data according to fitted model
	y_pred = regressor.predict(X)
	y_pred_df = pd.DataFrame(data=y_pred)

	# calculate residual + intercept, represents new data
	residuals = y_train_df - y_pred
	adjusted_volumes = residuals.add(intercept)

	if return_volumes == True:
		return adjusted_volumes

	# ----------------------------------------- start of combining adjusted data and chosen x-axis in order to plot ---------------------------------#

	# change x_axis to whatever you want
	if x_axis == 'Age':
		col_idx = 2
	elif x_axis == 'TIV':
		col_idx = 3
	else:
		print('???')

	# we need to use these columns to graph how well it has removed the effect of head size
	x_axis_column = X[:, col_idx].reshape(-1,1)
	# print(x_axis_column.shape)
	# print(y_pred.shape)
	x_axis_and_adjusted_voi_np = np.append(x_axis_column,adjusted_volumes,axis=1)
	# print(x_axis_and_adjusted_voi_np)
	x_axis_and_adjusted_voi = pd.DataFrame(data=x_axis_and_adjusted_voi_np,columns=[x_axis, "VOI"])

	# join together unadjusted columns
	x_axis_and_unadjusted_voi_np = np.append(x_axis_column, y, axis=1)
	x_axis_and_unadjusted_voi = pd.DataFrame(data=x_axis_and_unadjusted_voi_np,columns=[x_axis, "VOI"])

	# dynamically extracts the name of the two columns and makes it the pdf name
	unadjusted_pdf_name = str(x_axis_and_unadjusted_voi.columns[0]) + "_vs_" + voi_name
	adjusted_pdf_name = str(x_axis_and_adjusted_voi.columns[0]) + "_vs_adjusted_" + voi_name

	# create the plots and export them as pdfs
	export_pdf_mult(unadjusted_pdf_name, x_axis_and_unadjusted_voi, x_axis_and_adjusted_voi, "no", "GLM")

	# ----------------------------------------- start of joining together pdfs and creating statistics for final pdf ---------------------------------#


	# append pdf
	merger = PdfFileMerger()
	# merger.append(unadjusted_pdf_name+"_"+"no" + ".pdf")
	merger.append(unadjusted_pdf_name+"_"+"GLM" + ".pdf")

	# delete residual pdfs
	# os.remove(unadjusted_pdf_name+"_"+"no" + ".pdf")
	os.remove(unadjusted_pdf_name+"_"+"GLM" + ".pdf")

	adjustment_factor = "TIV"

	mdFile = MdUtils(file_name=unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "GLM_adj")

	mdFile.new_paragraph('Adjustment factor: ' + adjustment_factor)
	mdFile.new_paragraph('Adjustment method: GLM/multiple linear regression')

	mdFile.new_header(level=3, title='Unadjusted Volume of Interest Statistics')

	# create stats on unadjusted volumes
	stats = y_train_df.describe().to_string().splitlines()
	for lines in stats:
		mdFile.new_paragraph(lines)
	mdFile.new_paragraph('coefficient of variation: ' + str(variation(y)))

	mdFile.new_header(level=3, title='Adjusted Volume of Interest Statistics')
	stats = adjusted_volumes.describe().to_string().splitlines()
	for lines in stats:
		mdFile.new_paragraph(lines)
	mdFile.new_paragraph('coefficient of variation: ' + str(variation(y_pred)))

	# if the x axis is Age, additional statistics is produced
	if x_axis == 'Age':
		age_independent_variation = age_resdiual_variance(x_axis_and_adjusted_voi)
		mdFile.new_paragraph('coefficient of variation age independent: ' + str(age_independent_variation))

	mdFile.create_md_file()

	# create a html file
	markdown.markdownFromFile(input=unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "GLM_adj"+".md", output=unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "GLM_adj"+".html")

	# convert html to pdf
	HTML(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "GLM_adj"+".html").write_pdf(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "GLM_adj_stats"+".pdf")

	# append on the created stats pdf
	merger.append(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "GLM_adj_stats"+".pdf")
	merger.write(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "GLM_adj" + ".pdf")
	merger.close()

	# delete residual md, pdf and html file
	os.remove(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "GLM_adj"+".md")
	os.remove(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "GLM_adj"+".html")
	os.remove(unadjusted_pdf_name + "_" + adjustment_factor + "_scaled_" + "GLM_adj_stats"+".pdf")

	# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
	# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
	# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

	# the conduct analysis function has to return these values so that further plots can be made using these retun values
	# if the Age is being analysed, we return the Age independent variation
	if x_axis == 'Age':
		return age_independent_variation
	else:
		return variation(adjusted_volumes)[0]





"""
Measures of adjustment effectiveness functions here
"""
def age_resdiual_variance(data: pd.DataFrame) -> float:
	"""
	This function gives a measure of variance such that the effect that age has on brain volumes is accounted for
	Should only be used when x-axis is Age -- and statistical analysis is done on Age

	Why this needs to be done:
	If we want to find a measure of variance for a Volume in terms of age. Directly finding the variance of the volumes is not ideal as we are comparing volumes of different ages. When age naturally decreases brain volumes. Instead the variance of volumes need to be calculated at each age range/age integer value.
	This can be done by finding the line of best fit for Age vs VOI. Then for every row, finding the residuals and returning the variance of these residuals.

	Args:
		data: a DataFrame, needs to be a 2 column DataFrame, where column 0 is Age and column 1 is VOI (adjusted)
	
	Returns: the measure of variance taking into account age's effect on brain volume
	"""
	# x is Age
	x = data.iloc[:,0].to_numpy()
	# y is VOI
	y = data.iloc[:,1].to_numpy()
	mean_vol = np.average(y)

	# perform a regression on Age vs VOI
	reg = LinearRegression()
	x1 = x.reshape(-1, 1)
	y1 = y.reshape(-1, 1)
	reg.fit(x1, y1)

	# for all data points, we use the fitted regression to find the residuals for all entries
	# the coefficient of variation of the residual plus the mean volume is a measurement without the effects of Age on VOI
	diff_list = []
	for ind in data.index: 
		prediction = float(reg.predict(data[str(data.columns[0])][ind].reshape(1, -1)))
		volume = float(data[str(data.columns[1])][ind])
		difference = ( prediction - volume ) + mean_vol
		diff_list.append(difference)

	return variation(diff_list)

def classify_on_Age(data: pd.DataFrame, voi_idx: int):
	"""Another way of measuring the effectiveness of different adjustment methods is by observing the accuracy of a classifier when given different volumes. This function performs proportion, power-proportion and general linear model adjustments on a given volume of interest and feeds these volumes into a support vector machine classifier. This classifier is classifying on whether a patient is young (under 55) or old (above 70) given the features of sex, BMI and the adjusted volume. 

	Args:
		data: a DataFrame containing brain structure volumes
		voi_idx: an integer, the index in data that is to be used as the volume of interest
	
	Returns:
		Outputs the accuracy of the classifier in standard output. Additionally, creates a boxplot comparing the young and old volumes side by side.
	"""
	# creates the raw volume column
	voi_name = str(data.columns[voi_idx])
	raw_volume = data_splitter_single_column(data, voi_idx)
	raw_volume.reset_index(drop=True, inplace=True)

	# creates a tiv volume column -- as a comparison
	tiv_volume = data_splitter_single_column(data, 28)
	tiv_volume.reset_index(drop=True, inplace=True)

	# first get the adjusted volumes using each adjustment method -- are single columns
	proportion_adjusted_volumes = conduct_proportion_analysis(data, 28, voi_idx, 'TIV', return_volumes=True)
	power_proportion_adjusted_volumes = conduct_power_proportion_analysis(data, 28, voi_idx, 'TIV', return_volumes=True)
	glm_adjusted_volumes = conduct_GLM(data, [0, 3, 28], voi_idx, 'TIV', return_volumes=True)
	glm_adjusted_volumes = glm_adjusted_volumes.rename(columns={0: voi_name})

	# reset indexes so that they all align with each others
	proportion_adjusted_volumes.reset_index(drop=True, inplace=True)
	power_proportion_adjusted_volumes.reset_index(drop=True, inplace=True)
	glm_adjusted_volumes.reset_index(drop=True, inplace=True)

	data.reset_index(drop=True, inplace=True)

	# create base dataframe is going to be the features -- Sex, BMI and Age
	base_dataframe = data_splitter_range(data, [0, 3, 4])

	# join adjusted volumes to base
	proportion_adjusted_volumes = data_joiner(base_dataframe, proportion_adjusted_volumes)
	power_proportion_adjusted_volumes = data_joiner(base_dataframe, power_proportion_adjusted_volumes)
	glm_adjusted_volumes = data_joiner(base_dataframe, glm_adjusted_volumes)

	# join the raw volume column to base
	raw_volume = data_joiner(base_dataframe, raw_volume)

	# join the tiv volume column to base
	tiv_volume = data_joiner(base_dataframe, tiv_volume)

	# filter rows such that they are either less than or equal 55 or greater than or equal to 70
	raw_volume = raw_volume[(raw_volume['Age'] >= 70) | (raw_volume['Age'] <= 55)]
	proportion_adjusted_volumes = proportion_adjusted_volumes[(proportion_adjusted_volumes['Age'] >= 70) | (proportion_adjusted_volumes['Age'] <= 55)]
	power_proportion_adjusted_volumes = power_proportion_adjusted_volumes[(power_proportion_adjusted_volumes['Age'] >= 70) | (power_proportion_adjusted_volumes['Age'] <= 55)]
	glm_adjusted_volumes = glm_adjusted_volumes[(glm_adjusted_volumes['Age'] >= 70) | (glm_adjusted_volumes['Age'] <= 55)]
	tiv_volume = tiv_volume[(tiv_volume['Age'] >= 70) | (tiv_volume['Age'] <= 55)]
	
	# reset index
	raw_volume.reset_index(drop=True, inplace=True)
	proportion_adjusted_volumes.reset_index(drop=True, inplace=True)
	power_proportion_adjusted_volumes.reset_index(drop=True, inplace=True)
	glm_adjusted_volumes.reset_index(drop=True, inplace=True)
	tiv_volume.reset_index(drop=True, inplace=True)

	# assign labels -- 0 as young (less than or equal 55) and 1 as old (greater than or equal to 70)
	raw_volume['elderly'] = np.where(raw_volume['Age']>=70, 1, 0)
	proportion_adjusted_volumes['elderly'] = np.where(proportion_adjusted_volumes['Age']>=70, 1, 0)
	power_proportion_adjusted_volumes['elderly'] = np.where(power_proportion_adjusted_volumes['Age']>=70, 1, 0)
	glm_adjusted_volumes['elderly'] = np.where(glm_adjusted_volumes['Age']>=70, 1, 0)
	tiv_volume['elderly'] = np.where(tiv_volume['Age']>=70, 1, 0)

	# want to plot the box plots of <= 55 and > 70
	raw_volume.boxplot(column='Volume_of_brain_grey_and_white_matter', by='elderly')
	plt.savefig("raw_volume_boxplot.png")
	plt.close()
	proportion_adjusted_volumes.boxplot(column='Volume_of_brain_grey_and_white_matter', by='elderly')
	plt.savefig("proportion_adj_boxplot.png")
	plt.close()
	power_proportion_adjusted_volumes.boxplot(column='Volume_of_brain_grey_and_white_matter', by='elderly')
	plt.savefig("power-proportion_adj_boxplot.png")
	plt.close()
	glm_adjusted_volumes.boxplot(column='Volume_of_brain_grey_and_white_matter', by='elderly')
	plt.savefig("glm_adj_boxplot.png")
	plt.close()

	# remove age as a feature
	raw_volume = raw_volume.drop(columns=['Age'])
	proportion_adjusted_volumes = proportion_adjusted_volumes.drop(columns=['Age'])
	power_proportion_adjusted_volumes = power_proportion_adjusted_volumes.drop(columns=['Age'])
	glm_adjusted_volumes = glm_adjusted_volumes.drop(columns=['Age'])
	tiv_volume = tiv_volume.drop(columns=['Age'])

	# split into X and y for each dataframe
	# X should have Sex, BMI and volumes and y should have the young/old label
	raw_volume_X = raw_volume.iloc[:, :3].values
	raw_volume_y = raw_volume.iloc[:, -1].values
	# rescale X
	raw_X_scaler = StandardScaler()
	# gets mean and std
	raw_volume_X = raw_X_scaler.fit_transform(raw_volume_X)


	proportion_adjusted_X = proportion_adjusted_volumes.iloc[:, :3].values
	proportion_adjusted_y = proportion_adjusted_volumes.iloc[:, -1].values
	# rescale X
	proportion_X_scaler = StandardScaler()
	# gets mean and std
	proportion_adjusted_X = proportion_X_scaler.fit_transform(proportion_adjusted_X)


	power_proportion_adjusted_X = power_proportion_adjusted_volumes.iloc[:, :3].values
	power_proportion_adjusted_y = power_proportion_adjusted_volumes.iloc[:, -1].values
	# rescale X
	power_proportion_adjusted_X_scaler = StandardScaler()
	# gets mean and std
	power_proportion_adjusted_X = power_proportion_adjusted_X_scaler.fit_transform(power_proportion_adjusted_X)


	glm_adjusted_X = glm_adjusted_volumes.iloc[:, :3].values
	glm_adjusted_y = glm_adjusted_volumes.iloc[:, -1].values
	# rescale X
	glm_adjusted_X_scaler = StandardScaler()
	# gets mean and std
	glm_adjusted_X = glm_adjusted_X_scaler.fit_transform(glm_adjusted_X)

	# do the same for the tiv volume
	tiv_X = tiv_volume.iloc[:, :3].values
	tiv_y = tiv_volume.iloc[:, -1].values
	# rescale X
	tiv_X_scaler = StandardScaler()
	# gets mean and std
	tiv_X = tiv_X_scaler.fit_transform(tiv_X)


	# split into train and test sets
	raw_volume_X_train, raw_volume_X_test, raw_volume_y_train, raw_volume_y_test = train_test_split(raw_volume_X, raw_volume_y, test_size=0.2, random_state=0)

	proportion_adjusted_X_train, proportion_adjusted_X_test, proportion_adjusted_y_train, proportion_adjusted_y_test = train_test_split(proportion_adjusted_X, proportion_adjusted_y, test_size=0.2, random_state=0)

	power_proportion_X_train, power_proportion_X_test, power_proportion_y_train, power_proportion_y_test = train_test_split(power_proportion_adjusted_X, power_proportion_adjusted_y, test_size=0.2, random_state=0)

	glm_adjusted_X_train, glm_adjusted_X_test, glm_adjusted_y_train, glm_adjusted_y_test = train_test_split(glm_adjusted_X, glm_adjusted_y, test_size=0.2, random_state=0)

	tiv_X_train, tiv_X_test, tiv_y_train, tiv_y_test = train_test_split(tiv_X, tiv_y, test_size=0.2, random_state=0)
	
	# grid search cross-validation
	C = 1.0
	# run linear svm train and output error rate
	raw_volume_model = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C, max_iter=10000)
	raw_volume_model.fit(raw_volume_X_train, raw_volume_y_train)
	raw_volume_pred = raw_volume_model.predict(raw_volume_X_test)
	raw_volume_error_rate = metrics.zero_one_loss(raw_volume_y_test, raw_volume_pred)
	raw_volume_accuracy = 1 - raw_volume_error_rate
	print()
	print("Raw Volume Accuracy", raw_volume_accuracy)

	proportion_adjusted_model = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C, max_iter=10000)
	proportion_adjusted_model.fit(proportion_adjusted_X_train, proportion_adjusted_y_train)
	proportion_adjusted_pred = proportion_adjusted_model.predict(proportion_adjusted_X_test)
	proportion_adjusted_error_rate = metrics.zero_one_loss(proportion_adjusted_y_test, proportion_adjusted_pred)
	proportion_adjusted_accuracy = 1 - proportion_adjusted_error_rate
	print()
	print("Proportion adjusted accuracy", proportion_adjusted_accuracy)

	power_proportion_model = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C, max_iter=10000)
	power_proportion_model.fit(power_proportion_X_train, power_proportion_y_train)
	power_proportion_pred = power_proportion_model.predict(power_proportion_X_test)
	power_proportion_error_rate = metrics.zero_one_loss(power_proportion_y_test, power_proportion_pred)
	power_proportion_accuracy = 1 - power_proportion_error_rate
	print()
	print("Power-proportion adjusted accuracy", power_proportion_accuracy)

	glm_adjusted_model = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C, max_iter=10000)
	glm_adjusted_model.fit(glm_adjusted_X_train, glm_adjusted_y_train)
	glm_adjusted_pred = glm_adjusted_model.predict(glm_adjusted_X_test)
	glm_adjusted_error_rate = metrics.zero_one_loss(glm_adjusted_y_test, glm_adjusted_pred)
	glm_adjusted_accuracy = 1 - glm_adjusted_error_rate
	print()
	print("General linear model adjusted accuracy", glm_adjusted_accuracy)

	tiv_model = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C, max_iter=10000)
	tiv_model.fit(tiv_X_train, tiv_y_train)
	tiv_pred = tiv_model.predict(tiv_X_test)
	tiv_error_rate = metrics.zero_one_loss(tiv_y_test, tiv_pred)
	tiv_accuracy = 1 - tiv_error_rate
	print()
	print("Total Intracranial Volume accuracy", tiv_accuracy)






"""
Start of all the data manipulation functions here
Created for ease of use
"""
def data_splitter_single_column(data: pd.DataFrame, index: int) -> pd.DataFrame:
	"""Selects a single column from a pandas DataFrame and returns it. Useful when only interested in a single column

	Args:
		data: a DataFrame containing brain structure volumes
		index: an integer, the index in data that is to be selected
	
	Returns:
		A DataFrame with a single column.
	"""
	column_name = str(data.columns[index])
	return data.iloc[:,index].to_frame(column_name)

def data_splitter_range(data: pd.DataFrame, range):
	"""Selects a range of columns from a pandas DataFrame and returns it. Useful when interested in a range of columns

	Args:
		data: a DataFrame containing brain structure volumes
		range: a List of integers representing indexes, these indexes are to be selected 
	
	Returns:
		A DataFrame with the selected columns
	"""
	range_df = pd.DataFrame()
	for index in range:
		column_name = str(data.columns[index])
		temp_df = data.iloc[:,index].to_frame(column_name)
		range_df = pd.concat([range_df, temp_df], axis=1)
	return range_df

def data_joiner(data1: pd.DataFrame, data2: pd.DataFrame):
	"""Given two DataFrames, joins the first DataFrame with the second one (side by side).

	Args:
		data1: a DataFrame
		data2: a DataFrame
	
	Returns:
		A DataFrame
	"""
	return pd.concat([data1, data2], axis=1)






"""
Given that the data had some already normalised data
This function plots it out
"""
def csv_normalised_volumes_compare(data: pd.DataFrame):
	"""Creates pdfs for the csv columns that have volumes already normalised
	"""
	# change index to 4 for x axis as age, change to 28 for x axis as TIV
	x_axis_column = data_splitter_single_column(data, 28)

	grey_matter = data_splitter_single_column(data, 7)
	normalised_grey_matter = data_splitter_single_column(data, 6)
	x_axis_and_grey_matter = data_joiner(x_axis_column, grey_matter)
	x_axis_and_normalised_grey_matter = data_joiner(x_axis_column, normalised_grey_matter)
	export_pdf("csv_grey_matter", x_axis_and_grey_matter, "unspecified")
	export_pdf("csv_grey_matter_normalised", x_axis_and_normalised_grey_matter, "unspecified")
	# append pdf
	merger = PdfFileMerger()
	merger.append("csv_grey_matter_unspecified" + ".pdf")
	merger.append("csv_grey_matter_normalised_unspecified" + ".pdf")
	# this way calls to this function with different adjustment factors will have unique pdf names
	merger.write("csv_grey_matter_compared" + ".pdf")
	merger.close()
	# delete residual pdfs
	os.remove("csv_grey_matter_unspecified" + ".pdf")
	os.remove("csv_grey_matter_normalised_unspecified" + ".pdf")

	white_matter = data_splitter_single_column(data, 9)
	normalised_white_matter = data_splitter_single_column(data, 8)
	x_axis_and_white_matter = data_joiner(x_axis_column, white_matter)
	x_axis_and_normalised_white_matter = data_joiner(x_axis_column, normalised_white_matter)
	export_pdf("csv_white_matter", x_axis_and_white_matter, "unspecified")
	export_pdf("csv_white_matter_normalised", x_axis_and_normalised_white_matter, "unspecified")
	# append pdf
	merger = PdfFileMerger()
	merger.append("csv_white_matter_unspecified" + ".pdf")
	merger.append("csv_white_matter_normalised_unspecified" + ".pdf")
	# this way calls to this function with different adjustment factors will have unique pdf names
	merger.write("csv_white_matter_compared" + ".pdf")
	merger.close()
	# delete residual pdfs
	os.remove("csv_white_matter_unspecified" + ".pdf")
	os.remove("csv_white_matter_normalised_unspecified" + ".pdf")

	brain_matter = data_splitter_single_column(data, 11)
	normalised_brain_matter = data_splitter_single_column(data, 10)
	x_axis_and_brain_matter = data_joiner(x_axis_column, brain_matter)
	x_axis_and_normalised_brain_matter = data_joiner(x_axis_column, normalised_brain_matter)
	export_pdf("csv_brain_matter", x_axis_and_brain_matter, "unspecified")
	export_pdf("csv_brain_matter_normalised", x_axis_and_normalised_brain_matter, "unspecified")
	# append pdf
	merger = PdfFileMerger()
	merger.append("csv_brain_matter_unspecified" + ".pdf")
	merger.append("csv_brain_matter_normalised_unspecified" + ".pdf")
	# this way calls to this function with different adjustment factors will have unique pdf names
	merger.write("csv_brain_matter_compared" + ".pdf")
	merger.close()
	# delete residual pdfs
	os.remove("csv_brain_matter_unspecified" + ".pdf")
	os.remove("csv_brain_matter_normalised_unspecified" + ".pdf")
