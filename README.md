# GDELTProject

This repository contains data files and Python scripts for Blue Team’s ISE599 final project: Bringing Advance Clarity to Political Crises.  The files are listed below…

#Code:

GDELT Proj.py This Python script is critical in creating our cleaned data for further analysis.  It's ultimate purpose is to create a dataframe that includes both GDELT and ACLED data, with network and event information for each country by month ordered in a manner that allows us to run further analysis.

	- create_database() - Iterates through GDELT files to create cleaned GDELT database
	- Clean, Split, Transform, Merge Functions - All called by create_database()
		- edgifier() - Selects only edged data (data must have two actors)
		- country_split() - Splits data frames into individual countries
		- network_transform() - Slices the data frame by GS and AT values, creates NetworkX graphs, saves graphs to lists
		- analysis_row_creator() - Creates lists from nodes, degrees, country, and year/month. Combines into rows
		- col_namer() - Creates lists with column names to be merged with analysis_row_creator() later
	- ACLED_data() - Creates a cleaned dataframe from ACLED data
	- model_compiler() - Used to refine the best network slices based on GoldStein Scale and Average Tone
	- combine_dataframes() - Master Function - Calls create_database and ACLED_data then merges their two 

GDELTProject2.py is a Python script that performs time series analysis on the GDELT data and compares models to human forecasters from a recent forecasting competition

GDELT Downloader.pynb  This file uses the GDELTPyR package to download monthly data from GDELT
		
		- start_end() - Converts time series to str to select monthly start and end dates to pull from GDELT
		- create_file() - Selects data that involves our four contries and saves to new files

Model Tuning.pynb -
		This file was used to our paramaters through the use of pipeline and gridsearchCV.  This represents our
		final iteration of tuning as it was an iterative process.

Model Comparison.pynb  This file performed the model comparisons between the non-time series models.

		- Appendix Functions:
			- best_fit_distribution() - Used to fit a distribution of error from the training dataset between 
				y actual and y predicted.
			- ord_brier() - Calculates ordinal brier score when provided our forecasted bin values and 
				actual values.
			- calculate_brier() - Uses model, training set, and set to predict the contest months brier score. 
				Calls best_fit_distribution() and ord_brier()
			- calculate_brier_ind() - similar to calculate_brier() but for models fit to only one country at a time.
		- All Countries
			12: Reads the data, performs one hot encoding, adds (1,2,3) months lag, adds missing values, selects sets
			13: Performs Standard Scaler transformation
			14: Commented out PCA transformation (negatively effected model)
			17: Sets models with tuned parameters (Models tuned in Model Tuning)
			18: Creates a series of lists
			19: Loops through all of the models, performs 10 fold cross val, calls calculate_brier(), returns lists
				- Metrics
					- MSE
					- R^2
					- Brier Scores
			22: Merges lists of metrics into dataframe
			23: Brier score Averages
			24: Show predicted and actual fatalities by model for the competition month
			26: Creates box plot for 10 fold MSE of each model
 		- Individual Countries
			27: Creates new lists for individual countries
			28: Loops through data transformation (similar to 12:) for each country.  Runs models and colects metrics
				for individual countries
			29-31: Displays individual metrics (worse performance than bundled countries).

Importance Analysis.pynb This file runs importance analysis with the top performing models to gain insights from coefficient Analysis performed on linear models and importance Analysis performed on Ensemble Models

Initial Project Stuff.pynb  This file was used to create some of the initial map visualizations from the ACLED data Set


#Data:

battle_data.csv contains the cleaned data used for our analysis

new_dataTS.csv contains the cleaned data prepared for time series analysis

2015-03-01-2019-11-30-Central_African_Republic-Democratic_Republic_of_Congo-Ethiopia-South_Sudan - ACLED Data.csv is a file of battle death counts from ACLED.

East_Afr 2015 06 01.csv is an example of one month of GDELT data

