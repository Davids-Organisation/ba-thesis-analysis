To create all the Results and Graphs in one step type "python run_all_steps.py" in the commant line.


The follwing lists all Steps for individual replication:

	1.	Start with readcsv.py:
	•	This script creates a new CSV called treated_data.csv, which includes the initial transformations, calculation of RSV, AUC, and aggregate common acceptance measures.
	•	It also creates two folders: GraphsEPS and GraphsPDF, each containing a subfolder auc_individual_plots where graphs will be stored.
	2.	Run small_curvefits.py and large_curvefits.py:
	•	These scripts perform the initial model fittings to the mean RSV at every distance and print results in the console.
	•	Corresponding graphs from the thesis are stored in the folders.
	3.	Execute individual_model_fits.py:
	•	This script fits hyperbolic power functions to all individual discounting data, creating fitting_results.csv with R-Squared and AIC values.
	4.	Run individual_fit_desciptives.py:
	•	This calculates the mean fit quality, with results printed in the console.
	5.	Run valuation_dist_plot_rev.py:
	•	Generates the thesis’ distribution plots.
	6.	Execute magnitude_effect.py:
	•	Performs normality tests and mean comparisons, with results printed in the console.
	7.	Run auc_individual_case_plot.py:
	•	Plots AuC in small amount conditions for individual cases. Specify the desired case, and plots are saved in the auc_individual_plots folders.
	8.	Execute correlation_analysis.py:
	•	Performs the Spearman Correlation test and subsequent regression if correlations are found at the 10% level. Results are printed in the console.
	9.	Run reward_punishment_calc.py:
	•	Generates “Dilemma Gap” plots, with result lists printed in the console.
	10.	Execute gender_explorative.py:
	•	Analyzes gender differences, with results printed in the console.

Now all graphs and results should be generated. To analyze your own data, change the path of the read function in readcsv.py to your own dataset, with the same structure as RAW_DATASET.csv.



Bonus – Demo Files

To analyze demo discounting data from the demo_decision_phase.html file in the Combination_Discounting_Task folder:

	1.	Place the resulting one-lined demo_results.csv file in this project.
	2.	Run demo_readcsv.py to overwrite treated_data.csv with your demo results.

You can now run all scripts mentioned above except gender_explorative.py and correlation_analysis.py on your discounting data. The mean fit and individual fit remain unchanged.
