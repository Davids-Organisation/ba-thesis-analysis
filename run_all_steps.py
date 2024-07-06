import subprocess
import os

def run_script(script_name, output_file):
    with open(output_file, 'w') as f:
        process = subprocess.Popen(['python', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        f.write(stdout.decode())
        f.write(stderr.decode())

# Create a folder to save the text files
output_folder = 'replication_outputs'
os.makedirs(output_folder, exist_ok=True)

# Steps for Replication

# 1. Start with readcsv.py
run_script('readcsv.py', os.path.join(output_folder, 'step_1_readcsv_output.txt'))

# 2. Start with readcsv.py
run_script('descriptives.py', os.path.join(output_folder, 'step_2_descriptives_output.txt'))

# 3. Run small_curvefits.py
run_script('small_curvefits.py', os.path.join(output_folder, 'step_2_small_curvefits_output.txt'))

# 4. Run large_curvefits.py
run_script('large_curvefits.py', os.path.join(output_folder, 'step_3_large_curvefits_output.txt'))

# 5. Execute individual_model_fits.py
run_script('individual_model_fits.py', os.path.join(output_folder, 'step_4_individual_model_fits_output.txt'))

# 6. Run individual_fit_desciptives.py
run_script('individual_fit_desciptives.py', os.path.join(output_folder, 'step_5_individual_fit_desciptives_output.txt'))

# 7. Run valuation_dist_plot_rev.py
run_script('valuation_dist_plot_rev.py', os.path.join(output_folder, 'step_6_valuation_dist_plot_rev_output.txt'))

# 8. Execute magnitude_effect.py
run_script('magnitude_effect.py', os.path.join(output_folder, 'step_7_magnitude_effect_output.txt'))

# 9. Run auc_individual_case_plot.py
run_script('auc_individual_case_plot.py', os.path.join(output_folder, 'step_8_auc_individual_case_plot_output.txt'))

# 10. Execute correlation_analysis.py
run_script('correlation_analysis.py', os.path.join(output_folder, 'step_9_correlation_analysis_output.txt'))

# 11. Run reward_punishment_calc.py
run_script('reward_punishment_calc.py', os.path.join(output_folder, 'step_10_reward_punishment_calc_output.txt'))

# 12. Execute gender_explorative.py
run_script('gender_explorative.py', os.path.join(output_folder, 'step_11_gender_explorative_output.txt'))