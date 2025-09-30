import subprocess, sys, os

CLI_SCRIPTS = [
    'core/calc_hv.py',
    'core/calc_w_inter_curves.py',
    'core/calc_sensitivity_curves.py',
    'core/plot_w_inter_curves_2x2.py',
    'core/plot_sensitivity_with_gcc.py',
    'aux/calc_all_metrics.py',
    'aux/run_pipeline_aux.py',
]

def test_help_runs():
    for cli in CLI_SCRIPTS:
        if not os.path.exists(cli):
            # skip if the script is not present in the CI checkout
            continue
        subprocess.check_call([sys.executable, cli, '--help'])
