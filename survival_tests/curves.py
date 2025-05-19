import sys
sys.path.append('../../survival_tests')

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from schedule import survival_curve_from_schedule, termination_curve_from_train_data, schedule_termination_curve
from survival_curves import prepare_survival_curves_for_plot, plot_survival_funcs, create_scenario, create_survival_curves


def main():
    scenario_name = 'QBF-2011'
    instance_id = 0

    scenario = create_scenario(scenario_name, 'aslib_data/')

    EVENT_TIMES, SURVIVAL_FUNCTIONS, cutoff, train_scenario, test_scenario = create_survival_curves(scenario, instance_id)
    print("survival curves created")



    plot_survival_funcs(EVENT_TIMES, SURVIVAL_FUNCTIONS, cutoff, save=True)
    print("plotted")

    instance_id = 6

    scenario = create_scenario(scenario_name, 'aslib_data/')


    EVENT_TIMES, SURVIVAL_FUNCTIONS, cutoff, train_scenario, test_scenario = create_survival_curves(scenario,
                                                                                                    instance_id)
    print("survival curves created")

    plot_survival_funcs(EVENT_TIMES, SURVIVAL_FUNCTIONS, cutoff, filename="survival_functions2", save=True)
    print("plotted")


if __name__ == "__main__":
    main()