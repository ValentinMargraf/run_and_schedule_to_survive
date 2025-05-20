# Code for paper: "RunAndSchedule2Survive: Algorithm Scheduling Based on Run2Survive"

This repository holds the code for our paper "RunAndSchedule2Survive: Algorithm Scheduling Based on Run2Survive" by Valentin Margraf, Tom Körner, Alexander Tornede and Marcel Wever. Regarding questions please contact valentin.margraf@ifi.lmu.de .



## Abstract
The algorithm selection problem aims to identify the most suitable algorithm for a given problem instance under specific time
constraints, where suitability typically refers to a performance metric such as algorithm runtime. While previous work has employed
machine learning techniques to tackle this challenge, methods from survival analysis have proven particularly effective. This paper
presents RunAndSchedule2Survive to address the more general and complex problem of algorithm scheduling, where the objective
is to allocate computational resources across multiple algorithms to maximize performance within specified time constraints. Our
approach combines survival analysis with evolutionary algorithms to optimize algorithm schedules by leveraging runtime distributions
modeled as survival functions. Experimental results across various standard benchmarks demonstrate that our approach significantly
outperforms previous methods for algorithm scheduling and yields more robust results than its algorithm selection variant. More
specifically, RunAndSchedule2Survive achieves superior performance in 20 out of 25 benchmark scenarios, surpassing hitherto
state-of-the-art approaches.

## Execution Details (Getting the Code To Run)
For the sake of reproducibility, we will detail how to reproduce the results presented in the paper below.



## Installation

```bash
# Create and activate conda environment
conda create -n runandschedule python=3.8 -c conda-forge
conda activate runandschedule

# Clone the repository
git clone git@github.com:ValentinMargraf/run_and_schedule_to_survive.git
cd run_and_schedule_to_survive

# Install the package
pip install -r requirements.txt
```



### ASLib Data
Obviously, the code requires access to the ASLib scenarios in order to run the requested evaluations. It expects the ASLib scenarios (which can be downloaded from [Github](https://github.com/coseal/aslib_data)) to be located in a folder `data/aslib_data-master` on the top-level of your IDE project. I.e. your folder structure should look similar to this:
```
./survival_tests
./survival_tests/approaches
./survival_tests/approaches/survival_forests
./survival_tests/results
./survival_tests/singularity
./survival_tests/baselines
./survival_tests/data
./survival_tests/data/aslib_data-master
./survival_tests/conf
```


### Running Experiments
To reproduce our results, run the following experiments. Inside `survival_tests/` run `run_schedule.py` which will run our approach on all evaluated ASLib scenarios and folds.


### Generating Plots
All plots found in the paper can be generated using the self-explanatory Jupyter notebook `survival_curves.py` in the top-level `survival_tests/` folder.
