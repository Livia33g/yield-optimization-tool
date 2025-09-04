# Signac

[Signac](https://docs.signac.io/en/latest/) is a python framework that helps in managing project-related data with a well-defined indexable storage layout for data and metadata.  It is very useful for organizing, running, and analyzing the results of computational research, such as molecular dynamics simulations or machine learning workflows with large sets of parameters.  Any computational work that requires you to manage files and execute workflows may benefit from an integration with signac.  The signac documentation provides a useful [quickstart](https://docs.signac.io/en/latest/quickstart.html) and [tutorial](https://docs.signac.io/en/latest/tutorial.html) to help familiarize new users.  This document will also provide an overview of signac and its use in our project.  

# Installation

```bash
# Either
conda install -c conda-forge signac signac-flow

# Or
pip install signac signac-flow
```

# The Signac Project

A signac workflow is organized into what is called a project.  Once a project is initialized, it can be used to organize data and run scripts across parameter space.  A project can be initialized with the `init_project` function.  

```python
import signac

project = signac.init_project("[directory to init project | defaults to current]")
```

It’s as simple as that.  Once initialized, a project can then easily be accessed using the `get_project` function.

```python
project = signac.get_project()
```

## The Job

In signac, collections of parameter values are organized into *jobs* that are stored in a flat directory structure.  All data associated with a set of parameters is uniquely addressable to those parameters.  This organization allows you to go through your workflow on each individual job or parameter set.  

### Creating jobs

Creating jobs in signac is also quite simple.  A job is defined by its *statepoint*, which is a dictionary containing the names and values of its parameters.  The `open_job` function is used for this purpose.  

```python
sp = {"a": value_a, "b": value_b, "c": value_c}
job = project.open_job(sp)
job.init()
```

When the job is initialized, a directory is created that contains its statepoint in `signac_statepoint.json`.  

Creating new jobs is often done using the same file that is used to initialize the project, often called `init.py`.  This file can be called more than once to initialize new jobs with different parameters without reinitializing the project.  Jobs are uniquely defined, so repeated calls to initialize the same job will do nothing beyond the first time.  

- An example [`init.py`](http://init.py) file is presented here.
    
    ```python
    import signac
    
    project = signac.init_project()
    
    feateng_steps_trials = [3, 3, 3, 4, 4]
    featsel_runs_trials = [1, 2, 3, 3, 4]
    std_alphas = [0.5, 1]
    rejection_thrs = [1, 2]
    
    for feateng_steps, featsel_runs in zip(feateng_steps_trials, featsel_runs_trials):
            for std_alpha in std_alphas:
                    for rejection_thr in rejection_thrs:
                            sp = {
                                            "training_set_path": "training_set.csv",
                                            
                                            # --------------  HYPER-PARAMETERS FEATURE GENERATION -------------------
    
                                            #how to scale data. Supported 'standard_nomean', 'standard', 'none'
                                            "scaling_type": "standard_nomean",
    
                                            #whether to leave intercept to vary freely (True) or constrain its value to y0 = 0 (False).
                                            "fit_intercept": False, 
    
                                            # Autofeat hyperparameter.
                                            # Units of predictors. Keys must match column names in dataframe. 
                                            # Ignored predictors are assumed to be dimensionless.
                                            "units": {"T": "K", "c": "mol/kg"},
    
                                            # Autofeat hyperparameter.
                                            # number of times features are combined to obtain ever more complex features.
                                            # example FEATENG_STEPS = 3 with sqrt transformations will find terms like sqrt(sqrt(sqrt(x)))
                                            "feateng_steps": feateng_steps,
    
                                            # Autofeat hyperparameter.
                                            # Number of iterations for filtering out generated features.
                                            "featsel_runs": featsel_runs,
    
                                            # Autofeat hyperparameter.
                                            # Set of non-linear transformations to be applied to initial predictors.
                                            # Autofeat throws an error when using a single transformation. 
                                            # Repeat your transformation as a workaround if you only want o use one.
                                            # Possible tranformations are: "1/", "exp", "log", "abs", "sqrt", "^2", "^3", "1+", "1-", "sin", "cos", "exp-", "2^"
                                            "transformations": ["1/", "exp", "log", "sqrt", "^2", "^3", "1+", "1-", "exp-", "2^"],
    
                                            # --------------  HYPER-PARAMETERS FEATURE SELECTION -------------------
    
                                            # n-standard deviations criterion to choose optimal alpha from Cross Validation. 
                                            # Higher STD_ALPHA lead to sparser solutions.
                                            "std_alpha": std_alpha,
    
                                            #t-statistic rejection threshold. Coefficients with t-statistic < REJECTION_THR are rejected.
                                            "rejection_thr": rejection_thr,
    
                                            # --------------  SYSTEM RESOURCES -------------------
                                            "max_gb": 50
                                    }
                            job = project.open_job(sp)
                            job.init()
    ```
    

### Accessing Jobs

Jobs can be accessed using the FlowProject, as described in the next section, or with the `project.find_jobs` method.  This method returns a cursor that we can use to iterate over all jobs.  Different parts of the statepoint can be accessed as in the following example.  

```python
# Accessing statepoint values
for job in project.find_jobs():
	print(job.sp.a) # will print `value_a` for each job
```

A job’s environment can also easily be accessed using the `job.fn` method or by working in the job’s context.

```python
# Accessing job's environment
# Can use job.fn
with open(job.fn("example_data.txt")) as file:
	data = file.readlines()

# or in context
with job:
	with open("example_data.txt") as file:
		data = file.readlines()
```

# The FlowProject

`signac-flow` is a package in the signac framework that allows user to:

- Implement reproducible computational workflows for a project data space managed with signac.
- Specify operation dependencies with conditions, allowing linear or branched execution
- Run workflows from the command line.
- Submit jobs to high-performance computing (HPC) clusters.

This is achieved using the FlowProject, and we will demonstrate how this works in this section.  

To begin using signac-flow, import the FlowProject object as follows.

```python
from flow import FlowProject

class Project(FlowProject):
    pass

# ...

if __name__ == '__main__':
    Project().main()
```

## Project Operations

Part of your workflow can then be implemented using the `operation` designation.  An operation is a function taking a job instance as argument and identified as such by a decorator.

```python
from flow import FlowProject
import json

# ...

@Project.operation
def store_volume_in_json_file(job):
    with open(job.fn("volume.txt")) as textfile:
        data = {"volume": float(textfile.read())}
        with open(job.fn("data.json"), "w") as jsonfile:
            json.dump(data, jsonfile)

# ...
```

## Running the Workflow

Once a workflow with operations has been created, it can be run from the command line using the following syntax. 

```bash
python project.py run
```

To run a certain number of jobs, the `-n` flag can be used.

```bash
python project.py run -n 1
```

The above example will run the workflow for a single job.  

It is also possible to run a specific operation, using the `-o` flag.  

```bash
python project.py run -o store_volume_in_json_file
```

If you have a scheduling system, such as SLURM, set up in your environment, signac-flow will automatically detect this, and jobs can be submitted to the scheduler using the `submit` flag instead of the `run` flag.  

```bash
python project.py submit [-n number] [-o operation] [...]
```

## Pre and Post Conditions

In complex workflows, it can be useful to add pre and post conditions to operations.  A precondition will only allow the operation to be run while the precondition is true, and a postcondition will only allow the operation to be run while the postcondition is false.  They are implemented as follows.  

```python
from flow import FlowProject
import json

# ...

@Project.pre(volume_computed)
@Project.post.isfile("data.json")
@Project.operation
def store_volume_in_json_file(job):
    with open(job.fn("volume.txt")) as textfile:
        data = {"volume": float(textfile.read())}
        with open(job.fn("data.json"), "w") as jsonfile:
            json.dump(data, jsonfile)

# ...
```

## Labels

The last important aspects of the FlowProject workflow are labels.  Like operations, labels are implemented as decorated functions.  These return a boolean value that can help indicate at which stage of completion certain jobs are within the workflow.  Labels can be used as pre and post conditions, like the `volume_computed` label in the previous example.  Labels are implemented as follows.  

```python
# ...

@Project.label
def volume_computed(job):
    return job.isfile("volume.txt")

# ...
```

The status of labels within a project can be checked by running the following.

```python
python project.py status
```

The `-d` flag can be used to display even more detailed information.  

This should be all you need to know to start using signac!  For more features and advanced functions, be sure to consult the [framework documentation](https://docs.signac.io/en/latest/).
