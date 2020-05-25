# Batch Bayesian Optimization
## Implemented within a Model Reification and Fusion Framework

The code in this repository can be used to run a batch bayesian optimization routine using a 2D toy problem involving the reification and fusion of three reduced-order mechanical models to predict the output of a RVE Mechanical Property model.

Two versions of the code are currently in this repository. These two codes are explained below:

### Cost Constrained - Cost Controlled (CC_CC_optimization.py)
This version of the code has been manipulated to allow for running the code constraining the total cost as well as calling the truth model only after a certain budget allocation has been spent. 

### Cost Constrained - Iteration Controlled (CC_IC_optimization.py)
This version of the code runs the case study where the optimization terminates after a certain budget allocation has been met, or when a certain number of iterations have been completed. The second criteria in this code is that the truth model is called after a fixed number of iterations.

To run the code, the following python packages will need to be installed:
- pandas
- numpy
- scipy
- pyDOE
- scikit-learn
- george
- matplotlib

Both versions of the code take command line inputs to specify the different limits that will be used in the code, these are input as an ordered list:
1. GP Kernel to be used ('M52', 'M32', 'SE')
2. The total number of iterations to completed
3. The number of test samples to use (these are found using a latin-hypercube sampling of the design space
4. The number of hyperparameter sets to use in the batch optimization procedure
5. The number of clusters (medoids) to partition the data into
6. The number of iterations to run before calling the truth model
7. The total budget that can be expended before stopping the code
8. The budget that must be expended before calling the truth model

The code is set up to run an example problem if no additional inputs are entered. This example code will run for two iterations. The example code is equivalent to entering the following run command:

```
python CC_CC_optimization.py M52 2 10 50 2 1 14000 1000
```


