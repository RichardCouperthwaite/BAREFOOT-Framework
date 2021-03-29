# Batch Reification/Fusion Optimization (BAREFOOT) Framework

![Image of BAREFOOT](https://github.com/RichardCouperthwaite/BAREFOOT-Framework/blob/master/BAREFOOT.png)

The BAREFOOT Framework class structure allows for two different calculation setups:

### Single-Node
This version of the code will run on a single compute node. This compute node could be a single local PC, or a single node on a High Performance Computing Cluster.

### Multi-Node
This version of the code is designed to utlize multiple nodes in a High Performance Computing Cluster. The aim is to reduce the time required of the computations. The number of nodes used in this approach is specified by the user and the total number of calculations are split evenly between these nodes.



## Framework Initialization 
The framework is initialized in a two stage process. The first sets up the framework with general information. The parameters that can be edited are:

* ROMModelList : This is the list of functions that are the cheap information sources. These need to be in a form that ensures that by providing the unit hypercube input, the function will provide the required output
* TruthModel : This is the Truth model, or the function that needs to be optimized.
* calcInitData : This variable controls whether the initial data is calculated for each of the models or is retrieved from a file
* initDataPathorNum : This variable holds the number of initial datapoints to evaluate for each information source (including the Truth Model), or, when initial data is loaded from a file, holds the path to the initial data file
* multiNode : This variable reflects the number of subprocesses that will be used for the calculations. A value of zero indicates all calculations will be completed on the main compute node.
* workingDir : This is the path to the working directory. In some cases it may be desirable to store data separately from the code, this will allow the data to be stored in alternate locations. Can also be used if the relative directory reference is not working correctly.
* calculationName : This is the name for the calculation and will change the results directory name
* nDim : The number of dimensions for the input space that will be used
* restore_calc : This parameter toggles whether the framework data is set up from the information provided or retrieved from a save_state file. This can be used to restart a calculation
* updateROMafterTM : This parameter allows the reduced order models to be retrained after getting more data from the Truth Model. The model function calls do not change, so the training needs to reflect in the same function.
* externalTM : In cases where it is necessary to evaluate the Truth Model separate to the framework (for example, if the Truth Model is an actual experiment), this toggles the output of the predicted points to a separate file for use externally. The framework is shut down after the data is output, see test examples for how to restart the framework after the external Truth Model has been evaluated
* acquisitionFunc : The acquisition function to use to evaluate the next best points for the reduced order models. Currently the options are "KG" for Knowledge Gradient and "EI" for expected improvement.
* A, b, Aeq, beq: Equality and inequality constraints according to the following equations:
  * A*x <= b
  * Aeq*x == b
* ub, lb : Upper bounds and lower bounds for inputs, all inputs must receive a value (Specify 0 for lb and 1 for ub if there is no bound for that input)
* func : function constraints, must take the input matrix (x) and output a vector of length equal to the number of samples in the input matrix (x) with boolean values.
* keepSubRunning : Determines whether the subprocesses are left running while calling the Truth Model
* verbose : Determines the logging level for tracking the calculations.

The second stage sets up the specific calculation required:

* modelParam : This is a dictionary of hyperparameters for the low-order model GPs and the discrepancy GPs. See example below for dictionary structure.
* covFunc : Choice of covariance function for building GP models. Choices are currently limited to Squared Exponential ("SE"), Matern 3/2 ("M32"), and Matern 5/2 ("M52").
* iterLimit : This determines the total number of iterations that the calculation will run for.
* sampleCount : The number of test samples to use. Samples are selected by Latin Hypercube Sampling.
* hpCount : The number of hyperparameter sets to use.
* batchSize : The batch size for the evaluations of the real functions. This affects both the number of reduced order evaluations and the number of Truth Function evaluations.
* tmIter : The iteration limit before calling the Truth Function.
* totalBudget : The total budget (calculation time) that can be expended before the framework is terminated.
* tmBudget : The amount of budget that needs to be used before the Truth Function is evaluated.
* upperBound : The upper bound of the hyperparameters (usually setting to 1 is sufficient since inputs are on a unit hypercube).
* lowBound : The lower bound of the hypeparameter values.
* fusedPoints : The number of points per dimension to use when constructing the Fused GP. Points for evaluating the Fused GP are sampled linearly for each dimension, creating a grid of points to evaluate the fused mean and variance.

The code below is included in the "barefoot.py" file as a test, and provides the minimum input required for running the BAREFOOT Framework.

```
import matplotlib.pyplot as plt

def rom1(x):
    x = x*(2)+0.5
    return -np.sin(9.5*np.pi*x) / (2*x)

def rom2(x):
    x = x*(2)+0.5
    return -(x-1)**4

def tm(x):
    x = x*(2)+0.5
    # Gramacy & Lee Test Function
    return -(x-1)**4 - np.sin(10*np.pi*x) / (2*x)

def plot_results(calcName):
    x = np.linspace(0,1,1000)

    y1 = tm(x)
    y2 = rom1(x)
    y3 = rom2(x)
    
    plt.figure()
    plt.plot(x,y1,label="TM")
    plt.plot(x,y2,label="ROM1")
    plt.plot(x,y3,label="ROM2")
    plt.legend()

    with open('./results/{}/iterationData'.format(calcName), 'rb') as f:
        iterationData = load(f)
    
    plt.figure()
    plt.plot(iterationData.loc[:,"Iteration"], iterationData.loc[:,"Max Found"])

def singeNodeTest():
    np.random.seed(100)
    ROMList = [rom1, rom2]
    test = barefoot(ROMModelList=ROMList, TruthModel=tm, 
                    calcInitData=True, initDataPathorNum=[1,1,1,1], nDim=1, 
                    calculationName="SingleNodeTest", acquisitionFunc="EI")
    modelParam = {'model_l':[[0.1],[0.1]], 
                'model_sf':[1,1,1], 
                'model_sn':[0.01,0.01], 
                'means':[0,0], 
                'std':[1,1], 
                'err_l':[[0.1],[0.1]], 
                'err_sf':[1,1,1], 
                'err_sn':[0.01,0.01],
                'costs':[1,2,20]}
    test.initialize_parameters(modelParam=modelParam, iterLimit=30, 
                               sampleCount=10, hpCount=50, 
                               batchSize=2, tmIter=5)
    test.run_optimization()
    
    plot_results("SingleNodeTest")

if __name__ == "__main__":
    singeNodeTest()
```

For more information on the methods used in this framework, please see our publications:

1. Couperthwaite, Richard, Abhilash Molkeri, Danial Khatamsaz, Ankit Srivastava, Douglas Allaire, and Raymundo Arroyave. “Materials Design Through Batch Bayesian Optimization with Multisource Information Fusion.” JOM, October 13, 2020. https://doi.org/10.1007/s11837-020-04396-x.

