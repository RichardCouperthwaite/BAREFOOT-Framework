# Batch Reification/Fusion Optimization (BAREFOOT) Framework

Two versions of the code are currently in this repository. These two codes are explained below:

### Single-Node
This version of the code will run on a single compute node. This compute node could be a single local PC, or a single node on a High Performance Computing Cluster.

### Multi-Node
This version of the code is designed to utlize multiple nodes in a High Performance Computing Cluster. The aim is to reduce the time required of the computations. The number of clusters used in this approach is calculated by mutliplying the number of reduced-order models, number of samples, and number of hyperparameter sets together and then dividing by 1000. The code will use a single node for each set of 1000 (or part thereof) calculations. So for example, using 30 samples, with 500 hyperparameter sets and 3 reduced-order models a total of 46 nodes will be used.
