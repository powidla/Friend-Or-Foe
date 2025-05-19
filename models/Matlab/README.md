In this folder we provide MATLAB functions for metabolic modeling. Below we briefly describe the purpose of each function. Running this code requires Gurobi Academic License.

- ````getmetmodel_fromFile.m```` : loads a ````.mat```` model to the source; 

- ````getHGTmodel_fromFile.m```` : assigns each ````.mat```` model to a Microbe;

- ````createTempmetmodelMi_noenvrhslbs.m```` : builds S matrix from ````.mat```` model;

- ````createBaseTempmetmodelMiAndMj.m```` : creates a combined S matrix and model (two Microbes) to execute Algorithm 1 from the paper;

- ````runTempmodelMi_extras.m```` : runs Gurobi optimizer to solve Flux Balance Analysis for single Microbe (sole ````.mat```` model);

- ````runTempmodelMiAndMj_nWconstraints_extras.m```` : runs Gurobi optimizer to solve Flux Balance Analysis for a pair (Algorithm 1);

- ````JustGrowPairFromTempModels_noWTog_OutputRxnAndECFluxes.m```` : additional func to select which Microbe to optimize and which Microbe to fix;

- ````GrowPairFromTempModels_noWTog_OutputFluxes.m```` :  additional func to save fluxes from Algorithm 1;

- ````CategoriseNEW_withStrongComp.m```` : function that describes main part of Algorithm 2 from the paper, creates labels.
