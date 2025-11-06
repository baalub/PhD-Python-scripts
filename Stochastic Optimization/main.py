import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
import pandas as pd

# import modules from another directory
import sys

from pathlib import Path

# get the current directory
current_dir = Path.cwd()

# move back
parent_dir = current_dir.parent
file_path_1 = parent_dir / "Multivariate_case"
file_path_2 = parent_dir / "Confidence intervals"

sys.path.append(str(file_path_1))
sys.path.append(str(file_path_2)) 

from truncatedDistribution import TruncatedDistribution
from normal import Normal
from lognormal import LogNormal
from extendedtrnormal import ExtendedTrNormal

from inputCode import SimulationCodeInputs


from stoOptProb import RiemannianStochasticOptimization

# define the computer simulation
def G(x):
    return -x**2

# The integrand function for superquantile computation/optimization
def H(x, alpha, q):
    return q + 1/(1-alpha) * np.max(G(x)-q,0)

# implement the stochastic optimization algorithm to find the optimal quantile

# 1) define a python function to set the starting input distribution as an instance of the 
# JointDistribution class using the ExtendedTrNormal and ExtendedTrLogNormal classes

input_description_file = pd.read_excel(
    '/home/bketema/Python workspace/Cathare simulations/cathare_bethsy_input_description.ods'
    )

inputs = SimulationCodeInputs.from_dataframe(input_description_file)

# build the input distribution using the elements of the extended families of the truncated normal
# and lognormal distributions
extended_families = True
distribution_initial = inputs.buildInputDistribution(extended=extended_families)

# set the input distribution to the SimulationCodeInputs object
inputs.setInputDistribution(distribution_initial)

# 2) test: build a python function that allows to sample from this JointDistribution object

X = distribution_initial.getSample(100)

# 3) Given a collection of inputs that we want to perturb, say ['X2', 'X3', 'X36'], build a
# python function which allows to update the marginal distributions and sample from this new
# distribution

# inputs whose distribution is to be perturbed
perturb_inputs = ['X2', 'X5']
# perturb_indices = [1, 4] # corresponding indices in python's convention ?

# copula on the perturbed distribution, independent for the moment (here is where Pierre Schatz's
# internship on the multivariate case may come in later)
copulaOnPartialDistribution = ot.IndependentCopula(len(perturb_inputs))

# distribution on these perturbed inputs 
f_0 = inputs.buildPartialDistribution(
    extended=extended_families, perturb_inputs=perturb_inputs, copula=copulaOnPartialDistribution
    )

quit()

# 4) Apply the optimization algorithm

# first we decide if the non perturbed marginals are penalized (i.e. Dirac) or are kept random
penalize_non_perturbed_inputs: bool = True
# TODO
distribution_start = inputs

nbIter = 50
n_k = 100
eta = 0.01
alpha = 0.95

quantile_optimization_PCT = RiemannianStochasticOptimization(
    G, fullDistribution=distribution_start, partialDistribution=f_0, perturbedInputs=perturb_inputs
    )


iterations, Y_samples, samples, perturb_inputs = quantile_optimization_PCT.naturalEvolutionStrategy(
    # the fitness shaping : either as Wierstra et al. 2014 "utility" or Ollivier et al. 2017 "omega"
    fitnessShapingStr="omega", lamb=alpha,
    # step size
    eta=eta,
    # number of iterations of the algorithm
    nbIter=nbIter,
    # sample size at each iteration
    n_k=n_k
    )


# define a function that stores the iterations (objects' class, attributes, etc), the output samples 
# in a dataframe
df_iterations, df_Ysamples, df_perturbed_inputs = quantile_optimization_PCT.store_optimization_results(
    iterations, Y_samples=Y_samples
    )

# save to .csv filesempirical running
save_dir = ""
df_iterations.to_csv(save_dir+"/iterations.csv")
df_Ysamples.to_csv(save_dir+"/Ysamples.csv")
df_perturbed_inputs.to_csv(save_dir+"/perturbed_inputs.csv")

#distribution_end = iterations[-1]























# paramFam = "Normal"
# f0 = Normal(0, 1)  # starting point of the algorithm

# #f_base =  Normal(0, 1)   # sampling baseline distribution
# q0 = 3  # starting point of the algorithm
# eta = 2 # (learning rate)
# alpha = 0.01
# n = 1000

# # X_sample = f_base.getSample(n)
# # X_sample_G = np.array(X_sample).transpose()[0]
# # Y_sample = G(X_sample_G)


# # define the superquantile optimization problem
# superQuantileOpt = RiemannianStochasticOptimization(
#     H, paramFam, startingDist=f0, realParam=True
# )

# X = np.linspace(-4,4, 100)
# plt.plot(X, [f0.computePDF(x) for x in X])

# m = []
# s = []
# for i in range(10):
#     f, q = superQuantileOpt.superquantileRSGD(
#         G, startingDist=f0, startingQuantile=q0, eta=eta, alpha=alpha, iterations=n
#     )
#     m.append(f.getParameter()[0])
#     s.append(f.getParameter()[1])
#     print("final f =", f)
#     print("final q =", q)

#     print(f.computeQuantile(alpha))

#     plt.plot(X, [f.computePDF(x) for x in X], color = 'orange')

# print("m = ", m)
# print("s = ", s)
# plt.show()
