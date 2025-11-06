import openturns as ot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import LinearSegmentedColormap

import pandas as pd
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

from inputCode import SimulationCodeInputs
from likelihood import Likelihood

from normal import Normal
from gumbel import Gumbel
from triangular import Triangular
from truncatedDistribution import TruncatedDistribution
from jointdistribution import JointDistribution

# define the flood model H as a Python function (or the openturns function ?)
def G(x):
    Q, K, Zm, Zv = x[0], x[1], x[2], x[3]
    D = 300*K*np.sqrt((Zm-Zv)/5000)
    return (Q/D)**0.6



a_Q, b_Q = 0, 3000
a_K, b_K = 15, 60
a_Zm, b_Zm = 54, 56
a_Zv, b_Zv = 49, 51

# specify the input distributions for each input
inputVariables = ["Q", "K", "Zm", "Zv"]
inputNames = ["Strikler coefficient", "flow rate", "height m", "height v"]


truncationBounds = {"Q": [a_Q, b_Q],
                    "K": [a_K, b_K],
                    "Zm": [a_Zm, b_Zm],
                    "Zv": [a_Zv, b_Zv]}


inputs = SimulationCodeInputs(inputVariables, inputNames=None, distTypeStr=None, 
                              truncationBounds=truncationBounds, Dependence = False)

# build the marginal distributions
nominal_dist_Q = TruncatedDistribution(Gumbel(558, 1013), ot.Interval(a_Q, b_Q))
nominal_dist_K = TruncatedDistribution(Normal(30, 7.5),ot.Interval(a_K, b_K))

inputDistDic = {
    "Q": nominal_dist_Q,
    "K": nominal_dist_K,
    "Zm": Triangular(a_Zm, 55, b_Zm),
    "Zv": Triangular(a_Zv, 50, b_Zv)
}

# build the joint distribution of the inputs
nominal_distribution = JointDistribution(list(inputDistDic.values()), ot.IndependentCopula(4))
nominal_distribution.setDescription(ot.Description(inputVariables))

# set the input distribution to the inputs object
inputs.setInputDistribution(nominal_distribution)

# sample from the input distribution and evaluate H on the sample
n = 10000 # sample size

# sample from the whole vector of inputs
sample = nominal_distribution.getSample(n)

# marginal sample
marginal_sample = sample.getMarginal(0)
# output sample 
Y_sample = [G(x) for x in sample]

# print(Y_sample)
# choose a nominal distribution
nominal_dist = nominal_dist_Q
a = a_Q  # the truncation bounds
b = b_Q

str_id = "Q"


# ### a function that builds perturbed distributions for K
# def build_truncated_normal_K(*params):
#     return TruncatedDistribution(Normal(params[0], params[1]), ot.Interval(a_K, b_K))


alpha, beta = 0.95, 0.95

# define a grid 

m = np.linspace(500, 1050, 5)
s = np.linspace(500, 700, 5)


M, S = np.meshgrid(m, s)

# seaborn-v0_8-white
plt.style.use('seaborn-v0_8-white')

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')

cmap = LinearSegmentedColormap.from_list("custom", ["blue", "red"])

# # build the concentric spheres on the bottom
Delta = np.linspace(0.1, 0.5, 5, True)

list_x = []
list_y = []

for i, delta in enumerate(Delta):
    # compute the sphere parameters of radius delta
    SphereParams = np.array(nominal_dist.sampleFisherRaoSphere(delta, nbPts=50))
    # print("delta= ", delta)

    # extract the x-coord and the y-coord and store in a list
    x_coord = SphereParams[:,0]
    x_coord = np.append(x_coord, x_coord[0])
    list_x+=list(x_coord)
    #x_coord = np.append(x_coord, SphereParams[-1,0])
    y_coord = SphereParams[:,1]
    y_coord = np.append(y_coord, y_coord[0])
    list_y+=list(y_coord)

    #y_coord = np.append(y_coord, SphereParams[-1,1])
    color = cmap(i/len(Delta))
    ax.plot(x_coord, y_coord, np.ones(len(y_coord))*3.4, color=color, lw=0.5)

ax.plot([], [], [], label="Concentric spheres", color=cmap(1), lw=0.5)

# plt.show()

# iterate on the grid, define the perturbed distribution, build the likelihood ratio, compute the quantile estimation
# and confidence intervals for each perturbed distribution

Q_estimation =[]
CI_right_bound_H = []
CI_right_bound_B = []

# loop over list_x and list_y to compute the likelihoods
for i in range(len(list_x)):
    pert_dist = TruncatedDistribution(Gumbel(list_y[i], list_x[i]), ot.Interval(a, b))
    # build the likelihood
    likelihood = Likelihood(pert_dist, nominal_dist, pushforward=True)
    print(nominal_dist)
    print(pert_dist)
    # compute the estimation
    Q_estimation.append(likelihood.empiricalWeightedQuantile(Y_sample=Y_sample, alpha=alpha, X_sample = marginal_sample))
    
    # compute the right confidence interval
        # first compute the bounds
    bounds = likelihood.computeLikelihoodBounds(truncated=True, truncationBounds=[a, b], method="bruteforce")
        # set the bounds to the likelihood object
    likelihood.setLikelihoodBounds(bounds)

    # compute the CI
    #CI_H = likelihood.computeCI(Y_sample=Y_sample, alpha=alpha, beta=beta, X_sample=marginal_sample,
    #                intervalType="right bound", CiBoundType="Hoeffding")

    CI_B = likelihood.computeCI(Y_sample=Y_sample, alpha=alpha, beta=beta, X_sample=marginal_sample,
                    intervalType="right bound", CiBoundType="Bennett")

    #CI_right_bound_H.append(CI_H[1])
    CI_right_bound_B.append(CI_B[1])

# for i, mu in enumerate(m):
#     for j, sigma in enumerate(s):
#         # build the perturbed distribution
#         pert_dist = TruncatedDistribution(Gumbel(sigma, mu), ot.Interval(a, b))
#         # build the likelihood
#         likelihood = Likelihood(pert_dist, nominal_dist, pushforward=True)
#         # compute the estimation
#         Q_estimation[j, i] = likelihood.empiricalWeightedQuantile(Y_sample=Y_sample, alpha=alpha, X_sample = marginal_sample)
#         # compute the right confidence interval
#             # first compute the bounds
#         bounds = likelihood.computeLikelihoodBounds(truncated=True, truncationBounds=[a, b], method="scipy")
#             # set the bounds to the likelihood object
#         likelihood.setLikelihoodBounds(bounds)

#         # compute the CI
#         #CI_H = likelihood.computeCI(Y_sample=Y_sample, alpha=alpha, beta=beta, X_sample=marginal_sample,
#         #                intervalType="right bound", CiBoundType="Hoeffding")

#         CI_B = likelihood.computeCI(Y_sample=Y_sample, alpha=alpha, beta=beta, X_sample=marginal_sample,
#                         intervalType="right bound", CiBoundType="Bennett")

#         #CI_right_bound_H[j, i] = CI_H[1]
#         CI_right_bound_B[j, i] = CI_B[1]


# Q_estimation = np.ma.masked_where((M-700)**2+(S-600)**2<10, Q_estimation)



ax.plot_trisurf(list_x, list_y, Q_estimation, label= "95%-quantile estimation", cmap='viridis', vmax=4.8, lw=0.3, alpha=0.8)
#ax.plot_trisurf(list_x, list_y, CI_right_bound_H, label= "95%-CI Hoeffding", cmap="gist_heat", lw=0.6, alpha=0.6)
ax.plot_trisurf(list_x, list_y, CI_right_bound_B, label= "95%-CI Bennett", cmap="gist_heat", vmax=4.9, lw=0.6, alpha=0.6)
ax.view_init(elev=9, azim=-66, roll=1)

ax.scatter([1013],[558], [3.4], color='darkblue', label="Nominal parameter")
ax.legend(fontsize=16)


ax.set_xlabel('m', fontsize=20, labelpad=10)
ax.set_ylabel('s', fontsize=20, labelpad=10)
ax.set_zlabel('River height (m)', fontsize=20, labelpad=10)

# ax.set_zlim(3.7, 4.7)

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='z', labelsize=15)
# ax.set_
fig.savefig(f"/home/bketema/Python workspace/Simplified flood model/images/3D_flood_model.pdf", dpi=600) #, bbox_inches='tight')

# plt.show()

if False:
    # perturb the nominal distribution progressively along a geodesic

    # tangent vector 
    v = np.array([-50, 1]) # for the first curve for Q perturbation [2, 1]
    #v = np.array([-2, -1])
    delta = 1.3

    norm_v = np.sqrt(v.transpose()@nominal_dist.fisherInformation()@v)
    v = delta*v/norm_v

    # computing the discretized geodesic
    discretizeNum = 13 # number of discretization points on the geodesic
    discretizedGeodesic = nominal_dist._sequentialExponentialMap(v, discretizeNum=discretizeNum)

    for f in discretizedGeodesic:
        print(f.getParameter())


    ####---- median estimation along the geodesic ----####
    alpha = 0.95

    # build the likelihoods along the geodesic
    likelihoods_along_geodesic = [
        Likelihood(perturbed, nominal_dist, pushforward=True) for perturbed in discretizedGeodesic 
    ]

    median_estimation = [
        likelihood.empiricalWeightedQuantile(Y_sample=Y_sample, alpha=alpha, X_sample = marginal_sample)
        for likelihood in likelihoods_along_geodesic
    ]

    ####---- Quantile NACIs along the geodesic ----####
    beta = 0.95 # confidence level
    intervalType = "right bound"
    # first, compute the necessary bounds (support and variance bounds) on the likelihood
    NACI_along_geodesic_H = []
    NACI_along_geodesic_B = []

    for likelihood in likelihoods_along_geodesic:
        # compute the bounds
        bounds=likelihood.computeLikelihoodBounds(truncated=True, truncationBounds=[a, b], method="scipy")
        # print("bounds=", bounds)
        # set the bounds to the likelihood object
        likelihood.setLikelihoodBounds(bounds)

        # compute the CI
        CI_H = likelihood.computeCI(Y_sample=Y_sample, alpha=alpha, beta=beta, X_sample=marginal_sample,
                        intervalType=intervalType, CiBoundType="Hoeffding")

        CI_B = likelihood.computeCI(Y_sample=Y_sample, alpha=alpha, beta=beta, X_sample=marginal_sample,
                        intervalType=intervalType, CiBoundType="Bennett")

        NACI_along_geodesic_H.append(CI_H)
        NACI_along_geodesic_B.append(CI_B)

        print(CI_B)

    ####----------------  plots  ----------------####


    plt.style.use('seaborn-v0_8')

    fig, ax = plt.subplots(1, 3, figsize=(24, 5))
    fig.subplots_adjust(hspace=0.35)  # Increase vertical space

    # plotting the curve
    ax[0].set_xlabel(f"${str_id}$", fontsize=15)
    # plt.ylabel("", fontsize=14)
    ax[0].tick_params(axis='x', labelsize=13)
    ax[0].tick_params(axis='y', labelsize=10)

    line = np.linspace(a-(b-a)/10, b+(b-a)/10, 1000)

    # plot the nominal distribution
    vals = [discretizedGeodesic[0].computePDF(x) for x in line]
    ax[0].plot(line, vals, label = "Nominal distribution", color='dodgerblue', lw=2)
    colors = [(0.7, 0.2, p) for p in np.linspace(0.1, 0.2, discretizeNum+1)] # color shade

    # plot the perturbed distributions
    for j, f in enumerate(discretizedGeodesic[1:]):
        vals = [f.computePDF(x) for x in line]
        ax[0].plot(line, vals, color=colors[j], lw=2, alpha=0.5, label='__nolegend__')
        # print(f.getParameter())

    # ax[0].set_ylim(0, 0.0001)
    ax[0].plot([], [], color= colors[int(discretizeNum/2)], label='Perturbed distributions')
    ax[0].set_title(f"Perturbations of the nominal distribution", fontsize=16)
    ax[0].legend(fontsize=15)
    # plt.show()
    # plt.savefig("")

    Delta=np.linspace(1/discretizeNum, delta, discretizeNum, endpoint=True)

    # plt.figure(figsize=(16,9))
    # plt.style.use('seaborn-v0_8')

    ax[1].set_xlabel(r"$\delta$ (perturbation level)", fontsize=16)
    ax[1].set_ylabel("Height (m)", fontsize=16)
    ax[1].tick_params(axis='x', labelsize=13)
    ax[1].tick_params(axis='y', labelsize=13)

    # the IS quantile estimator
    ax[1].plot(Delta, median_estimation, lw =1.5, marker='o', linestyle='--', color='green', label='median estimation')

    # the CIs curves
        # Hoeffding
    upper_bounds = [NACI_along_geodesic_H[i][1] for i in range(discretizeNum)] 
    # lower_bounds = [NACI_along_geodesic_H[i][0] for i in range(discretizeNum)] 
    ax[1].plot(Delta, upper_bounds, lw =1.5, marker='o', linestyle='--', color='blue', label="95%-CI Hoeffding")
    # ax[1].plot(Delta, lower_bounds, lw =1.5, marker='o', linestyle='--', color="blue")
    ax[1].fill_between(Delta, np.zeros(discretizeNum), upper_bounds, color='lightblue', alpha=0.5)

    # np.zeros(discretizeNum)

        # Bennett
    upper_bounds = [NACI_along_geodesic_B[i][1] for i in range(discretizeNum)] 
    # lower_bounds = [NACI_along_geodesic_B[i][0] for i in range(discretizeNum)] 
    ax[1].plot(Delta, upper_bounds, lw =1.5, marker='o', linestyle='--', color='red', label="95%-CI Bennett")
    # ax[1].plot(Delta, lower_bounds, lw =1.5, marker='o', linestyle='--', color="red")
    ax[1].fill_between(Delta, np.zeros(discretizeNum), upper_bounds, color='red', alpha=0.1)

    ax[1].set_title(r"Hoeffding and Bennett right-sided 95%-CIs ($n$=5000, $\alpha$=0.95)", fontsize=16)
    ax[1].set_ylim(3.7, 5.8)
    ax[1].legend(fontsize=15)



    ax[2].set_xlabel(r"$\delta$ (perturbation level)", fontsize=16)
    ax[2].tick_params(axis='x', labelsize=13)
    ax[2].tick_params(axis='y', labelsize=13)

    # the IS Quantile estimator
    #ax[2].plot(Delta, perturbed_extreme["K"], lw =1.5, marker='o', linestyle='--', color='green', label='median estimation')

    # the CIs curves
        # Hoeffding
    likelihood_support = [NACI_along_geodesic_H[i][4] - NACI_along_geodesic_H[i][3]
                        for i in range(discretizeNum)]
    variances = [NACI_along_geodesic_H[i][5] for i in range(discretizeNum)]
    # upper_bounds = [NACI_along_geodesic_H[i]["K"][4] for i in range(discretizeNum)] 

    ax[2].plot(Delta, likelihood_support, lw =1.5, marker='o', linestyle='--', color='purple', label=r"likelihood bound ($b-a$)")
    ax[2].plot(Delta, variances, lw =1.5, marker='o', linestyle='--', color='green', label=r"second moment bound ($\nu$)")


    #     # Bennett
    # upper_bounds = [perturbed_median_NACI_Bennett[i]["K"][1] for i in range(discretizeNum)] 
    # ax[2].plot(Delta, upper_bounds, lw =1.5, marker='o', linestyle='--', color='blue', label="95%-CI Bennett")
    # ax[2].fill_between(Delta, np.zeros(discretizeNum), upper_bounds, color='lightblue', alpha=0.3)

    ax[2].set_title(r"Likelihood second moment and support size comparison", fontsize=16)
    ax[2].legend(fontsize=15)
    #plt.show()

    fig.tight_layout()
    plt.show()
    # fig.savefig(f"/home/bketema/Python workspace/Simplified flood model/images/H_vs_B_extreme_toy_model_input_{str_id}_1.pdf", dpi=600, bbox_inches='tight')


#######------------------ 3D plots ------------------#########



