import openturns as ot
import numpy as np
import matplotlib.pyplot as plt

from normal import Normal
from lognormal import LogNormal
from truncatedDistribution import TruncatedDistribution
from distribution import Distribution
from extendedtrnormal import ExtendedTrNormal

a= 0.1
b= 10

interval = ot.Interval(a,b)

import timeit
start = timeit.default_timer()

marginals = [ot.TruncatedDistribution(Normal(0, 1), ot.Interval(-1, 1)), ot.Beta(1, 2, 3, 4)]
f = ot.JointDistribution(marginals, ot.IndependentCopula(2))

print(f.getSample(5))

quit()


f1 = TruncatedDistribution(Normal(0, 1), ot.Interval(-2, 2))
#f2 = TruncatedDistribution(Normal(3, 4))


# sphere = f1.sampleFisherRaoSphere(delta=0.3, nbPts=10)
# for f in sphere:
#     print(f.getParameter())
f = Normal(0, 1)
v = np.array([0, -1])
h = f1.exponentialMap(v, h=0.001)
print(h.getParameter())

current_time = timeit.default_timer()

print(f"It took {current_time-start} seconds")



# n_tr = TruncatedDistribution(n, -1.0, 1.0)
# #N_ot_tr = TruncatedDistribution(N_ot, -1.0, 1.0)

# # compute Fisher information for truncated Normal distributions at n_tr

# # I = n.fisherInformation()
# # print(I)

# # I_tr = n_tr.fisherInformation()
# # print(I_tr)

# # compute fisher information print(N_ot.fisherInformation())

# N_ot = Distribution(ot.Gumbel())

# print(N_ot.fisherInformation())

# S = n.getSample(10)
# print(S)

# S_tr = n_tr.getSample(10)
# print(S_tr)


