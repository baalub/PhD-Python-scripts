"Bla Bla Bla"

import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import sympy as sp
import matplotlib.colors as mcolors
base_colors=list(mcolors.BASE_COLORS.keys())
from IPython.display import *
import openturns.experimental as otexp
import openturns as ot

from mpl_toolkits.mplot3d import axes3d
sp.init_printing(use_latex=True)

def pyth_integrand_marg(u,distrib):
    gradlog=np.array(distrib.computeLogPDFGradient(u))[np.newaxis]
    pgradlog=gradlog*gradlog.T
    flatpgradlog=pgradlog.flatten()
    return flatpgradlog
    

def fisher_marg(distrib,x,N=2**14):
    d=len(x)
    n=distrib.getDimension()
    pythintegrand=lambda u : pyth_integrand_marg(u,distrib)
    otintegrand=ot.PythonFunction(n,d**2,pythintegrand)

    sobol=ot.SobolSequence(n)

    experimentSobol=ot.LowDiscrepancyExperiment(sobol,distrib,N)
    integration = otexp.ExperimentIntegration(experimentSobol)
    value = integration.integrate(otintegrand)
    I_F=np.reshape(value,(-1,d))

    return I_F

n=ot.Normal(1)
x=n.getParameter()
print(fisher_marg(n,x))