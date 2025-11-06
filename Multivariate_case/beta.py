import openturns as ot
import numpy as np

from scipy.special import polygamma as pgam
from scipy.special import gamma
from scipy.integrate import odeint

import matplotlib.pyplot as plt

import previous_modules.beta_family as bf


class Beta(ot.Beta):

    def __init__(self, alpha, beta, a, b):
        super().__init__(alpha, beta, a, b)
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        #self._interval = ot.Interval(a, b)


    def fisherInformation(self):

        # computes the Fisher information for the Beta family
        # with respect to alpha and beta only
        # TODO add reference and make the change of variables
        # onto [0, 1]

        alpha = self.alpha
        beta = self.beta

        return bf.I(alpha, beta)
    

    def exponentialMap(self, v, h=0.005, curve=False):

        N = np.int64(1/h)
        #X = odeint(H_beta, X_0, np.linspace(0, Tf, N, endpoint=True))
        X0 = np.array([self.alpha, self.beta, v[0], v[1]])

        if curve:
            return bf.geod_Beta(1, h, X0)
        

        alpha_t, beta_t = bf.geod_Beta(1, h, X0)

        return Beta(alpha_t[-1], beta_t[-1], self.a, self.b)

    def sampleFisherRaoSphere(self, delta, nbPts):
        """ 
        sample from the Fisher-Rao sphere
        """

        I = self.fisherInformation()
        L = np.linspace(0,2*np.pi,nbPts, endpoint=False)
  
        spherePointsList = []
        for t in L:
            v = np.array([np.cos(t),np.sin(t)])
            l_J = np.sqrt(np.dot(np.transpose(v),np.dot(I,v)))
            v_J = delta*(v/l_J)
            spherePoint = self.exponentialMap(v_J)
            spherePointsList.append(spherePoint)
            
        return spherePointsList




if __name__ == "__main__":
    be = Beta(1, 1, 0, 1)

    sphere = be.sampleFisherRaoSphere(0.4, 100)
    sphere.append(sphere[0])


    L = np.linspace(-0.05, 1.05, 200)

    for f in sphere[1:]:
        plt.plot(L, [f.computePDF(x) for x in L], 'r', lw=0.2)

    plt.plot(L, [be.computePDF(x) for x in L], color='dodgerblue', lw=1)

    plt.show()

    
    # convert sphere into parameters
    fisher_sphere_params_nontr = np.array([g.getParameter() for g in sphere])

    # print(fisher_sphere_params)

    ### plot in parameter space

    # non_tr case 
    # plot th closed ball and sphere
    plt.fill(fisher_sphere_params_nontr[:,0], fisher_sphere_params_nontr[:,1], color="blue", alpha=0.05)
    plt.plot(fisher_sphere_params_nontr[:,0], fisher_sphere_params_nontr[:,1], color='red', alpha=0.5, lw=2.5) # , label=f"$\delta$={delta:.2}")
    
    plt.show()