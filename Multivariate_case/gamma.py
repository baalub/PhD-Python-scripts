import openturns as ot
import numpy as np 

import previous_modules.truncated_Gamma as trGam


class Gamma(ot.Gamma):
    """
    alpha (our notation) = alpha (ot notation)
    beta (our notation) = lambda (ot notation)"""

    def fisherInformation(self):

        # if args:
        #     return self._fisherInformationTrCase(args)

        alpha, beta = self.getParameter()[0], self.getParameter()[1]
        return trGam.I(alpha, beta)

    # def _fisherInformationTrCase(self, interval: ot.Interval):
    #     a, b = interval.getLowerBound(), interval.getUpperBound

    #     return trGam.J()


    def exponentialMap(self, v, h=0.01):    
        N = np.int64(1/h)
        #X = odeint(H_beta, X_0, np.linspace(0, Tf, N, endpoint=True))
        X0 = np.array([self.alpha, self.beta, v[0], v[1]])

        alpha_t, beta_t = trGam.geod_Gamma_non_tronquees(1, h, X0)

        return Gamma(alpha_t[-1], beta_t[-1])
    
    
    def sampleFisherRaoSphere(self, delta, nbPts):
        """
        taken from the tracer_sphere_fisher_avec_geod function from the truncated_Gaussian module
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