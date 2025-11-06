import openturns as ot
import numpy as np


class Triangular(ot.Triangular):
    """
    The fisher information for the triangular family is computed in the 
    mode parameter only
    """

    def fisherInformation(self):
        a = self.getParameter(0)
        m = self.getParameter(1)
        b = self.getParameter(2)

        return 1/(b-m) * 1/(m-a)

    
    def exponentialMap(self, v):
        pass

    def sampleFisherRaoSphere(self, delta):
        """
        computes two points on this 1D Fisher-Rao sphere
        """
        a = self.getParameter(0)
        m = self.getParameter(1)
        b = self.getParameter(2)

        alpha = (a+b)/2
        beta = (a-b)**2/4

        m_al_bet = (m - alpha)/np.sqrt(beta)
        
        s_plus = np.sqrt(beta)*np.sin( delta + np.arcsin(m_al_bet) ) + alpha
        s_minus = np.sqrt(beta)*np.sin( np.arcsin(m_al_bet) - delta) + alpha
        
        return [s_minus,s_plus]