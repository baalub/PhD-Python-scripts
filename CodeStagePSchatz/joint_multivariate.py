"""
Class-oriented information geometry for general joint distributions
"""
import openturns as ot
import numpy as np
import sympy as sp
from IPython.display import *
import normal_extended as next
class JointDistribution(ot.JointDistribution):
    """
    Joint distribution class inherited from openturns, with information geometry tools
    """
    def getMarginalNOT(self,i):
        marg=self.getMarginal(i)
        param=marg.getParameter()
        
        name=marg.getName()
        print(name)
        test=hasattr(next,name)
        print(test)
        margtype=getattr(next,name)
        
        margNOT=margtype()
        margNOT.setParameter(param)
        return margNOT
    
    def getFisherInformation(self):
        
        return 

if __name__ == "__main__":
    n1 = ot.Exponential(1)
    n2 = ot.Normal(1)
    n2.setParameter([0.5,1.2])
    c = ot.NormalCopula(ot.CorrelationMatrix([[1,0],[0,1]]))
    j = JointDistribution([n1,n2],c)
    m=j.getMarginalNOT(0)
    #print(m.getFisherInformation())
    #print(m.getPDF())
    