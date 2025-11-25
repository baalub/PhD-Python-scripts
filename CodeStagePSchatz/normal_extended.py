"""
Class-oriented information geometry for multivariate normal distributions
"""
import openturns as ot
import numpy as np
import sympy as sp
from IPython.display import *
import geocalculnormal as gcn


class Normal(ot.Normal):
    """
    Normal class inherited from openturns, with information geometry tools
    """
    def getFisherInformation(self,fixed_parameters = []):
        
        Fisher_formel = gcn.fisher_info(self.getDimension(),fixed_parameters)[0]
        return Fisher_formel


    def computeFisherInformation(self,fixed_parameters = []):
        
        d = self.getDimension()
        out = gcn.fisher_info(d,fixed_parameters)
        Fisher_formel = out[0]
        symbols = out[1]
        
        valueslist = np.array(self.getParameter())
        valuesdict = dict(zip(symbols,valueslist))
        
        return np.array(Fisher_formel.evalf(subs = valuesdict))



        


    def getChristoffel(self,fixed_parameters = []):
        d = self.getDimension()

        out = gcn.christoffel(d,fixed_parameters)
        Christo_formel = out[0]

        return Christo_formel
    

    def computeChristoffel(self,fixed_parameters = []):
        d = self.getDimension()
        
        out = gcn.christoffel(d,fixed_parameters)

        Christo_formel = out[0]

        symbols = out[1]
        
        valueslist = np.array(self.getParameter())
        valuesdict = dict(zip(symbols,valueslist))




        return [np.array(christo.evalf(subs = valuesdict)) for christo in Christo_formel]

    def computeGeodesic(self, v, x = None, fixed_parameters = [],ret_distrib = True):
        d = self.getDimension()
        if x == None:
            x = self.getParameter()
        
        out = gcn.christoffel(d,fixed_parameters)
        Christo_formel = out[0]
        symbols = out[1]
        fixed_symbols=[sp.symbols(parameter) for parameter in fixed_parameters]

        new_symbols = []
        for sym in symbols:
            if sym not in fixed_symbols:
                new_symbols.append(sym)
        dim = len(new_symbols)
        init_x = dict(zip(symbols,x))
        init_v = dict(zip(new_symbols,v))
        geo = gcn.geo_shooting(Christo_formel,symbols,init_x,init_v,fixed_parameters)
        if ret_distrib:            
            liste = []
            values = init_x
            for pt in geo:
                distrib = Normal(d)
                values.update(dict(zip(new_symbols,pt)))
                
                param = []
                for value in values.values():
                    param.append(value)
                distrib.setParameter(param)
                liste.append(distrib)
            return liste
        return geo[:d]
    

    def drawIsodensityGeodesic(self,v,x = None,fixed_parameters = []):
        
        geo = self.computeGeodesic(v,x,fixed_parameters,ret_distrib = True)

        gcn.isodensity_visual2D(geo, 4, v, a = -5, b = 5)
        


    def drawProjectionGeodesic(self,i,j,v,x=None,fixed_parameters=[]):
        d=self.getDimension()
        geo = self.computeGeodesic(v,x,fixed_parameters,ret_distrib = False)
        symbols=gcn.fisher_info(d)[1]
        
        
        gcn.projections(geo,symbols,i,j,sphere=False)
        

    def expRiem(self,v,x=None,fixed_parameters = [],ret_distrib=True):
        
        geo=self.computeGeodesic(v,x,fixed_parameters,ret_distrib)
        return geo[-1]
    
    def geoSphere(self,x=None,delta=1,N=5,fixed_parameters=[],ret_distrib=True):
        n=self.getParameterDimension()

        dim=n-len(fixed_parameters)
        randomDir=ot.RandomDirection(dim)
        l=[]
        for i in range(N):
            
            v=randomDir.getUniformUnitVectorRealization()
            vect=[delta*el for el in v]
            
            pt=self.expRiem(vect,x,fixed_parameters,ret_distrib)
            
            l.append(pt)
        
        if ret_distrib:
            n=Normal()
            n.setParameter(x)
            l.append(n)
        else:
            l.append(np.array(x))
        return l
    
    def drawIsodensitySphere(self,x=None,delta=1,N=5,fixed_parameters=[]):
        geo=self.geoSphere(x,delta,N,fixed_parameters,ret_distrib=True)
        gcn.isodensity_visual2D(geo,N=5,a=-5,b=5,sphere=True)
         

if __name__ == "__main__":

    # initial points
    Mu = [0, 0]
    rho = 0
    Sigma = [[1, rho],
             [rho, 1]]

    # initial velocity
    v_rho = 0
    v_mu = [0, 1]
    v_sigma = [[1, v_rho],
               [v_rho, 1]]

    n = Normal(Mu, [Sigma[0][0], Sigma[1][1]], ot.CorrelationMatrix(Sigma))
    n.drawIsodensityGeodesic(v=[v_mu[0], v_sigma[0][0], v_mu[1], v_sigma[1][1]], fixed_parameters=['m0', 'm1'])







