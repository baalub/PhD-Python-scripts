"""
Module de calcul pour normal_extended
"""
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import sympy as sp
import matplotlib.colors as mcolors
base_colors=list(mcolors.BASE_COLORS.keys())
from IPython.display import *
import openturns as ot

import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap

from mpl_toolkits.mplot3d import axes3d
sp.init_printing(use_latex=True)



def dens(d):
    """
    Densité formelle sympy d'une gaussienne de dimension d
    """
    #définition des symboles
    x=[sp.symbols('x%d'%i) for i in range(d)]
    m=[sp.symbols('mu%d'%i) for i in range(d)]
    s=[sp.symbols('sigma%d'%i) for i in range(d)]
    r=[sp.symbols('rho%d'%i) for i in range(int(d*(d-1)/2))]

    #définition de la matrice de covariance
    Covar=sp.zeros(d,d)
    for i in range(d):
        for j in range(d):
            if i==j:
                Covar[i+d*j]=s[i]**2
            else:
                Covar[i+d*j]=s[i]*s[j]*r[i+j*int((d-1+d-j-1)/2)-1]
    
    X=sp.Matrix(1,d,x)
    M=sp.Matrix(1,d,m)
    Det=Covar.det()
    CovInv=Covar**-1
    return sp.simplify(sp.exp(-(X-M)*CovInv*(X-M).T/2)/((2*sp.pi)**(d/2)*(Det**(1/2))))



def fisher_info(d,fixed_parameters=[]):
    """
    Calcul de l'information de Fisher formelle pour une gaussienne de dimension d,
    Possibilité de fixer des paramètres
    """
    #création des symboles sympy
    m=[sp.symbols('m%d'%i) for i in range(d)]
    s=[sp.symbols('s%d'%i) for i in range(d)]
    r=[sp.symbols('r%d'%i) for i in range(int(d*(d-1)/2))]
    
    #définition de la matrice de covariance
    Covar=sp.zeros(d,d)
    for i in range(d):
        for j in range(d):
            if i==j:
                Covar[i+d*j]=s[i]**2
            else:
                Covar[i+d*j]=s[i]*s[j]*r[i+j*int((d-1+d-j-1)/2)-1]
    CovInv=Covar**-1
    #calcul de I_F
    MuBloc=CovInv
    dCovar_s=[sp.diff(Covar,sigma) for sigma in s]
    dCovar_r=[sp.diff(Covar,ro) for ro in r]
    dCovar=dCovar_s+dCovar_r
    SigmaList=np.array([[np.trace(CovInv*dCovar[i]*CovInv*dCovar[j])/2 for i in range(int(d*(d+1)/2))] for j in range(int(d*(d+1)/2))])

    SigmaBloc=sp.Matrix(int(d*(d+1)/2),int(d*(d+1)/2),SigmaList.flatten())
    Fisher=sp.BlockDiagMatrix(MuBloc,SigmaBloc)
    Fisher2=Fisher.as_explicit()

    P=np.zeros((int(d*(d+3)/2),int(d*(d+3)/2)))
    for i in range(int(d*(d+3)/2)):
        if i<d:
            P[i][2*i]=1
        elif i<2*d:
            P[i][1+2*(i-d)]=1
        else:
            P[i][i]=1
    Psymp=sp.Matrix(P)
    Pinv=Psymp**-1
    #réorganisation selon notre paramétrisation ((mu_i,sigma_i),rho)
    Fisher_reorg=Pinv*Fisher2*Psymp
    symbols_reorg=[]
    #réarrangement des symboles sympy
    for i in range(d):
        symbols_reorg.append(m[i])
        symbols_reorg.append(s[i])
    for rho in r:
        symbols_reorg.append(rho)
    
    
    nindices=[]
    for sym in fixed_parameters:
        l=list(str(sym))

        if l[0]=='m':
            l.pop(0)
            index = 2*int(''.join(l))
            nindices.append(index)
        if l[0]=='s':
            l.pop(0)
            index = 2*int(''.join(l))+1
            nindices.append(index)
        if l[0]=='r':
            l.pop(0)
            index = 2*d + int(''.join(l))
            nindices.append(index)
    indices=[]
    for i in range(int(d*(d+3)/2)):
        if i not in nindices:
            indices.append(i)        
    #prise en comrandomDirpte des paramètres fixés
    Fisher_reorg=Fisher_reorg.extract(indices,indices)
    
    return [Fisher_reorg,symbols_reorg]


def christoffel(d,fixed_parameters=[]):
    inFish=fisher_info(d,fixed_parameters)
    
    Fisher=inFish[0]
    symbols=inFish[1]
    Finv=Fisher**-1
    fixed_symbols=[sp.symbols(parameter) for parameter in fixed_parameters]
    new_symbols=[]
    for sym in symbols:
        if sym not in fixed_symbols:
            new_symbols.append(sym)

    

    dim=len(new_symbols)
    lis=[]
    for k in range(dim):
        coord=sp.zeros(dim)
       
        for i in range(dim):
            for j in range(dim):
                ch=0
                for l in range(dim):
                    
                    a=sp.cancel(sp.diff(Fisher[i,l],new_symbols[j]))
                    b=sp.cancel(sp.diff(Fisher[j,l],new_symbols[i]))
                    c=sp.cancel(sp.diff(Fisher[i,j],new_symbols[l]))
                    s=(a+b-c)*Finv[l,k]/2
                    ch+=s
                coord[i,j]=sp.simplify(ch)
        
        lis.append(coord)
    return [lis,symbols]



def geo_shooting(christo, symbols, init_x, init_v, fixed_parameters=[], T=1.0, dt=0.01):
    n=len(symbols)
    fixed_symbols=[sp.symbols(parameter) for parameter in fixed_parameters]
    new_symbols=[]
    for sym in symbols:
        if sym not in fixed_symbols:
            new_symbols.append(sym)
    
    x=[]
    
    for key in init_x.keys():
        if key not in fixed_symbols:
            x.append(init_x[key])
    v=[]
    for value in init_v.values():
        v.append(value)

    
    y0=x+v

    dim=len(v)
    
    N=int(T/dt)

    values=init_x
    
    print(y0)
    def geodes(y,t):
        gamma=y
        
     
        for k in range(dim):
            values.update({new_symbols[k]:y[k]})
          
        

        dydt=[gamma[dim+k] for k in range(dim)]
      
        dydt.extend([-sum([sum([gamma[dim+i]*gamma[dim+j]*christo[k][i,j].evalf(subs=values) for j in range(dim)]) for i in range(dim)]) for k in range(dim)])
        return dydt
    
    time=np.linspace(0,T,N)
    
    sol = scp.integrate.odeint(geodes, y0, time)
    out=[]
    for el in sol:
        out.append(el[:dim])
    return out
    


#isodensity generator visual
def isodensity_visual2D(geo, N, v,  M=100, a=-4, b=4, sphere=False):

    plt.style.use('seaborn-v0_8')
    # fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))
    x = np.linspace(a, b, M)
    y = np.linspace(a, b, M)
    mesh=np.meshgrid(x, y)
    X, Y = mesh

    # for plotting
    lin=np.linspace(0.1,0.9,N)
    colors=[(lin[i],0,1-lin[i]) for i in range (N)]
    cmap = LinearSegmentedColormap.from_list("custom", ["blue", "red"], N)

    proxies = []
    labels = []


    geo_N=geo[::len(geo)//N]
    levels=[0.5,0.7,0.9]

    for i in range(N):
        
        distrib=geo_N[i]
        cov=distrib.getR()
        sigma=distrib.getSigma()
        
        # print(distrib.getParameter()[:4])

        level_modif=cov.computeDeterminant()
        
        new_levels=[lev/(2*np.pi*np.sqrt(level_modif)*sigma[0]*sigma[1]) for lev in levels]
        
        Z=[[distrib.computePDF(pt) for pt in l] for l in np.dstack((X,Y))]
    

        if sphere and i<N-1:
  

            plt.contour(X,Y,Z,new_levels,colors=cmap(i),alpha=0.3, lw=1)
    
        else:
            if i == 0:
                plt.contour(X,Y,Z,new_levels,colors=cmap(i))# , label="initial dist.")
                proxy_color = cmap(i)   # color of first level
                proxy = mlines.Line2D([], [], color=proxy_color)

                proxies.append(proxy)

                par = distrib.getParameter()
                
                # par is integer
                labels.append(f"initial dist. $N({int(par[0])}, {int(par[1])}, {int(par[2])}, {int(par[3])}, {par[4]})$")
                # labels.append(f"initial dist. $N({par[0]:.2}, {par[1]:.2}, {par[2]:.2}, {par[3]:.2}, {par[4]:.2})$")
                
            elif i == N-1:
                plt.contour(X,Y,Z,new_levels,colors=cmap(i), lw=2)# , label="final dist.")
                proxy_color = cmap(i)   # color of first level
                proxy = mlines.Line2D([], [], color=proxy_color)

                proxies.append(proxy)
                par = distrib.getParameter()
                labels.append(f"final dist. $N({par[0]:.2}, {par[1]:.2}, {par[2]:.2}, {par[3]:.2}, {par[4]:.2})$")

            plt.contour(X,Y,Z,new_levels,colors=cmap(i), lw=1) # , label="final dist.")
                

    plt.title(f"Geodesic curve in bivariate normal family", fontsize = 20)

    plt.legend(proxies, labels, fontsize=18, loc="lower right")

    plt.tick_params('both', labelsize=18)
    plt.xlabel(r'$x_1$', fontsize=22)
    plt.ylabel(r'$x_2$', fontsize=22)
    # plt.legend(fontsize=15)
    plt.axis("equal")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # for the figure's nametag
    str_v = " " 
    for vi in v:
        str_v += str(vi)

    # plt.savefig(f"/home/bketema/Python workspace/CodeStagePSchatz/figures/geod_{str_v}.pdf", dpi=600, bbox_inches='tight')
    plt.show()
        



def projections(simu,symbols,i,j,sphere=False,step=10):
    
    
    if not sphere:
        plt.plot(simu[:,i],simu[:,j])
        plt.scatter([simu[:,i][0]],[simu[:,j][0]],c='r')
    plt.scatter(simu[:,i][step::step],simu[:,j][step::step])
    plt.xlabel(symbols[i])
    plt.ylabel(symbols[j])
    plt.show()


    
    

    
    
