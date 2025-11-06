import openturns as ot
import numpy as np
import matplotlib.pyplot as plt
import os

from normal import Normal
from truncatedDistribution import TruncatedDistribution

import previous_modules.truncated_Gaussians as trGauss
import previous_modules.xtrnormal as xtrnorm

from multiprocessing import Pool


def parallel_geodesic(args):
    """
    helper function for computing geodesics in parallel
    """
    v, h, interval, m, s = args

    lower = interval.getLowerBound()[0]
    upper = interval.getUpperBound()[0]

    X_0 = np.array([m, s, v[0], v[1]])

    m_t, s_t = xtrnorm.geodesic(1, h, X_0, a=lower, b=upper)

    return m_t[-1], s_t[-1]



# from ChatGPT: allows to avoid starting from the same random number for rejection sampling
def init_rng():
    """
    Runs *once* in every worker.  
    Makes an independent NumPy Generator called `rng`.
    """
    global rng
    # SeedGenerator guarantees independent streams:
    seed_seq = np.random.SeedSequence([os.getpid(), os.getppid()])
    rng = np.random.default_rng(seed_seq)

def parallel_sampling(args):
    """
    helper function for sampling from an ext. tr. normal dist. using rejection
    sampling from the uniform distribution. To give to multiprocessing.Pool for
    parallelizing the sampling procedure.
    """
    k, s, a, b = args
    
    # the maximum of the density (a convex function) is always attained on the 
    # boundary
    bound = (b-a)*max(xtrnorm.f(b, k, s, a, b), xtrnorm.f(a, k, s, a, b))

    while True:
        u = np.random.uniform(0, 1)
        x = rng.uniform(a, b)

        if u < xtrnorm.f(x, k, s, a, b)/bound:
            return x

# args = (2, -1, -1, 1)
# print(parallel_sampling(args))

class ExtendedTrNormal:
    """
    Distribution class based on OpenTURNS where we extend the truncated normal family parametrized by the 
    (almost) exponential parameters (k, s) 
    """

    def __init__(self, k, s, lowerBound, upperBound, description='X0'):
        
        self.k = k
        self.s = s
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.description = description

        #self.mu = ExtendedTrNormal._changeOfVariablesBack(k, s)[0]
        #self.sigma = ExtendedTrNormal._changeOfVariablesBack(k, s)[1]

        self._tr_interval = ot.Interval(self.lowerBound, self.upperBound)

    def __repr__(self):
        return f"ExtendedTrNormal(k={self.k}, s={self.s})\n"

    @staticmethod
    def _changeOfVariablesForth(mu, sigma):
        return (mu/sigma**2, 1/sigma**2)
    
    @staticmethod
    def _changeOfVariablesBack(k, s):
        return (k/s, 1/np.sqrt(s))
    
    @staticmethod
    def _jacobian(k, s):
        j = np.zeros((2, 2))
        j[0, 0] = 1/s
        j[0, 1] = -k/s**2
        j[1, 0] = 0
        j[1, 1] = -s**(-3/2)/2

        return j

    @staticmethod
    def sinh(x):
        return (np.exp(x) - np.exp(-x))/2

    def getParameter(self, with_bounds=False):
        if with_bounds:
            return [self.k, self.s, self.lowerBound, self.upperBound]
        else:
            return [self.k, self.s]
        
    def setDescription(self, description: ot.Description):
        self.description = description

    def getDescription(self):
        return self.description

    def computePDF(self, x):

        k = self.k
        s = self.s
        a = self.lowerBound
        b = self.upperBound

        return xtrnorm.f(x, k, s, a, b)
    
    def computeCDF(self, x):
        pass

    def computeLogPDFGradient(self, x)-> ot.Sample:
        return xtrnorm.logpdf_gradient(x, self.k, self.s, a=self.lowerBound, b=self.upperBound)
    
    def getSample(self, sampleSize):
        # compute a sample from the extended truncated normal density
        # using rejection sampling or using the quantile transform
        
        s = self.s
        k = self.k

        if s > 0:
            # returns the openturns sample from the corresponding truncated normal distribution
            mu, sigma = k/s, 1/np.sqrt(s) 
            N = ot.Normal(mu, sigma)
            dist = ot.TruncatedDistribution(N, self._tr_interval)
            #dist.setDescription(self.getDescription())
            return dist.getSample(sampleSize)
        
        else:
            args = [(self.k, self.s, self.lowerBound, self.upperBound) for _ in range(sampleSize)]

            # else do rejection sampling in parallel
            # (when on HPC, should change the os.cpu_count())
            with Pool(processes=os.cpu_count(), initializer=init_rng) as pool:
                sample = pool.map(
                    parallel_sampling, args, chunksize=1
                    )

            s = ot.Sample(np.asarray(sample).reshape(-1, 1))
            #print("desc=", type(self.description))
            #s.setDescription(ot.Description(['X0']))
            return s

# if __name__ == "__main__":
#     f = ExtendedTrNormal(0, -1, 1, 2)
#     s = f.getSample(5000)

#     line = np.linspace(0, 3, 200)
#     val = [f.computePDF(x) for x in line]
#     plt.plot(line, val)
#     plt.hist(np.array(s).transpose()[0], bins=40, density=True)
#     plt.show()

#     print('sample=\n', x)
#     print('logpdfgrad=\n', f.computeLogPDFGradient(x))
#     quit()

    def fisherInformation(self):
        # use the fisherInformation for the truncated normal family with the change of coordinates ?
        """
        See the Supplementary Materials of the conference paper 'Geodesic non-completeness of the truncated
        normal family' (GSI'2025) 
        """

        k = self.k
        s = self.s
        a = self._tr_interval.getLowerBound()[0]
        b = self._tr_interval.getUpperBound()[0]

        if s>0:
            # do the computation in (mu, sigma) coordinates
            mu, sigma = ExtendedTrNormal._changeOfVariablesBack(k, s)
            n = Normal(mu, sigma)
            n_tr = TruncatedDistribution(n, ot.Interval(a, b))
            I_theta = n_tr.fisherInformation()
            J = ExtendedTrNormal._jacobian(k, s)
            J_t = J.transpose()

            return J_t@I_theta@J
        
        else:
            return xtrnorm.information(k, s, a=a, b=b)

# if __name__ == "__main__":
#     from normal import Normal
#     from truncatedDistribution import TruncatedDistribution
#     k = 0
#     s = 0.1
#     a = -1
#     b = 1
#     f = ExtendedTrNormal(k, s, a, b)
#     I_xi = f.fisherInformation()
#     #print("I_theta=\n", I_theta)
#     J = ExtendedTrNormal._jacobian(k, s)
#     J_t = J.transpose()
    
#     #print(s)
#     mu, sigma = ExtendedTrNormal._changeOfVariablesBack(k, s)
#     # print("mu, sigma=", mu, sigma)
#     n = Normal(mu, sigma)
#     n_tr = TruncatedDistribution(n, ot.Interval(a, b))
#     I_theta = n_tr.fisherInformation()
#     print("I_xi=\n ", I_xi)
#     print("product mat=\n", J_t@I_theta@J)
#     line = np.linspace(-2, 2, 200)
#     f_val = [f.computePDF(x) for x in line]
#     #n_val = 
#     quit()

    def exponentialMap(self, v, h=0.01):
        
        k = self.k
        s = self.s

        Y_0 = np.array([k, s, v[0], v[1]])

        k_t, s_t = xtrnorm.geodesic(1, h=0.005, Y_0=Y_0)

        #return ExtendedTrNormal(k+v[0], s+v[1], self.lowerBound, self.upperBound)
        return ExtendedTrNormal(k_t[-1], s_t[-1], self.lowerBound, self.upperBound)
    
    
if __name__ == "__main__":
    k = 0
    s = 1
    a = -1
    b = 1
    
    # dist in the ext fam
    f = ExtendedTrNormal(k, s, a, b)
    # dist in the trunc normal fam
    mu, sigma = ExtendedTrNormal._changeOfVariablesBack(k, s)
    n = Normal(mu, sigma)
    n_tr = TruncatedDistribution(n, ot.Interval(a, b))

    # jacobian for change of var
    J = ExtendedTrNormal._jacobian(k, s)
    J_t = J.transpose()
    J_inv = np.linalg.inv(J)

    # v = J_inv@np.array([0, 1])
    # tangent vector in (k,s) coordinates
    v = np.array([0, 1])

    # same vector in (mu, sigma) coordinates
    w = J@v
    print("w=", w)
    f_final = f.exponentialMap(v)
    n_final = n_tr.exponentialMap(w)

    print(f"({f_final.k}, {f_final.s})")
    param_final = n_final._custom_base_distribution.getParameter()
    print(ExtendedTrNormal._changeOfVariablesForth(param_final[0], param_final[1]))
#    print(n_final)

    line = np.linspace(-1.2, 1.2, 200)
    f_val = [f_final.computePDF(x) for x in line]
    n_val = [n_final.computePDF(x) for x in line]
    f_init = [f.computePDF(x) for x in line]

    plt.plot(line, f_val)
    plt.plot(line, n_val)
    plt.plot(line, f_init, color = "black")
    plt.show()
    quit()

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     f = ExtendedTrNormal(0, 0, -1, 1)
#     g = f.exponentialMap(v)
#     print(f)
#     print(g)
#     x = np.linspace(-1.2, 1.2, 100)

#     plt.plot(x, [g.computePDF(t) for t in x])
#     plt.plot(x, [f.computePDF(t) for t in x])
#     plt.show()
#     quit()

    def _tangentVectors(self, interval, delta, nbPts):   

        J = self.fisherInformation()
        L = np.linspace(0, 2*np.pi, nbPts, endpoint=False)
        Vects = []

        #TODO parallelize ?
        for t in L:
            v = np.array([np.cos(t),np.sin(t)])
            l_J = np.sqrt(np.dot(np.transpose(v),np.dot(J,v)))
            v_J = delta*(v/l_J)
            Vects.append(v_J)
        
        return Vects
    
    def sampleFisherRaoSphere(self, delta, nbPts, *args, **kwargs):
        """
        sample from the Fisher Rao sphere
        """

        I = self.fisherInformation()
        interval = self._tr_interval

        # the angles must be sampled uniformly according to the riemannian angle
        Vects = self._tangentVectors(interval, delta, nbPts)
    
        if kwargs:    
            args = [(v, kwargs['h'], interval, self.k, self.s) for v in Vects]

        else:
            args = [(v, 0.01, interval, self.k, self.s) for v in Vects]

        with Pool(processes=os.cpu_count()) as pool:
            sphereParams = pool.map(
                parallel_geodesic, args, chunksize=1
            )
  
        spherePointsList = []
        for m, s in sphereParams:
            spherePointsList.append(ExtendedTrNormal(m, s, self.lowerBound, self.upperBound))
            
        return spherePointsList


# if __name__ == "__main__":
#     # print(f(0.1, 0, 1))
#     quit()
#     from beta import Beta


#     a, b = -1, 1
#     k, s = 0, 1 #ExtendedTrNormal._changeOfVariablesForth(0, 1)


#     beta_dens = Beta(1, 1, a, b)


#     f = ExtendedTrNormal(k, s, lowerBound=a, upperBound=b)
    
#     v1 = 0
#     v2 = -1

#     nbPts= 100

#     K = np.zeros(15)
#     S = np.linspace(-1, 1, 15)

#     #sphere = [ExtendedTrNormal(K[i], S[i], a, b) for i in range(len(K))]
#     delta = 0.7

#     sphere = f.sampleFisherRaoSphere(delta = delta, nbPts=nbPts, h=0.01) #delta=delta, nbPts=nbPts)
#     #sphere_beta = beta_dens.sampleFisherRaoSphere(delta=delta, nbPts=nbPts)
    
#     # print(mean(k, s))
#     # print(second_moment(k, s))
#     # print(skew(k, s))
#     # print(kurtosis(k, s))

#     # print(I)
#     # print("det = ", np.linalg.det(I))

#     c = [(1, 0, 0) for x in np.linspace(0.2, 1, nbPts)]

#     # print(christoffel_symbols(0, 0))

#     plt.figure(figsize=(10, 6))
#     X = np.linspace(a-0.2, b+0.2, 300)
    
#     for l ,g in enumerate(sphere):
#         if l%10 == 0:
#             plt.plot(X, [g.computePDF(x) for x in X], color = c[0], lw = 2, alpha = 0.55)

#     # for l, g in enumerate(sphere_gtr):
#     #     if l%1 == 0:
#     #         plt.plot(X, [g.computePDF(x) for x in X], lw = 2, color = c[l], alpha=0.5)#, label=f"(k,s)=({g.k:.2f}, {g.s:.2f})")

#     plt.plot(X, [f.computePDF(x) for x in X], lw=2.5, color = "dodgerblue", label="nominal")
#     #plt.ylim(-0.1, 2.3)

#     plt.legend()
#     plt.tick_params(axis='both', labelsize=12) 
#     #plt.savefig("/home/bketema/Documents/CATHARE/prez DT figures/uniform_density.pdf", dpi=600, bbox_inches='tight')

#     plt.show()


    # plt.figure(figsize=(10, 6))
    # X = np.linspace(a-0.2, b+0.2, 300)


    # for l, g in enumerate(sphere_beta):
    #     if l%10 == 2:
    #         plt.plot(X, [g.computePDF(x) for x in X], lw = 2, color = c[l], alpha=0.5)#, label=f"(k,s)=({g.k:.2f}, {g.s:.2f})")

    # plt.plot(X, [f.computePDF(x) for x in X], lw=2.5, color = "dodgerblue", label="nominal")
    # plt.ylim(-0.1, 2.3)


    # plt.legend()
    # plt.tick_params(axis='both', labelsize=12) 
    # plt.savefig("/home/bketema/Documents/CATHARE/prez DT figures/uniform_density_beta.pdf", dpi=600, bbox_inches='tight')
    # plt.show()


##########################################
######### spheres #######################
########################################

    # plt.figure(figsize=(12, 7))
    # plt.grid(True, which='both', axis='both')

    # K = [g.k for g in sphere_gtr]
    # S = [g.s for g in sphere_gtr]

    # K.append(sphere_gtr[0].k)
    # S.append(sphere_gtr[0].s)

    # plt.xlabel("k", fontsize=12)
    # plt.ylabel("s", fontsize=12)
    
    # #plt.title("sphere in the (k,s)-parametrization")
    # plt.scatter([f.k], [f.s], label=f"(k,s)=({f.k}, {f.s})")
    # plt.plot(K, S, lw=2, label=f"{delta}-sphere")
    # plt.tick_params(axis='both', labelsize=12)  
    # plt.legend()

    # # plt.savefig("/home/bketema/Documents/CATHARE/prez DT figures/uniform_param.pdf", dpi=600, bbox_inches='tight')

    # plt.show()


    # plt.figure(figsize=(10, 6))
    
    # M = [g.mu for g in sphere].append(sphere[0].mu)
    # Sig = [g.sigma for g in sphere]

    # plt.title("sphere in the (mu, sigma)-parametrization")
    # plt.scatter([f.mu], [f.sigma])
    # plt.plot(M, Sig)
    # plt.show()



####### Testing the parallel sampling #############

if __name__ == "__main__":
    from timeit import default_timer
    f = ExtendedTrNormal(1, -5, -2, -1)
    
    start = default_timer()
    sample = f.getSample(10000)
    s = np.array(sample).transpose()[0]
    stop = default_timer()
    print(f"It took {stop - start} seconds")
    
    X = np.linspace(-2, -1, 100)
    plt.plot(X, [f.computePDF(x) for x in X])

    plt.hist(s, bins=40, density=True)
    plt.show()

    log_grad = f.computeLogPDFGradient(sample)
    arr = np.array(log_grad).transpose()[0]
    print(np.mean(arr))

    n = ot.Normal()
    s = n.getSample(10000)
    log_grad = n.computeLogPDFGradient(s)
    arr = np.array(log_grad).transpose()[0]
    print(np.mean(arr))
    print(arr@arr/len(arr))
    quit()

####################################################
