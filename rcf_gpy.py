import numpy as np
import scipy
import GPy

class RCF():
    """ built: 3/19/2024
    this an object of a Random-Contionus-Function (RCF), with-respect-to a gpy kernel
    RCF : IN -> OUT = R^(MO)
    we define a prior, and then sample to form a posterior.
    """

    def __init__(self, Domain:np.ndarray, MO:int=1, N:int=17, seed:int=777,
                 IN_noise=None, OUT_noise=None,
                 kernel=GPy.kern.RBF):
        """ !! note datatypes should be tf.float64 for stable Cholesky-operations
        GIVEN >
             Domain : 2d-np.ndarray (with shape=(d,2), with d=# of dims )
                  N : int (number-of-defining-points)
                 MO : int (Multiple-Output Dimension)
             **seed : int
           **kernel : GPy.kern
         **IN_noise : 1d-np.ndarray (len == Domain.shape[1])
        **OUT_noise : 1d-np.ndarray (len == MO)

        GET   >
            None
        """

        self.dtype  = np.float64
        self.IN     = Domain.astype(self.dtype)  ### : np.ndarray (IN-space range)
        self.N      = N      ### number of defining points
        self.MO     = MO     ### int (dimension of OUT)
        self.kernel = kernel(self.IN.shape[0])
        self.seed   = seed ### define pseudo-random seed

        np.random.seed( self.seed )

        ### define anisotropic i.i.d white-noise
        if IN_noise is None:
            self.IN_noise=np.zeros(self.IN.shape[0], dtype=self.dtype)
        else:
            self.IN_noise = IN_noise
        if OUT_noise is None:
            self.OUT_noise=np.zeros(self.MO, dtype=self.dtype)
        else:
            self.OUT_noise = OUT_noise

        ### define IN-space defining-points
        self.R_ix  = np.random.uniform(0,1, (self.N, self.IN.shape[0])).astype(self.dtype)
        self.R_ix *= (self.IN[:,1] - self.IN[:,0])
        self.R_ix += self.IN[:,0]

        ### compute cholesky-factorization
        ### this will fail if K is not-PSD LinAlgError: Matrix is not positive definite
        try:
            L_ij = np.linalg.cholesky( self.kernel.K( self.R_ix ) ) ## not immutable
        except:
            #print("not PSD added to diag")
            L_ij = np.linalg.cholesky( self.kernel.K( self.R_ix ) + np.diag( 1.e-8 * np.random.rand(self.N).astype(self.dtype) ) )

        ### compute OUT-space defining-points
        D_iX  = np.random.normal(0,1,(self.N, self.MO)).astype(self.dtype)
        D_iX *= np.diag(L_ij)[:,None]
        D_iX  = np.matmul(L_ij, D_iX)

        self.S_jX  = scipy.linalg.cho_solve((L_ij, True), D_iX)

    def __call__(self, D_ax):
        """ evaluate for arbitrary values/points in OUT given points in IN.
        GIVEN >
              self
              D_ax : 2d-np.ndarray (D_ax ∈ IN)
        GET   >
              D_aX : 2d-np.ndarray (D_aX ∈ OUT, note captial 'X')
        """
        D_ax += self.IN_noise*np.random.normal(0,1,D_ax.shape).astype(self.dtype)
        D_aX  = np.matmul( self.kernel.K(D_ax, self.R_ix), self.S_jX )
        D_aX += self.OUT_noise*np.random.normal(0,1,D_aX.shape).astype(self.dtype)
        return D_aX
