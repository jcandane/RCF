from jax import config
config.update("jax_enable_x64", True)

import jax
import gpjax as gpx

class RCF():
    """ built: 3/19/2024
    this an object of a Random-Contionus-Function (RCF), with-respect-to a gpJAX kernel
    RCF : IN -> OUT = R^(MO)
    we define a prior, and then sample to form a posterior.
    """

    def __init__(self, Domain, N:int, MO:int=1, seed:int=777,
                 IN_noise=None, OUT_noise=None,
                 kernel=gpx.kernels.RBF() ):
        """ initialize RCF object
        GIVEN >
             Domain : 2d-jax.Array (domain of input points, shape=(n-dimenions, 2))
                  N : int (number of points)
                 MO : int (Multiple-Output Dimension)
             **seed : int (opinonal, integer to define JAX PRNGKey random-key)
           **kernel : (opinonal, defaults to gpJAX's RBF kernel)
         **IN_noise : 1d-jax.Array (opinonal)
        **OUT_noise : 1d-jax.Array (opinonal)
        GET >
            None
        """
        self.dtype  = jax.numpy.float64
        self.IN     = Domain.astype(self.dtype) ### 2d-jax.Array
        self.N      = N      ### number of defining points
        self.MO     = MO     ### int (dimension of OUT)
        self.kernel = kernel
        self.seed   = seed

        self.key    = jax.random.PRNGKey(self.seed) ### define random sampling key
        ### define anisotropic i.i.d white-noise
        if IN_noise is None:
            self.IN_noise=jax.numpy.zeros(self.IN.shape[0], dtype=self.dtype)
        else:
            self.IN_noise = IN_noise
        if OUT_noise is None:
            self.OUT_noise=jax.numpy.zeros(self.MO, dtype=self.dtype)
        else:
            self.OUT_noise = OUT_noise

        ### find a series of random defining points, keep looping until we find a stable configuration of initial-points
        c_i        = jax.numpy.diff(self.IN, axis=1).reshape(-1)
        self.R_ix  = c_i[None,:]*jax.random.uniform(self.key, (N, self.IN.shape[0]), dtype=self.dtype)
        self.R_ix += self.IN[:,0][None,:]

        Σ_ij      = self.kernel.gram(self.R_ix).A
        L_ij = jax.numpy.linalg.cholesky(Σ_ij)
        if jax.numpy.sum( jax.numpy.isnan(L_ij).astype( jax.numpy.int32 ) )==0:
            None
        else: ### if cholesky-factorization fails... add random diagonal
            L_ij = jax.numpy.linalg.cholesky( Σ_ij + jax.numpy.diag( jax.random.uniform( self.key, (self.N, ) , dtype=self.dtype) ) ) ## not immutable
        ###

        D_iX  = jax.random.normal( self.key, (self.N,self.MO) , dtype=self.dtype)
        D_iX *= jax.numpy.diag(L_ij)[:,None] #*jax.numpy.ones(self.MO, dtype=self.dtype)[None,:]
        D_iX  = L_ij @ D_iX

        ## correlate D_iX using the Cholesky-factor, yielding random/correlated normal-samples
        self.S_iX = jax.scipy.linalg.cho_solve((L_ij, True), D_iX)

    def evaluate(self, D_ax):
        """ evaluate for arbitrary values/points in OUT given points in IN.
        GIVEN >
              self
              D_ax : 2d-jax.Array (D_ax ∈ IN)
        GET   >
              D_aX : 2d-jax.Array (D_aX ∈ OUT, note captial 'X')
        """
        D_ax += self.IN_noise*jax.random.normal(self.key, D_ax.shape, dtype=self.dtype)
        D_aX  = self.kernel.cross_covariance(D_ax, self.R_ix) @ self.S_iX
        D_aX += self.OUT_noise*jax.random.normal(self.key, D_aX.shape, dtype=self.dtype)
        return D_aX