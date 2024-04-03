import tensorflow as tf
import gpflow

class RCF():
    """ built: 3/19/2024
    this an object of a Random-Contionus-Function (RCF), with-respect-to a gpflow kernel
    RCF : IN -> OUT = R^(MO)
    we define a prior, and then sample to form a posterior.
    """

    def __init__(self, Domain:tf.Tensor, MO:int=1, N:int=17, seed:int=777,
                 IN_noise=None, OUT_noise=None,
                 kernel=gpflow.kernels.SquaredExponential()):
        """ !! note datatypes should be tf.float64 for stable Cholesky-operations
        GIVEN >
             Domain : 2d-tf.Tensor (with shape=(d,2), with d=# of dims )
                  N : int (number-of-defining-points)
                 MO : int (Multiple-Output Dimension)
             **seed : int
           **kernel : gpflow.kernels
         **IN_noise : 1d-tf.Tensor (len == Domain.shape[1])
        **OUT_noise : 1d-tf.Tensor (len == MO)
        GET   >
            None
        """

        self.dtype  = tf.float64 ## datatype= float64 for stable cholesky-factor
        self.IN     = tf.cast(Domain, self.dtype)  ### : tf.Tensor (IN-space range)
        self.N      = N      ### number of defining points
        self.MO     = MO     ### int (dimension of OUT)
        self.kernel = kernel
        self.seed = seed ### define pseudo-random seed

        tf.random.set_seed( self.seed )

        ### define anisotropic i.i.d white-noise
        if IN_noise is None:
            self.IN_noise=tf.zeros(self.IN.shape[0], dtype=self.dtype)
        else:
            self.IN_noise = IN_noise
        if OUT_noise is None:
            self.OUT_noise=tf.zeros(self.MO, dtype=self.dtype)
        else:
            self.OUT_noise = OUT_noise

        ### define IN-space defining-points
        self.R_ix  = tf.random.uniform( (self.N, self.IN.shape[0]) , dtype=self.dtype)
        self.R_ix *= (self.IN[:,1] - self.IN[:,0])
        self.R_ix += self.IN[:,0]

        ### compute cholesky-factorization
        L_ij = tf.linalg.cholesky( self.kernel.K(self.R_ix) )
        if tf.reduce_sum( tf.cast( tf.math.is_nan(L_ij) , tf.int32 ), [0,1] )==0:
            None
        else: ### if cholesky-factorization fails... add small random diagonal
            L_ij = tf.linalg.cholesky( self.kernel.K(self.R_ix) + tf.linalg.diag( tf.random.uniform( (self.N, ) , dtype=self.dtype) ) )

        ### compute OUT-space defining-points
        D_iX  = tf.random.normal((self.N, self.MO), dtype=self.dtype)
        D_iX *= tf.linalg.diag_part(L_ij)
        D_iX  = tf.matmul(L_ij, D_iX)

        ### compute (L \ D) used to interpolate arbtirary points
        self.S_iX  = tf.linalg.cholesky_solve(L_ij, D_iX)

    def __call__(self, D_ax):
        """ evaluate for arbitrary values/points in OUT given points in IN.
        GIVEN >
              self
              D_ax : 2d-tf.Tensor (D_ax ∈ IN)
        GET   >
              D_aX : 2d-tf.Tensor (D_aX ∈ OUT, note captial 'X')
        """
        D_ax += self.IN_noise*tf.random.normal(D_ax.shape, dtype=self.dtype)
        D_aX  = tf.matmul(self.kernel(D_ax, self.R_ix), self.S_iX)
        D_aX += self.OUT_noise*tf.random.normal(D_aX.shape, dtype=self.dtype)
        return D_aX
