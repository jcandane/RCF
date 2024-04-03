import torch
import gpytorch

class RCF():
    """ built: 3/19/2024
    this an object of a Random-Contionus-Function (RCF), with-respect-to a gpytorch kernel
    RCF : IN -> OUT = R^(MO)
    we define a prior, and then sample to form a posterior.
    """

    def __init__(self, Domain, MO:int=1, N:int=17, seed:int=777,
                 IN_noise=None, OUT_noise=None,
                 kernel=gpytorch.kernels.RBFKernel()):
        """
        GIVEN >
            Domain  : 2d-torch.Tensor (domain of input points)
                 N  : int (number of points)
                MO  : int (Multiple-Output Dimension)
             **seed : int
           **kernel : gpytorch.kernels
        ** IN_noise : 1d-torch.Tensor
        **OUT_noise : 1d-torch.Tensor
        GET   >
            None
        """

        self.dtype  = torch.float64
        self.IN     = Domain.type(self.dtype) ### 2d-torch.Tensor
        self.N      = N      ### number of defining points
        self.MO     = MO     ### int (dimension of OUT)
        self.kernel = kernel
        self.seed   = seed ### define random sampling key

        torch.manual_seed(self.seed)

        ### define anisotropic i.i.d white noise
        if IN_noise is None:
            self.IN_noise = torch.zeros( self.IN.shape[0] , dtype=self.dtype)
        else:
            self.IN_noise = IN_noise
        if OUT_noise is None:
            self.OUT_noise = torch.zeros( self.MO , dtype=self.dtype)
        else:
            self.OUT_noise = OUT_noise

        ### find a series of random defining points,
        ### keep looping until we find a stable configuration of initial-points
        ### --> "A not p.d., added jitter of 1.0e-08 to the diagonal" pytorch safety
        self.R_ix  = torch.rand(N, self.IN.shape[0], dtype=self.dtype)
        self.R_ix *= torch.diff(self.IN, axis=1).reshape(-1)
        self.R_ix += self.IN[:,0]

        ### compute cholesky-factorization
        L_ij       = torch.linalg.cholesky( self.kernel(self.R_ix) ).to_dense()

        ### compute OUT-space defining-points
        D_iX       = torch.normal(0, 1, size=(self.N, self.MO), dtype=self.dtype)
        D_iX      *= torch.diag(L_ij).reshape(-1,1)
        D_iX       = torch.matmul(L_ij, D_iX)

        ### compute (L \ D) used to interpolate arbtirary points
        self.S_jX  = torch.cholesky_solve(D_iX, L_ij)

    def __call__(self, D_ax):
        """ evaluate for arbitrary values/points in OUT given points in IN.
        GIVEN >
              self
              D_ax : 2d-torch.Tensor (D_ax ∈ IN)
        GET   >
              D_aX : 2d-torch.Tensor (D_aX ∈ OUT, note captial 'X')
        """
        D_ax += self.IN_noise*torch.normal(0, 1, size=D_ax.shape, dtype=self.dtype)
        D_aX  = torch.matmul(self.kernel(D_ax, self.R_ix), self.S_jX)
        D_aX += self.OUT_noise*torch.normal(0, 1, size=D_aX.shape, dtype=self.dtype)
        return D_aX
