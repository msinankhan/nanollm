import torch
from torch import Tensor
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G:Tensor, steps:int) -> Tensor:
    assert G.ndims>=2
    a,b,c=(3.4445,-4.7750,2.0315)

    X=G.bfloat16()

    if G.size(-2)>G.size(-1):  # We are comparing the row vs column size, if the matrix is taller instead of being wide, we end up with a bigger matrix, and hence more computation.
        X=X.mT                 # As we compute A=X@X.T, the resultant A is a sq. matrix, ( remember, dimension of A= row of X and column of X^T) 
                               # hence the A becomes bigger if row is bigger than column in X.


        X=X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7) # Here we are calculating the Frobenius norm, which helps the Newton Schulz iteration converge by ensuring ||I-X.X^T||<1. 
                                                    #It basically shrinks Eigenvalues to a safe range. The (+1e-7) term simply prevents division by 0.


        #NEW-SCHULZ Iteration:
        """Here we are trying to approximate the ratatory vectors without performing SVD
        X(X^T.X)^-1/2 approximates to U.V^T
        
        But taking a square root is also computationally expensive, hence we rely on a 5th order Polynomial approximation technique to calculate it i.e computing    X←(aI+bA+cA2)X
        
        Also, because we are approximating the square root, we will still have singular values, but they end up in a Uniform(0.5, 1.5), which is good enough for learing.
        
        """

        for _ in range(steps):
            A=X@X.mT
            B=b*A+c*A@A
            X=a*X+B@A


        if G.size(-2)>G.size(-1):
            X=X.mT      #Undo earlier transpose.

        return X # It returns ≈U.V^T (not exact, but directionally correct.)
    

class Muon(torch.optim.Optimizer):

    def __init__(self,params,lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults=dict(lr=lr, momentum=momentum, nesterov=nesterov,ns_steps=ns_steps)

        params:list[Tensor]=[*params]

        param_groups=[]

        for size in {p.numel() for p in params}:
            group=dict(params=[p for p in params if p.numel()==size])