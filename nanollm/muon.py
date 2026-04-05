import torch
from torch import Tensor
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G:Tensor, steps:int) -> Tensor:
    assert G.ndim>=2
    a,b,c=(3.4445,-4.7750,2.0315)

    X=G.bfloat16()

    if G.size(-2)>G.size(-1):  # We are comparing the row vs column size, if the matrix is taller instead of being wide, we end up with a bigger matrix, and hence more computation.
        X=X.mT                 # As we compute A=X@X.T, the resultant A is a sq. matrix, ( remember, dimension of A= row of X and column of X^T) 
                               # hence the A becomes bigger if row is bigger than column in X.


    X=X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7) # Here we are calculating the Frobenius norm, which helps the Newton Schulz iteration converge by ensuring ||I-X.X^T||<1. 
                                                    #It basically shrinks Eigenvalues to a safe range. The (+1e-7) term simply prevents division by 0.


        #NEW-SCHULZ Iteration:
    """
        Here we are trying to approximate the ratatory vectors without performing SVD
        X(X^T.X)^-1/2 approximates to U.V^T
        
        But taking a square root is also computationally expensive, hence we rely on a 5th order Polynomial approximation technique to calculate it i.e computing X←(aI+bA+cA2)X
        
        Also, because we are approximating the square root, we will still have singular values, but they end up in a Uniform(0.5, 1.5), which is good enough for learing.
        
    """

    for _ in range(steps):
        A=X@X.mT
        B=b*A+c*A@A
        X=a*X+B@X


    if G.size(-2)>G.size(-1):
        X=X.mT      #Undo earlier transpose.

    return X # It returns ≈U.V^T (not exact, but directionally correct.)
    

class Muon(torch.optim.Optimizer):

    def __init__(self,params,lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults=dict(lr=lr, momentum=momentum, nesterov=nesterov,ns_steps=ns_steps)

        params:list[Tensor]=[*params]

        param_groups=[]

        for size in {p.numel() for p in params}: #p.numel() returns the number of elements in the tensor.
            group=dict(params=[p for p in params if p.numel()==size])
            param_groups.append(group)

        super().__init__(param_groups,defaults)


    @torch.no_grad
    def step(self):
        for group in self.param_groups:
            params: list[Tensor]=group['params']
            for p in params:
                g=p.grad
                
                assert g is not None
                state= self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"]=torch.zeros_like(g)  #Initially momentum is initialized as a tensor of the same shape and 
                                                                  #data size as the gradient tensor g.

                buf: Tensor= state["momentum_buffer"] # Momentum is a method that smoothens the optimization trajectory 
                                                      # by adding a term that helps the optimizer remember the past gradients.
                                                      # It adds a fraction of the previous update vector to the current gradient

                buf.lerp_(g,1-group["momentum"]) # Performs linear interpolation, 
                                                 # [torch.lerp(input, end, weight, *, out=None) => out i ​= start i ​+ weight i ​×(end i ​−start i ) ] i.e 
                                                 # (buf←buf+(1−μ)(g−buf) which re-arranges becomes, buf=μbuf+(1-μ)g ;  
                                                 # note: μ= group["momentum"]

                g=g.lerp_(buf,group["momentum"]) if group ["nesterov"] else buf  #Nesterov momentum is an advanced form of momentum-based optimization.
                                                                                # It modifies the update rule by calculating the gradient at the upcoming position 
                                                                                # rather than the current position of the weights.

                g=zeropower_via_newtonschulz5(g,steps=group["ns_steps"])

                p.add_(g,alpha=-group["lr"]*max(1, p.size(-2)/p.size(-1))**0.5)   # ASPECT RATIO: 
                                                                                  # Orthogonal matrices have norm proportional to √(rows)
                                                                                  # **Wide vs tall matrices behave differently**
                                                                                  # The  √max(1,m/n) factor ensures:
                                                                                  # No bias toward tall matrices
                                                                                  # Step magnitude comparable across shapes




class DistMuon(torch.optim.Optimizer):
    def __init__(self, params, lr:float=0.02, momentum:float=0.95, nesterov:bool=True, ns_steps:int=5):
        defaults=dict(lr=lr, momentum=momentum, nesterov=nesterov,ns_steps=ns_steps)
        params=list(params)

        assert all(p.ndim==2 for p in params), "Muon expects 2D parameters only."

        rank=dist.get_rank()

        shapes=sorted({p.shape for p in params})

        param_groups=[]

        for shape in shapes:
            group_params=[p for p in params if p.shape==shape]
            device,dtype=group_params[0].device, group_params[0].dtype
            assert all(p.device==device for p in group_params)
            assert all(p.dtype==dtype for p in group_params)

            if rank==0:
                print(f"Muon: Grouping {len(group_params)} of shape {shape}, {device}, and dtype: {dtype}")

            param_groups.append( dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0]) , kind='muon' ) )    # Zero Buffer is a placeholder tensor
                                                                                                            # Used to pad reduce_scatter / all_gather calls
                                                                                                            # Required when number of params ≠ world_size

        super().__init__(param_groups,defaults)


    @torch.no_grad
    def step(self):

        """
        Gradients are averaged across GPUs.
        Momentum is computed on one “owner” rank per parameter.
        Updated parameters are replicated back to all ranks.
        """

        rank=dist.get_rank()
        world_size=dist.get_world_size()

        assert (p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads."


        # Kick off all the reduce scatter operations to average up the gradients across all ranks - Gradient Averaging
        all_reduce_futures=[]

        for group in self.param_groups:
            params=group["params"]

            zero_buffer=group["zero_buffer"]

            for base_i in range(0,len(params),world_size):
                owner_idx=base_i+rank

                rs_input=[p.grad for p in params[base_i:base_i+world_size]]

                rs_input.extend([zero_buffer]* (world_size-len(rs_input)))

                rs_output=params[owner_idx].grad if owner_idx< len(params) else torch.empty_like(zero_buffer)

                work=dist.reduce_scatter(rs_output,rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()

                all_reduce_futures.append(work)



        # Now each rank computes the update and gathers

        future_idx=0
        all_gather_futures=[]

        for group in self.param_groups:
            params=group["params"]
            zero_buffer=group["zero_buffer"]

            for base_i in range(0,len(params), world_size):
                owner_idx=base_i + rank
                all_reduce_futures[future_idx].wait()     #Orthogonalization is expensive
                future_idx+=1                             # Doing it on every GPU is wasteful
                                                          # So: one GPU computes, others wait
                

                if owner_idx<len(params):
                    p=params[owner_idx]
                    g=p.grad

                    if g is None:
                        # This would be a real error — this rank owns this param but has no grad
                        raise RuntimeError(
                            f"Rank {rank}: owner param at index {owner_idx} has no grad. "
                            f"Did you call loss.backward() before step()?"
                        )
                    
                    state=self.state[p]

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)

                    buf: Tensor=state["momentum_buffer"]
                    buf.lerp_(g,1.0-group["momentum"])

                    g=g.lerp_(buf,group["momentum"]) if group["nesterov"] else buf

                    g=zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                    # if g is None:
                    #     print(f"Rank {rank}: p.grad is None for param at index {owner_idx}")
                    #     continue

                    scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)

                    p.add_(g,alpha=-group["lr"]*scale)

                ag_input=params[owner_idx] if owner_idx<len(params) else zero_buffer
                ag_output=params[base_i:base_i+world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range (world_size-len(ag_output))]) #pad
                work=dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        torch.futures.collect_all(all_gather_futures).wait()




        