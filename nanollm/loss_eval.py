import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model,batches,steps,token_bytes):

    """Computes bits per bytes instead of average loss which helps us compare models across varying vocab_size.
        The way this works is that instead of just
        calculating the average loss as usual, you calculate the sum loss, and independently
        also the sum bytes (of all the target tokens), and divide.
        
         This normalizes the loss by the number of bytes that the target tokens represent."""

    total_nats=torch.tensor(0.0,dtype=torch.float32,device=model.get_device())
    total_bytes=torch.tensor(0, dtype=torch.int64, device=model.get_device())

    batch_iter=iter(batches)
    for _ in range(steps):
        x,y =next(batch_iter)
        loss2D=model(x,y,loss_reduction='none') #(B,T)
        loss2D=loss2D.view(-1) #(B*T)
        y=y.view(-1) # Flatten

        if (y.int()<0).any():
            valid  = y>=0
            y_safe = torch.where(valid, y, torch.zeros_like(y)) #torch.where=(condition,this,other) chooses an element from this or other based on the condition.
                                                                # here, it choose 0 where ever y is less than 0.

            num_bytes2D=torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y,dtype=token_bytes.dtype)
            )

            total_nats+=(loss2D*(num_bytes2D>0)).sum()
            total_bytes+=num_bytes2D.sum()

        else:
            num_bytes2D=token_bytes[y]
            total_nats+=(loss2D*(num_bytes2D>0)).sum()
            total_bytes+=num_bytes2D.sum()

    world_size= dist.get_world_size() if dist.is_initialized() else 1
    if world_size>1:
        dist.all_reduce(total_nats,op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes,op=dist.ReduceOp.SUM)


    total_nats=total_nats.item()
    total_bytes=total_bytes.item()

    if total_bytes==0:
        return float('inf')
    bpb=total_nats/(math.log(2) *total_bytes)

    return bpb
        




