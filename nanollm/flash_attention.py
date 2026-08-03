import torch
import torch.nn.functional as F

def _load_flash_attention_3():
    if not torch.cuda.is_available():
        return None

    try:
        major,_ = torch.cuda.get_device_capability()

        import os
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = "1"

        from kernels import get_kernel, has_kernel

        if major ==9:
            hf_kernel= "varunneal/flash-attention-3"
            return get_kernel(hf_kernel).flash_attn_interface

        else:
            hf_kernel = "kernels-community/flash-attn3"
            if has_kernel(hf_kernel):
                return get_kernel(hf_kernel).flash_attn_interface

            else:
                return None

    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3= _fa3 is not None

_override_impl = None


def _resolve_use_fa3():

    if _override_impl=="fa3":
        assert HAS_FA3 , "Cannot override to FA3: not available on this hardware."
        return True 

    if _override_impl == "sdpa":
        return False
    
    if HAS_FA3:
        from nanollm.commons import COMPUTE_DTYPE

        if COMPUTE_DTYPE == torch.bfloat16:
            return True

        return False

    return False


def _sdpa_attention(q,k,v,window_size, enable_gqa): 
    Tq=q.size(2)
    Tk=k.size(2)

    window=window_size[0]

    if (window<0 or window>=Tq) and Tq==Tk:
        return F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=enable_gqa)

    if Tq==1:
        if window>=0 and window< Tk:
            start= max(0,Tk-(window+1))
            k=[:,:,start:,:]
            v=[:,:,start:,:]

        return F.scaled_dot_product_attention(q,k,v,is_causal=False,enable_gqa=enable_gqa)



    device=q.device

    row_idx=(Tk-Tq) +torch.arange(Tq,device=device).unsqueeze(1)
    col_idx=torch.arange(Tk,device=device).unsqueeze(0)

    mask= col_idx<=row_idx

    if window >=0 and window <Tk:
        mask = mask & ((row_idx-col_idx)<=window)

    return F.scaled_dot_product_attention(q,k,v, attn_mask=mask, enable_gqa=enable_gqa)



