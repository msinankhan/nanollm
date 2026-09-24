import torch
import torch.nn.functional as F

import warnings

# ---------------------------------------------------------------------------
# Capability probe: decide the backend from the GPU, BEFORE loading anything.
#
# Why not try/except? Because on some GPUs a wrong backend LOADS fine and only
# fails at kernel launch. FA3's sm_80 cubins run on sm_89 by binary
# compatibility, so a try/except would happily pick FA3 on an Ada card and then
# die mid-training on Blackwell. Capability check first; load second.
# ---------------------------------------------------------------------------

COMPUTE_CAP = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
CAP_STR = f"sm_{COMPUTE_CAP[0]}{COMPUTE_CAP[1]}" if COMPUTE_CAP else "no CUDA"


def _smoke_test(fn, label):
    """Importing a kernel proves nothing; executing it proves everything.

    A tiny windowed call exercises both the causal path and the sliding-window
    path, so a backend that lacks local-attention support fails here rather
    than 4000 steps into a training run.
    """
    q = torch.randn(1, 64, 2, 64, dtype=torch.bfloat16, device="cuda")
    fn(q, q, q, causal=True, window_size=(32, 0))
    torch.cuda.synchronize()


def _try_fa2():
    """flash-attn 2.x. setup.py builds for 80;90;100;110;120 and compiles in
    local/sliding-window attention by default (DISABLE_LOCAL is commented out).
    The only option that works on sm_120."""
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    _smoke_test(flash_attn_func, "fa2")
    import flash_attn as _m
    return (flash_attn_func, flash_attn_with_kvcache,
            f"flash-attn {getattr(_m, '__version__', '?')} natively compiled for {CAP_STR}")


def _try_fa3_hub():
    """FlashAttention 3 from the HF kernels hub.

    `version=` is REQUIRED as of kernels 0.17.x -- the old
    get_kernel('repo') one-arg form now raises ValueError.

    metadata declares archs ["8.0", "9.0a"] only:
      - sm_90 loads the sm90a build directly
      - sm_80/86/89 load the sm_80 build via binary compatibility
      - sm_120 is REJECTED by the arch validator -> has_kernel() is False
    """
    major = COMPUTE_CAP[0] if COMPUTE_CAP else -1
    if major not in (8, 9):
        return None, f"{CAP_STR} not in FA3's declared archs (8.0, 9.0a)"
    from kernels import get_kernel, has_kernel
    repo = "varunneal/flash-attention-3" if major == 9 else "kernels-community/flash-attn3"
    if not has_kernel(repo):
        return None, f"has_kernel({repo}) is False"
    itf = get_kernel(repo, version=2).flash_attn_interface
    _smoke_test(itf.flash_attn_func, "fa3_hub")
    return (itf.flash_attn_func, itf.flash_attn_with_kvcache), f"{repo} v2 on {CAP_STR}"


# ---------------------------------------------------------------------------
# Order of preference, from capability. Not from trial and error.
#
#   sm_120 (Blackwell)  -> fa2 only; FA3 cannot run there at all
#   sm_90  (Hopper)     -> native FA3 kernels, so prefer them
#   sm_80/86/89         -> fa2 is natively built; fa3_hub only via compat
#   anything else       -> sdpa (the EAGER fallback inside flash_attn_func)
# ---------------------------------------------------------------------------
if COMPUTE_CAP is None:
    PREFERENCE = []
    PREFERENCE_NOTE = "no CUDA device, using SDPA"
elif COMPUTE_CAP[0] == 12:
    PREFERENCE = [_try_fa2]
    PREFERENCE_NOTE = "Blackwell sm_120: only flash-attn 2 supports this architecture"
elif COMPUTE_CAP[0] == 9:
    PREFERENCE = [_try_fa3_hub, _try_fa2]
    PREFERENCE_NOTE = "Hopper sm_90: native FA3 kernels preferred"
elif COMPUTE_CAP[0] in (8, 10, 11):
    PREFERENCE = [_try_fa2, _try_fa3_hub]
    PREFERENCE_NOTE = f"{CAP_STR}: natively built FA2 preferred over FA3 compat path"
else:
    PREFERENCE = []
    PREFERENCE_NOTE = f"{CAP_STR} unsupported by any fused kernel, using SDPA"


_FAILURES = []
BACKEND = None
BACKEND_REASON = ""

for _candidate in PREFERENCE:
    _name = _candidate.__name__.replace("_try_", "")
    try:
        _res = _candidate()
        # a loader may decline gracefully (returning None) or succeed
        if isinstance(_res, tuple) and len(_res) == 2 and _res[0] is None:
            _FAILURES.append(f"{_name}: {_res[1]}")
            continue
        _func, _kvcache, _why = _res
        BACKEND, BACKEND_REASON = _name, _why
        break
    except Exception as _e:
        _FAILURES.append(f"{_name}: {type(_e).__name__}: {str(_e)[:120]}")

if BACKEND is None:
    BACKEND_REASON = (PREFERENCE_NOTE +
                      ("; tried -> " + " | ".join(_FAILURES) if _FAILURES else ""))


def _backend_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Dispatch to the selected fused kernel, or fall through to SDPA."""
    if BACKEND == "fa2":
        return _func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: (B, T, H, D) -> (B, H, T, D)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    return _sdpa_attention(q, k, v, window_size, enable_gqa).transpose(1, 2)


def _backend_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                     causal=False, window_size=(-1, -1)):
    """Dispatch to the selected fused kernel, or fall through to SDPA."""
    if BACKEND == "fa2":
        return _func(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
                     causal=causal, window_size=window_size)

    B, T_new, H, D = q.shape
    pos = int(cache_seqlens[0].item())
    if k is not None and v is not None:
        k_cache[:, pos:pos + T_new, :, :] = k
        v_cache[:, pos:pos + T_new, :, :] = v

    end = pos + T_new
    enable_gqa = H != k_cache.size(2)
    y = _sdpa_attention(q.transpose(1, 2), k_cache[:, :end].transpose(1, 2),
                        v_cache[:, :end].transpose(1, 2), window_size, enable_gqa)
    return y.transpose(1, 2)


# Keep gpt.py working unchanged: it calls
#   flash_attn.flash_attn_func(...)  and  flash_attn.flash_attn_with_kvcache(...)
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=_backend_func,
    flash_attn_with_kvcache=_backend_kvcache,
)

# Aliases so nothing else in the repo breaks.
HAS_FA3 = BACKEND in ("fa2", "fa3_hub")
USE_FA3 = BACKEND in ("fa2", "fa3_hub")


def backend_report():
    """Print this at startup. Never assume which kernel you got."""
    return f"attention backend: {BACKEND or 'sdpa'} ({BACKEND_REASON})"


def _sdpa_attention(q,k,v,window_size, enable_gqa): 
    Tq=q.size(2)
    Tk=k.size(2)

    window=window_size[0]

    if (window<0 or window>=Tq) and Tq==Tk:
        return F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=enable_gqa)

    if Tq==1:
        if window>=0 and window< Tk:
            start= max(0,Tk-(window+1))
            k=k[:,:,start:,:]
            v=v[:,:,start:,:]

        return F.scaled_dot_product_attention(q,k,v,is_causal=False,enable_gqa=enable_gqa)



    device=q.device

    row_idx=(Tk-Tq) +torch.arange(Tq,device=device).unsqueeze(1)
    col_idx=torch.arange(Tk,device=device).unsqueeze(0)

    mask= col_idx<=row_idx

    if window >=0 and window <Tk:
        mask = mask & ((row_idx-col_idx)<=window)

    return F.scaled_dot_product_attention(q,k,v, attn_mask=mask, enable_gqa=enable_gqa)




def flash_attn_func(q,k,v,causal=False,window_size=(-1,-1)):
    if _Use_FA3:
        return _fa3.flash_attn_func(q,k,v,causal=causal,window_size=window_size)

    q=q.transpose(1,2)
    k=k.transpose(1,2)
    v=v.transpose(1,2)
    enable_gqa=q.size(1) != k.size(1)

    y= _sdpa_attention(q,k,v,window_size, enable_gqa)

    return y.transpose(1,2)


def flash_attn_with_kvcache(q,k_cache,v_cache,k=None,v=None,cache_seqlens=None,causal=False,window_size=(-1,-1)):
    if _Use_FA3:
        return _fa3.flash_attn_with_kvcache(q,k_cache,v_cache,k=k,v=v,cache_seqlens=cache_seqlens,causal=causal,window_size=window_size)


    B,T_new,H,D=q.shape
    pos=cache_seqlens[0].item()

    if k is not None and v is not None:
        k_cache[:,pos:pos+T_new,:,:] = k
        v_cache[:,pos:pos+T_new,:,:] = v

    end=pos+T_new

    k_full= k_cache[:,:end,:,:]
    v_full= v_cache[:,:end,:,:]

    q_sdpa=q.transpose(1,2)
    k_sdpa=k_full.transpose(1,2)
    v_sdpa=v_full.transpose(1,2)

    enable_gqa= q_sdpa.size(1)!=k_sdpa.size(1)
    y_sdpa= _sdpa_attention(q_sdpa,k_sdpa,v_sdpa,window_size,enable_gqa)

    return y_sdpa.transpose(1,2)


from types import SimpleNamespace

flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)