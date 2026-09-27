import os
import torch
import torch.nn.functional as F


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

    This exercises BF16, grouped-query attention, causal sliding-window
    attention, and backward before a training run starts.
    """
    q = torch.randn(1, 64, 4, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(1, 64, 2, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(1, 64, 2, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    out = fn(q, k, v, causal=True, window_size=(32, 0))
    out.float().square().mean().backward()
    torch.cuda.synchronize()


# --- every loader returns the SAME shape: (payload_or_None, reason) ---------
# payload is (attention_fn, kvcache_fn) when it succeeded, None when declined.


def _try_fa2():
    """flash-attn 2.x. setup.py builds for 80;90;100;110;120 and compiles in
    local/sliding-window attention by default (DISABLE_LOCAL is commented out).
    The only option that works on sm_120."""
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    _smoke_test(flash_attn_func, "fa2")
    import flash_attn as _m
    return ((flash_attn_func, flash_attn_with_kvcache),
            f"flash-attn {getattr(_m, '__version__', '?')} natively compiled for {CAP_STR}")


def _try_fa2_hub():
    """Prebuilt FlashAttention 2 binaries, including sm_120."""
    from kernels import get_kernel
    # Pin the verified cxx11 build containing torch 2.10 / CUDA 12.8 / sm_120.
    # Package v3 uses the stable ABI, whose GQA backward is broken upstream:
    # https://github.com/huggingface/kernels-community/issues/1085
    revision = "f50dc99ed079b35990bc895d43fd353ea0cb376d"
    module = get_kernel("kernels-community/flash-attn2", revision=revision)
    interface = getattr(module, "flash_attn_interface", module)
    func = interface.flash_attn_func
    kvcache = interface.flash_attn_with_kvcache
    _smoke_test(func, "fa2_hub")
    return ((func, kvcache),
            f"kernels-community/flash-attn2 cxx11 revision {revision[:7]} on {CAP_STR}")


def _try_fa3_hub():
    """FlashAttention 3 from the HF kernels hub.

    `version=` is REQUIRED as of kernels 0.17.x -- and that applies to
    has_kernel() too, not just get_kernel(). Calling either without a
    version/revision raises ValueError before any arch check happens.

    metadata declares archs ["8.0", "9.0a"] only:
      - sm_90 loads the sm90a build directly
      - sm_80/86/89 load the sm_80 build via binary compatibility
      - sm_120 is REJECTED by the arch validator -> has_kernel() is False

    varunneal/flash-attention-3 (which nanochat prefers on Hopper) now fails
    publisher-trust verification, so it is attempted only with
    trust_remote_code=True, and only as a second choice.
    """
    from kernels import get_kernel, has_kernel
    candidates = [("kernels-community/flash-attn3", False),
                  ("varunneal/flash-attention-3", True)]
    reasons = []
    for repo, needs_trust in candidates:
        try:
            if not has_kernel(repo, version=2, trust_remote_code=needs_trust):
                reasons.append(f"{repo}: has_kernel=False")
                continue
            itf = get_kernel(repo, version=2,
                             trust_remote_code=needs_trust).flash_attn_interface
            _smoke_test(itf.flash_attn_func, "fa3_hub")
            return ((itf.flash_attn_func, itf.flash_attn_with_kvcache),
                    f"{repo} v2 on {CAP_STR}")
        except Exception as e:
            reasons.append(f"{repo}: {type(e).__name__}: {str(e)[:80]}")
    return None, " | ".join(reasons)


# ---------------------------------------------------------------------------
# Order of preference, from capability. Not from trial and error.
#
#   sm_120 (Blackwell)  -> fa2 only; FA3 cannot run there at all
#   sm_90  (Hopper)     -> native FA3 kernels, so prefer them
#   sm_80/86/89         -> fa2 natively built; fa3_hub only via compat
#   anything else       -> sdpa  (the fallback inside _backend_func)
# ---------------------------------------------------------------------------
if COMPUTE_CAP is None:
    PREFERENCE = []
    PREFERENCE_NOTE = "no CUDA device, using SDPA"
elif COMPUTE_CAP[0] == 12:
    PREFERENCE = [_try_fa2_hub, _try_fa2]
    PREFERENCE_NOTE = "Blackwell sm_120: prebuilt FA2 preferred"
elif COMPUTE_CAP[0] == 9:
    PREFERENCE = [_try_fa3_hub, _try_fa2_hub, _try_fa2]
    PREFERENCE_NOTE = "Hopper sm_90: native FA3 kernels preferred"
elif COMPUTE_CAP[0] in (8, 10, 11):
    PREFERENCE = [_try_fa2_hub, _try_fa2, _try_fa3_hub]
    PREFERENCE_NOTE = f"{CAP_STR}: natively built FA2 preferred over FA3 compat path"
else:
    PREFERENCE = []
    PREFERENCE_NOTE = f"{CAP_STR} unsupported by any fused kernel, using SDPA"

# Optional override for reproducible A/B tests.
_override_impl = os.environ.get("NANOLLM_ATTN_BACKEND", "auto").lower()
if _override_impl not in ("auto", "fa2_hub", "fa2", "fa3_hub", "sdpa"):
    raise ValueError("NANOLLM_ATTN_BACKEND must be auto, fa2_hub, fa2, fa3_hub, or sdpa")
if _override_impl == "fa2_hub":
    PREFERENCE = [_try_fa2_hub]
elif _override_impl == "fa2":
    PREFERENCE = [_try_fa2]
elif _override_impl == "fa3_hub":
    PREFERENCE = [_try_fa3_hub]
elif _override_impl == "sdpa":
    PREFERENCE = []

_FAILURES = []
BACKEND = None
BACKEND_REASON = ""

for _candidate in PREFERENCE:
    _name = _candidate.__name__.replace("_try_", "")
    try:
        _payload, _why = _candidate()
    except Exception as _e:
        _FAILURES.append(f"{_name}: {type(_e).__name__}: {str(_e)[:120]}")
        continue
    if _payload is None:          # declined gracefully (wrong arch, etc.)
        _FAILURES.append(f"{_name}: {_why}")
        continue
    _func, _kvcache = _payload
    BACKEND, BACKEND_REASON = _name, _why
    break

if _override_impl == "sdpa":
    BACKEND, BACKEND_REASON = None, "forced by NANOLLM_ATTN_BACKEND=sdpa"
elif _override_impl != "auto" and BACKEND != _override_impl:
    raise RuntimeError(f"Requested attention backend {_override_impl!r} failed on {CAP_STR}: " + " | ".join(_FAILURES))

if BACKEND is None:
    BACKEND_REASON = (PREFERENCE_NOTE +
                      ("; tried -> " + " | ".join(_FAILURES) if _FAILURES else ""))


def _backend_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Dispatch to the selected fused kernel, or fall through to SDPA."""
    if BACKEND in ("fa2_hub", "fa2", "fa3_hub"):
        return _func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: (B, T, H, D) -> (B, H, T, D)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    return _sdpa_attention(q, k, v, window_size, enable_gqa).transpose(1, 2)


def _backend_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                     causal=False, window_size=(-1, -1)):
    """Dispatch to the selected fused kernel, or fall through to SDPA."""
    if BACKEND in ("fa2_hub", "fa2", "fa3_hub"):
        return _kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
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


from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=_backend_func,
    flash_attn_with_kvcache=_backend_kvcache,
)

HAS_FA3 = BACKEND in ("fa2_hub", "fa2", "fa3_hub")
USE_FA3 = BACKEND in ("fa2_hub", "fa2", "fa3_hub")


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
