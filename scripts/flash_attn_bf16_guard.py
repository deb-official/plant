# =============================================================================
# flash_attn_bf16_guard.py
#
# Downcasts fp32 q/kv tensors to bf16 immediately before they reach
# flash-attn's kernel entry point (flash_attn_varlen_kvpacked_func).
#
# Phi-3-small's custom attention (loaded via trust_remote_code) has handed
# this kernel fp32 q/kv from three independent sources found one at a time
# across Combo 3 debugging, despite the base model itself always being
# loaded in bf16: its RotaryEmbedding cache defaulting to
# torch.get_default_dtype() at construction, adapters loaded via
# PeftModel.from_pretrained (PEFT keeps these fp32 regardless of the base
# model's dtype), and the LTRModel/accelerator.prepare wrapping path in
# Stage 4. Patching each source individually proved fragile -- this was
# the third one found. Guaranteeing bf16 at the kernel boundary instead
# catches all of them, present and future, and is numerically safe since
# it only meets the kernel's own dtype requirement (accumulation up to
# this point stays at whatever precision it already was). See
# README_EURLEX_WIKI10.md's "root cause of the flash-attn fp32 assert"
# porting notes for the full trail.
#
# Must be imported -- and install_bf16_boundary_guard() called -- before
# any AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
# call for a Phi-3-small model in the same process. The dynamically
# downloaded modeling_phi3_small.py binds flash_attn_varlen_kvpacked_func
# via `from flash_attn... import ...` the first time it's imported, which
# only happens at that from_pretrained() call -- so the patch just needs
# to already be in place on flash_attn.flash_attn_interface beforehand,
# regardless of whether the caller later does a `from ... import` or an
# attribute-access call.
# =============================================================================
import logging

import torch

_PATCHED_MARKER = "_xcube_bf16_boundary_guard"


def _downcast(t):
    if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
        return t.to(torch.bfloat16)
    return t


def install_bf16_boundary_guard():
    """Idempotent. No-ops if flash_attn isn't installed (Mistral/Llama runs
    don't need it -- see settings.ini for why flash-attn isn't a default
    dependency)."""
    try:
        import flash_attn.flash_attn_interface as _fai
    except ImportError:
        logging.info(
            "flash_attn not installed -- skipping bf16 boundary guard "
            "(expected unless running a Phi-3-small combo)"
        )
        return

    if getattr(_fai, _PATCHED_MARKER, False):
        return

    if not hasattr(_fai, "flash_attn_varlen_kvpacked_func"):
        logging.warning(
            "flash_attn_interface.flash_attn_varlen_kvpacked_func not found "
            "-- bf16 boundary guard NOT installed; Phi-3-small will hit the "
            "fp32 flash-attn assert if any upstream fp32 source is present"
        )
        return

    _orig = _fai.flash_attn_varlen_kvpacked_func

    def _guarded(*args, **kwargs):
        if "q" in kwargs:
            kwargs["q"] = _downcast(kwargs["q"])
        elif args:
            args = (_downcast(args[0]),) + args[1:]
        if "kv" in kwargs:
            kwargs["kv"] = _downcast(kwargs["kv"])
        elif len(args) > 1:
            args = (args[0], _downcast(args[1])) + args[2:]
        return _orig(*args, **kwargs)

    _fai.flash_attn_varlen_kvpacked_func = _guarded
    setattr(_fai, _PATCHED_MARKER, True)
    logging.info(
        "Installed fp32->bf16 boundary guard on "
        "flash_attn_interface.flash_attn_varlen_kvpacked_func"
    )
