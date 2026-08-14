"""Save/load a ComfyUI CONDITIONING to disk, and ship a pre-computed one.

WHY: the ReDetail graph runs with BOTH prompt boxes empty, so its text conditioning is a constant.
Encoding it is the only reason the 15.4GB (int8) or 26GB (bf16) text encoder is ever loaded. Cache
it once and that model never loads again.

MEASURED on an RTX 5090, 17 frames 640x384 -> 1280x768:
    stock graph      29.2s   peak 30.4GB of 32
    no encoder       24.0s   peak 24.8GB
    output           bit-identical, PSNR inf against the stock run
On a 32GB card the stock graph peaks at 95% of the board, so 5.6GB of headroom is the difference
between running and OOM. Nothing in ComfyUI core serialises a CONDITIONING (SaveLatent/LoadLatent
are latents only), which is why this exists.

SHIPPED FILES. `redetail_pos.pt` and `redetail_neg.pt` sit beside this file, so you do not need the
text encoder at all -- not even once, to bootstrap. They were produced by
`gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf` (type `ltxv`) encoding the empty string.

  PROVENANCE MATTERS HERE: they are the output of Lightricks' gated LTX-2.5 encoder weights, so
  they are a DERIVATIVE of those weights and are distributed under the LTX-2 Community License
  Agreement, exactly like the workflow files. They are NOT covered by this project's MIT grant.
  See LICENSE.

  They were generated from the Q5_K_M quantisation. The int8_convrot encoder is a different
  quantisation of the same model, so its embedding of the empty string will be numerically close
  but is NOT verified identical. If you have an encoder and want to be certain, regenerate:
      python3 redetail.py --bootstrap-cond --encoder <your encoder>
"""
import os

import torch

import folder_paths

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(folder_paths.get_output_directory(), "cond_cache")


def _to(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    if isinstance(obj, dict):
        return {k: _to(v, dev) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(v, dev) for v in obj)
    return obj


def _search(name):
    """Written caches win over shipped ones, so regenerating actually takes effect."""
    return [os.path.join(CACHE, f"{name}.pt"), os.path.join(HERE, f"{name}.pt")]


class SaveConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",),
                             "name": ("STRING", {"default": "redetail"})}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/cache"

    def save(self, conditioning, name):
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, f"{name}.pt")
        # Always to CPU: a tensor saved from CUDA carries its device, and loading one onto a full
        # card is precisely the OOM this is meant to avoid.
        torch.save(_to(conditioning, "cpu"), path)
        print(f"[cond_cache] saved {path} ({os.path.getsize(path)/1e6:.3f} MB)")
        return (conditioning,)


class LoadConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"name": ("STRING", {"default": "redetail_pos"})}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load"
    CATEGORY = "conditioning/cache"

    @classmethod
    def IS_CHANGED(cls, name):
        # Key on mtime, not the name. Otherwise ComfyUI's execution cache keeps serving the old
        # conditioning after the file is rewritten, and a comparison silently measures the
        # previous run.
        for p in _search(name):
            if os.path.exists(p):
                return os.path.getmtime(p)
        return float("nan")

    def load(self, name):
        for p in _search(name):
            if os.path.exists(p):
                # weights_only=False: a CONDITIONING holds plain python objects beside its tensors.
                # Only ever point this at a file this node wrote or that shipped with it.
                return (_to(torch.load(p, map_location="cpu", weights_only=False), "cpu"),)
        raise FileNotFoundError(
            f"no cached conditioning '{name}' in any of: {', '.join(_search(name))}")


NODE_CLASS_MAPPINGS = {"SaveConditioning": SaveConditioning,
                       "LoadConditioning": LoadConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveConditioning": "Save Conditioning (cache)",
                              "LoadConditioning": "Load Conditioning (cache)"}
