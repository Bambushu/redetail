# Changelog

## 1.1

Two fixes you should not skip, and two additions.

### Fixed: the workflow would not queue on most non-default installs

`GemmaAPITextEncode`, the prompt enhancer, reads its `ckpt_name` list from `models/diffusion_models`.
ComfyUI validates every node upstream of an output **even on a branch that is never taken**, so
anyone whose transformer lives elsewhere got

```
ckpt_name: 'ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors' not in []
```

and could not queue at all. Disabling the enhancer did not help, because the failure was at
validation, not execution. That is everyone on the GGUF path, which this project recommended until
this release.

The conditioning now comes straight from `LTXVConditioning`, past the two switches, so the enhancer
is unreachable and ComfyUI prunes it before validating. **The render is unchanged**, verified
bit-identical (PSNR inf) over 33 frames.

This removes the `ltxv_` API-key branch. That is deliberate: it needed the local checkpoint to read
its model id, so it could never have worked on the installs it was breaking, and the cached
conditioning below covers its only real purpose.

Reported by u/dirtybeagles, who described it precisely enough to reproduce on the first try.

### Corrected: int8_convrot is not Blackwell-only

The previous release said `int8_convrot` requires Blackwell and would not load on anything older,
and `--setup` **hard-failed** those cards. That was wrong. Users report it running on a 3090, a 4090
and a 1070, and NVIDIA has shipped INT8 tensor cores since Pascal.

One rented 4090 would not load it and we concluded architecture. The likelier cause is
`comfy-kitchen 0.2.10`, which several ComfyUI images ship, which fails on every convrot checkpoint,
and whose error reads like unsupported weights rather than a stale library. **If int8 will not load,
check `comfy-kitchen>=0.2.26` before blaming your card.**

Sorry. That sent people to downloads they did not need.

### Added: the text encoder is now optional

Both prompt boxes in this graph are empty, so the text conditioning is a constant. It ships
pre-computed in `tools/comfyui_cond_cache/` (26KB per file), so the encoder need never be
downloaded or loaded.

Copy that directory into `ComfyUI/custom_nodes/`, restart, and add `--cached-cond`.

Measured on an RTX 5090, 17 frames 640x384 to 1280x768:

| | time | peak VRAM |
|---|---|---|
| with the encoder | 29.2s | 30.4 GB of 32 |
| `--cached-cond` | **24.0s** | **24.8 GB** |

Output is bit-identical, PSNR inf. The stock graph peaks at 95% of a 32GB board, so 5.6GB of
headroom decides whether a card runs it. It also removes a 15.4GB (int8) or 26GB (bf16) download.

`--bootstrap-cond` regenerates the cache from your own encoder if you would rather not use ours.

### Added: Apple Silicon

`workflows/ReDetail_LTX25_upscale_MAC.json` runs the GGUF transformer with cached conditioning and
loads no text encoder at all. Verified by rendering, not inspection: 33 frames 640x384 to 1280x768
in **4.4 min** on an M5, three models loaded (transformer, video VAE, audio VAE) and no encoder.

Per frame-megapixel that is about **6x slower than an RTX 5090**, not the 30x other local ports
cost. A 10s clip is roughly 34 min at 2x, 19 min at 1.5x. Quality holds: PSNR 37.4 dB against the
same source rendered on the int8 path.

**One ComfyUI core edit is required.** `comfy/ldm/lightricks/vae/na_diffusion_decoder.py` builds
RoPE frequencies in `float64`, which MPS does not support, so the clip samples all the way through
and dies on the last node. `rope_inv_freqs` should compute on the CPU and move the fp32 result to
the device; it already returns float32, so nothing else changes.

### Known limitation

The shipped conditioning was produced by the **Q5_K_M** encoder. `int8_convrot` is a different
quantisation of the same model, and its embedding of the empty string is expected to be numerically
close but **has not been verified identical**. If that matters to you, regenerate with
`--bootstrap-cond`, which takes precedence over the shipped files.

## 1.0

Initial release. Drag-and-drop ComfyUI workflow for the LTX-2.5 IC-LoRA Pixel Spatial Upscaler,
plus `redetail.py` for real clips: sizing onto the /64 grid, splitting long clips on their own
scene cuts, the 8n+1 frame grid, and re-muxing the original audio.
