# Changelog

## 2.0

A second model, now the default, and two fixes for current ComfyUI that 1.x CLI users should not
skip.

### New: refine mode, the default

Lightricks' **Refine-Details** IC-LoRA on their tiled-fusion upscale graph. It rebuilds the fine
detail a soft clip is missing and leaves the rest where it was: framing, colour, motion and the
face. It works in 1024x576 tiles fused at every denoising step, so VRAM follows the tile rather
than the frame, and 4K fits a 32GB card.

Measured on the seven MiniMax H3 demo clips (640x384 upscaled 2x), against 1.1's pixel renders
of the same clips. Detail and flicker are ratios against a lanczos upscale:

| | **refine** | **pixel** |
|---|---|---|
| detail added | 1.4-2.4x | 2.3-3.8x |
| fidelity to the source | **31.8-38.9 dB, SSIM 0.92-0.98** | 24.2-31.3 dB, SSIM 0.73-0.91 |
| frame-to-frame flicker | **1.4-2.7x** | 2.2-5.1x |
| render time (97 frames, 768x1376 to 1216x2112, 32GB card) | 526 s | **206 s** |
| peak VRAM (same run) | 28.2 GB | 31.9 GB |
| 4K (97 frames to 2176x3904, RTX 5090) | **one pass: 14 min, 28.4 GB VRAM** | VRAM grows with frames x pixels |

Refine stayed closer to the source and flickered less on all seven.

- `workflows/ReDetail_LTX25_refine.json`, built by `build_refine_workflow.py` from Lightricks'
  `LTX-2.5_V2V_TiledFusion_Upscale.json`. The stock graph derives its tile from the source, and
  any source within 200px of a preset keeps its own size: a 768x1376 clip became ONE tile 1.8x
  the area the LoRA trained on. The tile is pinned to the trained 1024x576. Its two prompt-
  enhancer branches are disconnected, for the same reason 1.1 disconnected the pixel graph's, and
  its Missing Models panel points at the int8 files instead of 68GB of bf16.
- `redetail.py --model refine` (the default): exact scales on refine's /32 grid (1920x1088 at
  1.5x is now exactly 2880x1632), the card's 8-frame tail pad, the tile turned for portrait
  clips, and `--guide-strength`, the fidelity dial pixel mode never had.
- Cached conditioning for refine's prompts: `redetail_refine_pos/neg.pt`. `--cached-cond` now
  works in both modes.
- **Pixel mode is unchanged** and one flag away: `--model pixel`, `ReDetail_LTX25_upscale.json`.

### Fixed: the CLI handed back your input instead of the render

On current ComfyUI (seen on 0.37.2), `LoadVideo` reports the clip it loaded as an output, ahead of
the render, and 1.x downloaded the first video it found in the job's history. Every run
"succeeded" and returned the clip it had uploaded, re-encoded at the source size. Frame counts
matched, so nothing caught it. The CLI now asks for the save node's file by id, and refuses any
chunk that comes back at the wrong size.

### Fixed: `--bootstrap-cond` on current ComfyUI

ComfyUI now ships a core node called `SaveConditioning`, and it shadows this project's node of the
same name. ReDetail's cache nodes are now also registered as `ReDetailSaveConditioning` and
`ReDetailLoadConditioning`, which the CLI uses. **Copy the updated `tools/comfyui_cond_cache/`
folder over your old one** and restart. The old names still load the 1.x Mac workflow.

### Changed: the kornia pin is gone

1.x pinned `kornia==0.7.4` because ComfyUI-LTXVideo imported a function kornia 0.8 removed, and
that pin broke other node packs. The pack stopped importing it on 22 September 2026. Verified: the
current pack loads cleanly on kornia 0.8.3. Keep the pin only on an older pack.

### Changed: SaveVideo quality

The CLI asks SaveVideo for crf 12 instead of its default 23, which smooths exactly the texture
these models add. The setting's location moved between ComfyUI releases, so it is read from the
server's own schema, and nothing is sent to a server that has none.

### Measured: the shipped pixel conditioning against the int8 encoder

1.1 said the Q5_K_M tensors were "expected to be numerically close" to the int8 encoder's. They
are close, not identical: cosine similarity 0.934 on the tensor, and 48.3 dB PSNR between a
17-frame render using them and the same render through the int8 encoder. A cache is bit-identical
to the encoder that made it (PSNR inf, frame hashes equal, both modes). For exact parity with
your encoder, regenerate with `--bootstrap-cond`.

### Also

- `--setup` checks the mode you will run (`--model`), including the fusion nodes and the right
  LoRA.
- `tools/check_release.py` re-derives both size tables from the code, and asserts every refine
  fix in the shipped files.

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
