# ReDetail

**Generative video re-detailing and upscaling for ComfyUI, on LTX-2.5.** A drag-and-drop workflow
plus a CLI, now with two models:

- **refine** (the default since 2.0): Lightricks'
  [Refine-Details IC-LoRA](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details).
  It rebuilds the fine detail a soft clip is missing and leaves everything else where it was,
  faces included. It works in 1024x576 tiles fused at every step, so VRAM follows the tile, not
  the frame, and **4K fits a 32GB card** (system RAM is the limit there; see Memory).
- **pixel** (the 1.x model): Lightricks'
  [Pixel Spatial Upscaler](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler).
  It re-renders the whole clip and invents the fine detail as it goes. A repaint, not a polish:
  more invented detail, and faces drift.

## Which one

Measured on the seven MiniMax H3 demo clips (640x384, 10s, upscaled 2x; the pixel renders are the
1.1 release's own). Detail and flicker are ratios against a plain lanczos upscale of the same clip;
fidelity compares the result, scaled back down, with the source. Ranges across the seven:

| | **refine** | **pixel** |
|---|---|---|
| detail added | 1.4-2.4x | 2.3-3.8x |
| fidelity to the source | **31.8-38.9 dB, SSIM 0.92-0.98** | 24.2-31.3 dB, SSIM 0.73-0.91 |
| frame-to-frame flicker | **1.4-2.7x** | 2.2-5.1x |
| render time (97 frames, 768x1376 to 1216x2112, 32GB card) | 526 s | **206 s** |
| peak VRAM (same run) | 28.2 GB | 31.9 GB |
| 4K (97 frames to 2176x3904, RTX 5090) | **one pass: 14 min, 28.4 GB VRAM** | VRAM grows with frames x pixels |

Refine stayed closer to the source, and flickered less, on all seven clips. The time and VRAM rows
are single runs: one 97-frame H3 clip on a 32GB RTX PRO 4500, and the 4K pass on an RTX 5090,
which also needed 52 GB of system RAM (see [Memory](#picking-a-size)).

**Refine** wherever a face, a product or the framing has to survive, and for anything going to 4K.
**Pixel** for the most invented detail on soft AI footage with nothing to preserve, at less than
half refine's render time.

## Quick start

**Drop `workflows/ReDetail_LTX25_refine.json` into ComfyUI.** The graph is the whole render. Its
note panels walk you through install, prep and queue; read the one marked **START HERE** first.
(Pixel mode is `workflows/ReDetail_LTX25_upscale.json`.)

Your clip needs two things before it loads, because the graph can't do them for you:

```bash
# 1. an audio track (the graph carries audio through; most AI clips are silent)
ffmpeg -i in.mp4 -f lavfi -i anullsrc=r=48000:cl=stereo -shortest \
       -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac ok.mp4

# 2. a length of 8n+1 frames, or the model silently drops the tail
ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 ok.mp4
ffmpeg -i ok.mp4 -frames:v 97 -c:a copy trimmed.mp4      # 97 = 8*12+1, ~4s at 24fps
```

Then load the clip and pick `output_size` on the **Preprocess** group: FullHD (the default) or 4K.

**Or skip all of that.** `redetail.py` drives the same graphs over ComfyUI's HTTP API and does the
prep, sizing, splitting and audio for you:

```bash
python3 redetail.py --setup                              # checks your install, names what is missing
python3 redetail.py clip.mp4 --scale 1.5                 # refine
python3 redetail.py clip.mp4 --scale 1.5 --model pixel   # the 1.x behaviour
```

Use the CLI for anything longer than one short pass. It solves the target resolution, splits long
clips on their own cuts, keeps every chunk on the frame grid, pads refine's soft last frames
away, and re-muxes your original audio.

## Install

Python 3, `ffmpeg` and `ffprobe` on PATH, and a running ComfyUI with
[`Lightricks/ComfyUI-LTXVideo`](https://github.com/Lightricks/ComfyUI-LTXVideo). **Refine mode
needs the pack from 24 September 2026 or later**, when `LTXVTiledFusionSampler` landed. Install the
pack's `requirements.txt` into the Python ComfyUI actually runs from, then **restart ComfyUI**.
`No module named 'colour'` in the boot log means that step was skipped.

**One pin, into that same Python:**

```bash
/path/to/ComfyUI/venv/bin/python -m pip install "comfy-kitchen>=0.2.26"
```

```powershell
.\python_embeded\python.exe -m pip install "comfy-kitchen>=0.2.26"
```

A venv install is invisible to a system-Python ComfyUI, and the symptom is baffling: the node pack
imports fine by hand while ComfyUI still reports its nodes missing.

**The kornia pin is gone.** 1.x told you to install `kornia==0.7.4`, because the pack imported a
function that kornia 0.8 removed, and that pin broke other node packs. The pack stopped importing
it on 22 September 2026, so any pack new enough for refine mode runs on current kornia. If you
stay on an older pack for pixel mode only, keep the pin.

**The models.** All three Hugging Face repos are gated; accept the licence on **each**, including
the base repo, or the weights 403 while the README loads fine. In the refine workflow, ComfyUI's
**Missing Models** panel links straight to the right files.

| file | folder | size | mode |
|---|---|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `models/diffusion_models/` | 21.5 GB | both |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `models/text_encoders/` | 15.4 GB | both (optional, see below) |
| `ltx-2.5-video-vae-bf16.safetensors` | `models/vae/` | 1.5 GB | both |
| `ltx-2.5-audio-vae-bf16.safetensors` | `models/vae/` | 0.4 GB | both |
| `ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors` | `models/loras/` | 1.3 GB | refine |
| `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` | `models/loras/` | 0.3 GB | pixel |

**`int8_convrot` is not Blackwell-only.** 1.0 said it was; users run it on a 3090, a 4090 and a
1070. If it fails to load, check `comfy-kitchen>=0.2.26` before blaming your card: 0.2.10, which
several ComfyUI images ship, fails on every convrot checkpoint with an error that reads like
unsupported weights.

The GGUF path is still the answer if int8 genuinely does not work for you, and it is lighter
regardless:

```bash
python3 redetail.py clip.mp4 --scale 1.5 --model pixel \
  --gguf LTX-2.5-Distilled-Q4_K_M.gguf \
  --encoder gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --clip-device cpu --budget 150 --decode-tile 256 --decode-temporal 32
```

That needs [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) plus the Q4_K_M
transformer from [`Abiray/LTX-2.5-Distilled-GGUF`](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF)
in `models/unet/`. Measured on an RTX 4090 in pixel mode: sampling took 70s for 8 steps at 21.8 GB
of 24.5 peak. Refine mode takes the same flags: on the RTX 5090, the Q4_K_M transformer with
`--cached-cond` rendered 17 frames 640x384 to 1280x768 at 48.0 dB PSNR (SSIM 0.991) against the
int8 render. Refine on an actual 24GB card has not been measured yet.

## Picking a size

**Refine** needs both output dimensions divisible by **32**, and it resizes your clip to the canvas
itself, so the source can be any size. `--scale` lands on the nearest valid size that holds your
aspect:

| your source | 1.5x | 2x |
|---|---|---|
| 640x384 | 960x576 | 1280x768 |
| 768x1376 | **1120x2016** (1.47x) | 1536x2752 |
| 1280x720 | **1920x1088** (1.51x) | 2560x1440 |
| 1920x1080 | **2848x1600** (1.48x) | **3808x2144** (1.99x) |
| 1920x1088 | 2880x1632 | 3840x2176 |

In the workflow you pick `output_size` instead: FullHD is a 1920 long edge, 4K a 3840 one (a
1920x1080 source becomes 3840x2176). Lightricks calls 4K the sweet spot for a 1080p source.
**Don't go to 8K in one pass**: a 1024x576 tile then covers 13% of the frame, twice the scale the
LoRA trained at, and it starts inventing texture. Refine to 4K and upscale that with lanczos.

**Pixel** needs **/64**, not /32, because its guide is half resolution and split into 2x2 latent
patches. Get it wrong and you get `Error while processing rearrange-reduction pattern`.

| your source | 1.5x | 2x |
|---|---|---|
| 640x384 | 960x576 | 1280x768 |
| 768x1408 | 1152x2112 | 1536x2816 |
| 1024x576 | **1472x832** (1.44x) | 2048x1152 |
| 1088x1920 | **1600x2816** (1.47x) | 2176x3840 |
| 1920x1088 | **2816x1600** (1.47x) | 3840x2176 |

The bold pixel rows have **no exact 1.5x on the /64 grid at all**. 1920x1088 is the clearest
case: its exact 1.5x is 2880x1632, and 1632 isn't /64 (refine, on /32, takes it as is). 2x is
exact far more often, because doubling a /64 number stays /64. The CLI also puts a pixel-mode
*source* on the grid, so it may crop a few percent or resample; watch the `source ->` line it
prints if your framing is tight.

**Memory, refine.** VRAM follows the tile, not the canvas, and long clips run as 97-frame windows
inside the sampler, so one render covers a whole shot. Measured on an RTX 5090: 97 frames to
2176x3904 in one pass took 14 min at **28.4 GB of VRAM**.

What grows with length and size is **system RAM**. ComfyUI keeps the model weights staged in RAM
(about 22 GB here), and it holds both the clip lanczos-resized to the output size and the decoded
result as full float frames: about **33 MB per output frame-megapixel**. That 97-frame 4K pass
peaked at **52 GB of RAM** with `--cached-cond`; with the text encoder also loaded, 105 frames at
that size ran a 57 GB machine out of memory. `--budget` caps the frame-megapixels per chunk
(refine default 500, about 60 frames at 4K). Lower it on a smaller machine, and use
`--cached-cond` for anything at 4K.

**Memory, pixel.** Keep `frames × (width × height ÷ 1,000,000)` under your card's budget. The
96 GB row is measured; the others are starting points:

| VRAM | keep under | example that fits |
|---|---|---|
| 96 GB | ~850 | 129 frames at 2816x1600 |
| 48 GB | ~350 | 129 frames at 1472x832 |
| 24 GB | ~150 | 129 frames at 960x576 |

Sampling and decode run out of memory *independently*. Our 4090 sampled all 8 steps cleanly and
then OOMed in VAE decode. If you crash after the progress bar completes, lower `--decode-tile`
rather than your resolution.

**What pixel's scales cost.** Single runs on one clip (243 frames from 768x1408) on one RTX PRO
6000, indicative rather than spec: 1:1 took 3 min at 52 GB, 1.5x took 7 min at 65 GB, 2x took
17 min at 80.5 GB.

## When something breaks

| you see | it means | do this |
|---|---|---|
| `VAEEncodeAudio: input audio is None` | silent clip | add a silence track |
| `No module named 'colour'`, LTX nodes missing | pack requirements not installed | install its requirements.txt, restart |
| `LTXVTiledFusionSampler` missing | node pack too old for refine | update ComfyUI-LTXVideo (24 Sep 2026+) |
| `rearrange-reduction pattern` | a pixel-mode dimension isn't /64 | use the pixel sizing table |
| `'NoneType' has no attribute 'Params'` | comfy-kitchen too old | `>=0.2.26`, in ComfyUI's own Python |
| `cannot import name 'pad' from kornia…` | pack older than 22 Sep 2026 on kornia 0.8 | update the pack, or pin `kornia==0.7.4` |
| `Is a directory` on Load Image (pixel) | no first frame loaded | load your clip's first frame |
| node missing / `no class_type` | pack failed to import | restart ComfyUI, then read its **boot log** |
| OOM *after* the progress bar finishes | decode spike, not sampling | lower `--decode-tile` |
| pixel output came back 1:1 | canvas link not severed | use the shipped workflow, not the stock example |

`python3 redetail.py --setup` catches most of these before you hit them, including on a remote
ComfyUI. It needs a directly reachable, unauthenticated endpoint, and it sends no credentials.
Pass the same `--model` / `--gguf` / `--encoder` flags you intend to render with, or it checks
the wrong file set.

## Scope

**Refine** behaves like restoration without being it. It was trained to rebuild a frame's missing
fine band from a degraded copy of that frame, so it keeps framing, exposure, colour and the face
that is there: on our test clip it sat 0.1 px from the source geometry. Its limits, from
Lightricks' card and our own tests:

- **It cannot repair a malformed face.** It rebuilds the detail of whatever face is there. The
  card offers a reference image at the -1 token for faces, text and logos; on a face that was
  wrong in the source it changed nothing visible for us, so the CLI does not ship it. Lowering
  `--guide-strength` buys texture, not identity: in one test 0.6 drifted from the face and 0.4
  drew a different person.
- Small text and logos can come out as glyph-shaped noise.
- Archive and broadcast footage comes out cleaner but emptier, because it reads codec damage as
  noise. Lightricks pairs it with their Restoration IC-LoRA for that.
- Pristine camera raw comes out slightly softer: it removes grain it cannot rebuild.
- An upscale cannot exceed its source. Out-of-focus areas gain close to nothing.

**Pixel** is a generative re-renderer, as
[its model card](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler)
says too. **Nothing in it preserves identity.** In informal tests on one person it resolved lashes
where the source was mush, and it also **added freckles that were not in the original**. On a
motocross clip it re-drew the jersey graphic and number plate: stable across frames, but not the
source's markings. **Any logo, number or text gets re-imagined.**

Either way, compare a **face** against your source, not overall sharpness.

**Known limitations.** No resume: a failure on chunk 9 of 10 re-renders all nine. No polling
deadline, so if ComfyUI dies mid-run the CLI waits rather than failing; Ctrl-C and re-run. Uploads
buffer in memory. Mid-shot seams are possible when memory forces a split with no cut nearby. The
drag-and-drop workflows prep nothing; that's what the CLI is for. And it overwrites its output
without asking, using `<output>_chunks/` as scratch (it deletes only files it created there).

## Every flag

```
input                 video to upscale (omit with --setup)
--setup               check the install and exit
--model     refine    'refine' (Refine-Details, default) or 'pixel' (the 1.x upscaler)
--scale     1.5       1.0 re-detail in place · 1.5 · 2.0 (1080p -> 4K-class)
--guide-strength 1.0  refine only: how hard the guide holds to your clip. Lower trades fidelity for texture
--comfy               ComfyUI URL (default http://127.0.0.1:8188)
--out                 output file (default <input>_redetail.mp4)
--budget              frame-megapixels per chunk (850 pixel, 500 refine). Lower it if you run out of memory
--audio     original  'original' re-muxes your track; 'generated' keeps the model's
--keep-chunks         keep the per-chunk intermediates
--gguf                GGUF transformer in models/unet. Lighter; try int8 first
--encoder             text encoder filename. Prefer --cached-cond and load none
--clip-device         'cpu' or 'default'; omitted leaves the workflow value alone
--decode-tile         VAEDecodeTiled tile_size, lower this for decode OOMs
--decode-temporal     VAEDecodeTiled temporal_size (default 128)
--bootstrap-cond      load the encoder ONCE, cache both modes' prompt conditioning, exit
--cached-cond         load that cache instead of the encoder
```

## Skip the text encoder entirely

Both graphs' prompts are fixed (pixel's boxes are empty; refine's carry Lightricks' look-and-style
prompts), which makes their text conditioning a constant. So the only reason the text encoder is
ever loaded is to compute the same few-KB tensors every run.

**Those tensors ship in this repo, so you never download the encoder at all.** Copy
`tools/comfyui_cond_cache/` into `ComfyUI/custom_nodes/`, restart, and run:

```bash
python3 redetail.py clip.mp4 --cached-cond --scale 1.5
```

That drops **15.4GB** (int8) or **26GB** (bf16) from what you have to fetch. Measured in pixel
mode on an RTX 5090, 17 frames at 640x384 to 1280x768:

| | time | peak VRAM |
|---|---|---|
| with the encoder | 29.2s | 30.4 GB of 32 |
| `--cached-cond` | **24.0s** | **24.8 GB** |

**A cache is bit-identical to the encoder that made it**: PSNR inf and equal frame hashes, in both
modes. The shipped refine pair came from the `int8_convrot` encoder. The shipped pixel pair came
from Q5_K_M, and against an int8 encoder render it sits at 48.3 dB: close, not identical. To make
your own instead, which then take precedence:

```bash
python3 redetail.py --bootstrap-cond --encoder <your encoder>      # loads it once, then never again
```

They are model output, so they carry the LTX-2 Community Licence, not this project's MIT grant. See
[LICENSE](LICENSE).

## Apple Silicon

**Pixel mode only, via `workflows/ReDetail_LTX25_upscale_MAC.json`**: the pixel graph with the
GGUF transformer wired in and the text encoder removed entirely, so nothing in it will try to load
a model you do not have. Its own note panel carries the Mac-specific setup. Refine mode has not
been tested on Apple Silicon.

Measured on an M5: 33 frames 640x384 to 1280x768 in **4.4 min**, three models loaded (transformer
and the two VAEs), no encoder. Per frame-megapixel that is about **6x slower than an RTX 5090**. A
10s clip is roughly 34 min at 2x, 19 min at 1.5x. PSNR 37.4 dB against the same source rendered on
the int8 path, so quality holds.

The cached conditioning above is what makes it fit at all: the only non-int8 encoder is the 26GB
bf16 one. A Q5_K_M encoder, if you use one instead, needs the gemma4 patch for `ComfyUI-GGUF`,
which ships beside the encoder itself.

**One ComfyUI core edit is required.** `comfy/ldm/lightricks/vae/na_diffusion_decoder.py` builds
RoPE inverse frequencies in `float64`, which MPS does not support, so the run samples all the way
through and then dies in `VAEDecodeTiled` with `Cannot convert a MPS Tensor to float64`. The
function already returns float32, so compute it on the CPU and move the fp32 result across. Nothing
else changes and CUDA is unaffected.

2.0 was tested against ComfyUI 0.37.2 with ComfyUI-LTXVideo from 1 October 2026, kornia 0.8.3 and
comfy-kitchen 0.2.37, on an RTX 5090, October 2026. The Apple Silicon and 4090 numbers are from
1.1, on ComfyUI 0.32.0, August 2026.

## Licence

**The MIT grant in `LICENSE` covers this project's own code only**: `redetail.py`, the
`build_*_workflow.py` scripts, `tools/` and `demos/`. It covers neither the weights nor the
workflows.

**The workflow files are modified derivatives** of Lightricks'
`example_workflows/2.5/LTX-2.5_V2V_TiledFusion_Upscale.json` (refine) and
`example_workflows/2.5/LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json` (pixel), distributed
**exclusively under the LTX-2 Community License Agreement**. A complete copy is included at
[`workflows/LICENSE-LTX-2-Community.txt`](workflows/LICENSE-LTX-2-Community.txt), as that licence
requires. `LICENSE` lists exactly what was changed. Lightricks' unmodified examples are not
redistributed here; the build scripts tell you where to get them.

That licence carries **use restrictions** which apply to you and to anyone you pass these files to,
and any entity over **$10,000,000** annual revenue needs a separate paid commercial licence from
[Lightricks](https://ltx.io/model/licensing) first. The weights are under the same Agreement; the
IC-LoRAs under their own repositories' terms. All three Hugging Face repos are gated.
