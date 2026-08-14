# ReDetail

**A generative video re-detailer for ComfyUI.** It runs your clip back through
[LTX-2.5's pixel spatial upscaler](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler),
which doesn't sharpen or recover anything. It re-renders the whole thing and invents the fine
detail as it goes. A repaint, not a polish.

That makes it very good on AI-generated and soft, low-detail footage, which has no real detail to
recover in the first place. It also makes it a bad idea anywhere a face has to stay the same
person, or a logo has to stay the same logo. See [Scope](#scope).

## Quick start

**Drop `workflows/ReDetail_LTX25_upscale.json` into ComfyUI.** The graph is the whole render. Its
six note panels walk you through install, prep and queue. Read the one marked **START HERE** first.

Your clip needs two things before it loads, because the graph can't do them for you:

```bash
# 1. an audio track (the graph encodes A/V jointly; most AI clips are silent)
ffmpeg -i in.mp4 -f lavfi -i anullsrc=r=48000:cl=stereo -shortest \
       -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac ok.mp4

# 2. a length of 8n+1 frames, or the model silently drops the tail
ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 ok.mp4
ffmpeg -i ok.mp4 -frames:v 129 -c:a copy trimmed.mp4     # 129 = 8*16+1, ~5.4s at 24fps
```

Then load the clip, load its first frame into **Load Image** (it ships with a REPLACE ME
placeholder), set `shorter_size` to your source's shorter edge, and set the output size from the
[table below](#picking-a-size).

**Or skip all of that.** `redetail.py` drives the same graph over ComfyUI's HTTP API and does the
prep, sizing, splitting and audio for you:

```bash
python3 redetail.py --setup            # checks your install, tells you what's missing
python3 redetail.py clip.mp4 --scale 1.5
```

Use the CLI for anything longer than one short pass. It solves the target resolution, splits long
clips on their own cuts, keeps every chunk on the frame grid, and re-muxes your original audio.

## Install

Python 3, `ffmpeg` and `ffprobe` on PATH, and a running ComfyUI with
[`Lightricks/ComfyUI-LTXVideo`](https://github.com/Lightricks/ComfyUI-LTXVideo). **Restart ComfyUI
after installing the node pack** or its nodes won't appear.

**Two dependencies, into the same Python ComfyUI actually runs from:**

```bash
/path/to/ComfyUI/venv/bin/python -m pip install "kornia==0.7.4" "comfy-kitchen>=0.2.26"
```

```powershell
.\python_embeded\python.exe -m pip install "kornia==0.7.4" "comfy-kitchen>=0.2.26"
```

A venv install is invisible to a system-Python ComfyUI, and the symptom is baffling: the node pack
imports fine by hand while ComfyUI still reports its nodes missing. The `kornia` pin is exact and
older than some other packs want, so if other workflows break after installing ReDetail, that's why.

**The models.** Both Hugging Face repos are gated; accept the licence on **each**, including the
base repo, or the weights 403 while the README loads fine.

| file | folder | size |
|---|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `models/diffusion_models/` | 21.5 GB |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `models/text_encoders/` | 15.4 GB |
| `ltx-2.5-video-vae-bf16.safetensors` | `models/vae/` | 1.5 GB |
| `ltx-2.5-audio-vae-bf16.safetensors` | `models/vae/` | 0.4 GB |
| `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` | `models/loras/` | 0.3 GB |

**Your GPU decides which weights you need.** `int8_convrot` requires **Blackwell** (RTX 50-series,
RTX PRO 6000, B200). That's an architecture limit, not a memory one. On a 4090 they won't load at
any resolution however much VRAM is free.

On anything older, run the GGUF path. All three parts are required; swapping only the transformer
still fails, because the int8 *text encoder* is Blackwell-only too:

```bash
python3 redetail.py clip.mp4 --scale 1.5 \
  --gguf LTX-2.5-Distilled-Q4_K_M.gguf \
  --encoder gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --clip-device cpu --budget 150 --decode-tile 256 --decode-temporal 32
```

That needs [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) plus the Q4_K_M
transformer from [`Abiray/LTX-2.5-Distilled-GGUF`](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF)
in `models/unet/`, and the bf16 encoder (26GB) on CPU, where it wants ~26GB of system RAM. Measured
on an RTX 4090: sampling took 70s for 8 steps at 21.8 GB of 24.5 peak. Other Ada and Ampere cards
should follow, but we haven't tested them.

## Picking a size

Both output dimensions must divide by **64**, not 32, because the IC-LoRA guide splits the latent into 2x2
patches. Get it wrong and you get `Error while processing rearrange-reduction pattern`.

| your source | 1.5x | 2x |
|---|---|---|
| 640x384 | 960x576 | 1280x768 |
| 768x1408 | 1152x2112 | 1536x2816 |
| 1024x576 | **1472x832** (1.44x) | 2048x1152 |
| 1088x1920 | **1600x2816** (1.47x) | 2176x3840 |
| 1920x1088 | **2816x1600** (1.47x) | 3840x2176 |

The bold rows have **no exact 1.5x on the grid at all**. 1920x1088 is the clearest case: its exact
1.5x is 2880x1632, and 1632 isn't /64. 2x is exact far more often, because doubling a /64 number
stays /64. The CLI works all this out, and also puts your *source* on the grid, so it may crop a few
percent or resample, so watch the `source ->` line it prints if your framing is tight.

**Memory.** Keep `frames × (width × height ÷ 1,000,000)` under your card's budget. The 96 GB row is
measured; the others are starting points:

| VRAM | keep under | example that fits |
|---|---|---|
| 96 GB | ~850 | 129 frames at 2816x1600 |
| 48 GB | ~350 | 129 frames at 1472x832 |
| 24 GB | ~150 | 129 frames at 960x576 |

Sampling and decode run out of memory *independently*. Our 4090 sampled all 8 steps cleanly and
then OOMed in VAE decode. If you crash after the progress bar completes, lower `--decode-tile`
rather than your resolution.

**What each scale costs.** Single runs on one clip (243 frames from 768x1408) on one RTX PRO 6000,
indicative rather than spec: 1:1 took 3 min at 52 GB, 1.5x took 7 min at 65 GB, 2x took 17 min at
80.5 GB. 1.5x is the sweet spot on cost; bigger scales synthesize more detail, but whether you
*want* more is taste, not a quality ranking.

## When something breaks

| you see | it means | do this |
|---|---|---|
| `Is a directory` on Load Image | no image loaded | load your clip's first frame |
| `VAEEncodeAudio: input audio is None` | silent clip | add a silence track |
| `rearrange-reduction pattern` | a dimension isn't /64 | use the sizing table |
| `'NoneType' has no attribute 'Params'` | comfy-kitchen too old | `>=0.2.26`, in ComfyUI's own Python |
| `cannot import name 'pad' from kornia…` | kornia 0.8.x | pin `kornia==0.7.4` |
| node missing / `no class_type` | pack failed to import | restart ComfyUI, then read its **boot log** |
| OOM *after* the progress bar finishes | decode spike, not sampling | lower `--decode-tile` |
| output came back 1:1 despite your settings | canvas link not severed | use the shipped workflow, not the stock example |

`python3 redetail.py --setup` catches most of these before you hit them, including on a remote
ComfyUI. It needs a directly reachable, unauthenticated endpoint, and it sends no credentials. Pass
the same `--gguf` / `--encoder` flags you intend to render with, or it checks the wrong file set.

## Scope

Treat this as a generative re-renderer, not restoration, as
[the model card](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler) says
too. **Nothing in the model preserves identity.** Features that survive are luck, not a mechanism.

In informal tests on one person, some features looked consistent afterwards and lashes resolved
individually where the source was mush. But it also **added freckles that were not in the
original**. On a motocross clip it re-drew the jersey graphic and number plate: stable across
frames, but not the source's markings. **Any logo, number or text gets re-imagined.** Fine on
AI-generated footage, disqualifying on real sponsor or product work.

So compare a **face** against your source, not overall sharpness. There's no fidelity dial.

**Known limitations.** No resume: a failure on chunk 9 of 10 re-renders all nine. No polling
deadline, so if ComfyUI dies mid-run the CLI waits rather than failing; Ctrl-C and re-run. Uploads
buffer in memory. Mid-shot seams are possible when VRAM forces a split with no cut nearby. The
drag-and-drop workflow preps nothing. That's what the CLI is for. And it overwrites its output
without asking, using `<output>_chunks/` as scratch (it deletes only files it created there).

## Every flag

```
input                 video to upscale (omit with --setup)
--setup               check the install and exit
--scale     1.5       1.0 re-detail in place · 1.5 the sweet spot · 2.0 heaviest
--comfy               ComfyUI URL (default http://127.0.0.1:8188)
--out                 output file (default <input>_redetail.mp4)
--budget    850       frame-megapixels per chunk, the VRAM dial. Lower it if you OOM
--audio     original  'original' re-muxes your track; 'generated' keeps the model's
--keep-chunks         keep the per-chunk intermediates
--gguf                GGUF transformer in models/unet, required on non-Blackwell
--encoder             text encoder filename (bf16 on non-Blackwell)
--clip-device         'cpu' or 'default'; omitted leaves the workflow value alone
--decode-tile         VAEDecodeTiled tile_size, lower this for decode OOMs
--decode-temporal     VAEDecodeTiled temporal_size (default 128)
```

Tested against ComfyUI 0.32.0 with ComfyUI-LTXVideo, `kornia==0.7.4` and `comfy-kitchen 0.2.26`,
on an RTX PRO 6000 Blackwell and an RTX 4090, August 2026.

## Licence

**The MIT grant in `LICENSE` covers this project's own code only**: `redetail.py`,
`build_ui_workflow.py`, `tools/` and `demos/`. It covers neither the weights nor the workflows.

**The workflow files are modified derivatives** of Lightricks'
`example_workflows/2.5/LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json`, distributed **exclusively
under the LTX-2 Community License Agreement**. A complete copy is included at
[`workflows/LICENSE-LTX-2-Community.txt`](workflows/LICENSE-LTX-2-Community.txt), as that licence
requires. `LICENSE` lists exactly what was changed. Lightricks' unmodified example is not
redistributed here; `build_ui_workflow.py` tells you where to get it.

That licence carries **use restrictions** which apply to you and to anyone you pass these files to,
and any entity over **$10,000,000** annual revenue needs a separate paid commercial licence from
[Lightricks](https://ltx.io/model/licensing) first. The weights are under the same Agreement; the
IC-LoRA under its own terms. Both Hugging Face repos are gated.
