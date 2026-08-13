# ReDetail

**A generative video re-detailer for ComfyUI.** It runs your clip back through
[LTX-2.5's pixel spatial upscaler](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler),
which doesn't sharpen or recover anything — it re-renders the whole thing and invents the fine
detail as it goes. A repaint, not a polish.

That makes it very good at AI-generated footage and soft, low-detail sources, which have no real
detail to recover in the first place. It also makes it a bad idea anywhere the face has to stay the
same person. More on that in [Scope](#scope).

**You need:** Python 3, `ffmpeg` and `ffprobe` on your PATH, a running ComfyUI with
[`Lightricks/ComfyUI-LTXVideo`](https://github.com/Lightricks/ComfyUI-LTXVideo), and the five model
files below. Run both commands from a clone of this repo — `redetail.py` reads `workflows/`
relative to itself.

```bash
python3 redetail.py --setup          # check your install first
python3 redetail.py myclip.mp4 --scale 1.5
```

| file | what it is |
|---|---|
| `workflows/ReDetail_LTX25_upscale.json` | the drag-and-drop ComfyUI workflow |
| `redetail.py` | the CLI — handles sizing, splitting, frame arithmetic and audio for real clips |
| `workflows/ltx25_upscale_API.json` | the API-form graph the CLI patches per chunk. Not something you can POST as-is, and there's no build step asserting it stays in step with the drag-and-drop file |

### Using the drag-and-drop workflow by hand

Drop the `.json` into ComfyUI, then — because the graph can't do any of this for you:

1. Prepare the clip: add silence if it's silent, trim to 8n+1 frames
   (see [Preparing a clip](#preparing-a-clip--for-the-manual-workflow-only)).
2. Extract its first frame, and load **both** files. The workflow ships with a placeholder image;
   if you leave it, you'll render the placeholder.
3. Open the **Source Video** group and set `shorter_size` to your source's shorter edge. It ships
   at 768, which is right only if your source happens to be 768 on its short side.
4. Set the output width and height from the [sizing table](#picking-a-size).

---

## Getting set up

Start with `python3 redetail.py --setup`. It talks to your ComfyUI, checks the model files are
present and in the right folders, confirms the nodes loaded, and flags int8 weights on a card that
can't run them. Everything below is what it checks, in case you'd rather do it by hand.

Pass the same variant flags you intend to render with, or it will check the wrong file set:

```bash
python3 redetail.py --setup --comfy http://127.0.0.1:8188
python3 redetail.py --setup --gguf LTX-2.5-Distilled-Q4_K_M.gguf \
        --encoder gemma4-12b-with-proj-ltx-2.5-bf16.safetensors --clip-device cpu
```

`--comfy` can point at another machine, as long as it's a **directly reachable ComfyUI that accepts
unauthenticated API requests** — the client sends no credentials or custom headers, so hosted
services behind an API key or an SSO proxy won't work.

Two things `--setup` does *not* do: it doesn't read your installed `kornia` / `comfy-kitchen`
versions (it only names them when a node fails to import), and its GPU check is a name-substring
allowlist — `5070`, `5080`, `5090`, `RTX PRO 6000`, `B200`, `B300`, `GB200` — not a compute
capability probe. An unlisted Blackwell card, or a multi-GPU box, can be classified wrongly.

### Your GPU decides which weights you need

The `int8_convrot` weights need a **Blackwell** card — RTX 50-series, RTX PRO 6000, B200. That's an
architecture requirement rather than a memory one: on a 4090 they won't load at any resolution, no
matter how much VRAM is free.

On anything older, use the GGUF build instead: it dequantizes inside the compute kernel, so it isn't
tied to a tensor layout. We measured a full render on an **RTX 4090** — see
[Running on a 4090](#running-on-a-4090). Other Ada and Ampere cards should follow from the same
reasoning, but we haven't tested them, and VRAM headroom will differ.

### One exact pin, one minimum version

`kornia` must be exactly 0.7.4; `comfy-kitchen` just has to be new enough.

```bash
# POSIX — venv or manual install
/path/to/ComfyUI/venv/bin/python -m pip install "kornia==0.7.4" "comfy-kitchen>=0.2.26"
```

```powershell
# Windows portable
.\python_embeded\python.exe -m pip install "kornia==0.7.4" "comfy-kitchen>=0.2.26"
```

They have to go into **the same Python ComfyUI actually runs from**. A venv install is invisible to
a system-Python ComfyUI, and the symptom is baffling: the node pack imports fine when you try it by
hand, and ComfyUI still reports its nodes missing. That one cost us a full debugging session on a
fresh machine.

One warning worth reading twice: **`kornia==0.7.4` is older than some other node packs want.** If
other workflows break after you install ReDetail, this pin is why. We can't avoid it — kornia 0.8.x
breaks ComfyUI-LTXVideo outright with `cannot import name 'pad' from kornia.geometry.transform.pyramid`.

### The node pack

[`Lightricks/ComfyUI-LTXVideo`](https://github.com/Lightricks/ComfyUI-LTXVideo), and **restart
ComfyUI afterwards** or its nodes simply won't appear. If a node pack fails to import, ComfyUI logs
it as a *warning* and carries on — every node then resolves to `class_type: null` and your prompt is
rejected with "Node has no class_type". When that happens, read the boot log rather than the graph.

### The models

Both Hugging Face repos are gated, and you need to accept the licence on **each** — including the
base repo, not just the LoRA. If you don't, the weights 403 while the README loads fine, which looks
like "no access" rather than "you haven't clicked accept".

| file | folder | size |
|---|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `models/diffusion_models/` | 21.5 GB |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `models/text_encoders/` | 15.4 GB |
| `ltx-2.5-video-vae-bf16.safetensors` | `models/vae/` | 1.5 GB |
| `ltx-2.5-audio-vae-bf16.safetensors` | `models/vae/` | 0.4 GB |
| `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` | `models/loras/` | 0.3 GB |

Then run a 17-frame clip at 640x384 → 960x576 before anything real. It took about a minute on our
RTX PRO 6000, and it cleanly separates "my install is broken" from "my settings are wrong". We never
timed the smoke test on the 4090, so expect longer there without a number to hang on it — the CPU
text encoder dominates short runs.

---

## Preparing a clip — for the manual workflow only

**Skip this section if you're using the CLI.** It adds silence and pads each chunk to 8n+1 itself.
This is what you'd otherwise have to do by hand before ComfyUI sees your video.

**It needs an audio track.** The graph encodes audio and video together and stops with
`VAEEncodeAudio: input audio is None` if there isn't one. Since most AI-generated clips are silent,
this catches nearly everyone on their first run:

```bash
ffmpeg -i in.mp4 -f lavfi -i anullsrc=r=48000:cl=stereo -shortest \
       -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac ok.mp4
```

**Its length has to be 8n+1 frames.** The model rounds length *down* to the nearest 8n+1 — we
watched 132 come back as 129, 108 as 105, 54 as 49 — so any other count quietly loses frames off the
end. Across several chunks of a long clip that adds up, and the picture drifts against the audio.

```bash
# -count_frames decodes and counts. Slower, but plenty of containers store no nb_frames
# at all and report "N/A", which is exactly when you need the number.
ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 in.mp4
ffmpeg -i in.mp4 -frames:v 129 -c:a copy ok.mp4     # 129 = 8*16+1, about 5.4s at 24fps
```

---

## Picking a size

Both output dimensions have to divide by **64**, not 32. The IC-LoRA guide splits the latent into
2x2 patches, so the /32 latent has to be even on both axes. Get it wrong and you get
`Error while processing rearrange-reduction pattern`.

| your source | 1.5x | 2x |
|---|---|---|
| 640x384 | 960x576 | 1280x768 |
| 768x1408 | 1152x2112 | 1536x2816 |
| 1024x576 | **1472x832** (1.44x) | 2048x1152 |
| 1088x1920 | **1600x2816** (1.47x) | 2176x3840 |
| 1920x1088 | **2816x1600** (1.47x) | 3840x2176 |

The bold rows are the ones that surprise people: **plenty of common sizes have no exact 1.5x on the
/64 grid at all.** 1920x1088 is the clearest case — its exact 1.5x is 2880x1632, and while 2880 is
fine (64x45), 1632 isn't (64x25.5). So you get 2816x1600 instead: a genuine 1.47x, with a 0.27%
aspect difference that comes to about half a pixel across a face. 1024x576 lands further off still,
at 1.44x. 2x is exact far more often, because doubling a /64 number stays /64.

These are the CLI's own answers, and they're not worth deriving by hand.

**The CLI also puts your source on the /64 grid**, which means it may change your framing. If the
remainder is small it crops — up to 3% off an edge. If cropping would cost more than that it looks
for an exact-aspect /64 size and resamples the whole frame instead, falling back to a crop when no
such size exists. A 432x768 clip resizes to 576x1024 rather than losing 11% of its width. Watch the
`source ->` line it prints if your framing is tight.

**What each scale costs.** Single runs on one clip (243 frames from a 768x1408 source) on one RTX
PRO 6000 — indicative, not a spec:

| | time | VRAM |
|---|---|---|
| 1:1 | 3.0 min | 52 GB |
| 1.5x | 7 min | 65 GB |
| 2x | 17 min | 80.5 GB |

So 1.5x runs at roughly 40% of 2x's cost. Bigger scales synthesize more detail, but whether you
*want* more is a taste call rather than a quality ranking — it's all invented either way. 1:1 is a
real mode, not a failed upscale: it re-renders detail on the same grid, it just has no extra pixels
to put it in.

---

## Fitting it in memory

Work out `frames × (width × height ÷ 1,000,000)` and keep it under your card's budget. The 96 GB row
is measured; the other two are the starting points we'd pick, not measured ceilings:

| VRAM | keep under | something that fits |
|---|---|---|
| 96 GB | ~850 | 129 frames at 2816x1600 |
| 48 GB | ~350 | 129 frames at 1472x832 |
| 24 GB | ~150 | 129 frames at 960x576 |

Those come from three measurements on the int8 path at 243 frames: 1.08MP → 52GB, 2.43MP → 65GB,
4.33MP → 80.5GB. Interpolate between them, but don't extrapolate past them — the GGUF path has a
much lower floor, and a line fitted to these numbers would tell you a 4090 can't render at all,
which it demonstrably can.

**Sampling and decode run out of memory independently.** Our 4090 sampled all 8 steps cleanly and
then OOMed in VAE decode. If you crash *after* the progress bar completes, lower `--decode-tile`
rather than dropping your resolution — decode is a brief spike, not a sustained cost.

**Long clips get split**, and boundaries prefer the clip's own cuts. That matters because the model
invents detail, so two chunks can invent slightly *different* texture. Mid-shot you'd notice; at a
real cut the picture changes completely and it's invisible. The CLI scene-detects and snaps to a
nearby cut where it can — but when VRAM forces a split and no cut is close, it splits mid-shot, and
you may see a seam there.

---

## Running on a 4090

Measured end to end on an RTX 4090, 24GB:

```bash
python3 redetail.py clip.mp4 --scale 1.5 \
  --gguf LTX-2.5-Distilled-Q4_K_M.gguf \
  --encoder gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --clip-device cpu \
  --budget 150 --decode-tile 256 --decode-temporal 32
```

Sampling took **70 seconds** for 8 steps, peaking at **21.8 GB of 24.5**. End to end, a 243-frame
clip at 960x576 took about six minutes — most of that the text encoder on CPU rather than GPU work.

Three pieces are required, and each one blocks on its own:

**The GGUF transformer**, from [`Abiray/LTX-2.5-Distilled-GGUF`](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF)
into `models/unet/`, with [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) installed.
Q4_K_M is 15.1GB against int8's 21.5GB, and 16% faster — 90s versus 108s, both measured on the same
Blackwell card, since a 4090 can't run int8 at all and the two aren't comparable there.

**The text encoder on CPU.** This is the part people miss: the int8_convrot *encoder* is
Blackwell-only too, so swapping just the transformer still fails. bf16 is the only alternative
Lightricks ships, and at 26GB it won't fit beside the DiT — so it goes on the CPU, where it wants
about 26GB of system RAM. It encodes once and then sits idle through every sampling step, so that
cost is one-off rather than per-step.

**Tiled decode**, for the reason above.

---

## When something breaks

| what you see | what it means | what to do |
|---|---|---|
| `Is a directory` on Load Image | no image loaded | load your clip's first frame |
| `VAEEncodeAudio: input audio is None` | silent clip | add a silence track (see Preparing a clip) |
| `rearrange-reduction pattern` | a dimension isn't /64 | use the sizing table |
| `'NoneType' has no attribute 'Params'` | comfy-kitchen too old | `>=0.2.26`, in ComfyUI's own Python |
| `cannot import name 'pad' from kornia…` | kornia 0.8.x | pin `kornia==0.7.4` |
| node missing / `no class_type` | pack failed to import | restart ComfyUI, then read its **boot log** |
| OOM *after* the progress bar finishes | decode spike, not sampling | lower `--decode-tile` |
| output came back 1:1 despite your settings | canvas link not severed | use the shipped workflow, not the stock example |

`python3 redetail.py --setup` catches most of these before you hit them.

---

## Scope

Treat this as a generative re-renderer, not a restoration tool. That's what
[the upscaler's own model card](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler)
says too — read it before you point this at footage that has to stay faithful.

Here's what that looked like in practice, on informal eyeball comparisons of one person across a
handful of clips — no metric, no face embedding, so read it as an anecdote and not a result. Some
features looked consistent afterwards (eye colour, a distinctive hair streak, brow shape); lashes
and brow hairs resolved individually where the source was mush. In those same renders it also
**added freckles that were not in the original**, because freckles are exactly the kind of
high-frequency skin detail it likes to invent.

**Nothing in the model preserves identity.** Features that survive are luck, not a mechanism. Treat
every fine detail you see as synthesized and check anything identity-critical against the source
yourself.

So compare a **face** against your source, not overall sharpness. There's no fidelity dial to turn
down. If a face drifts, the workflow isn't broken — that's the model doing what it does.

The graph also regenerates a soundtrack as a by-product. For anything finished you'll want your own
audio back, which `redetail.py` does by default:

```bash
ffmpeg -i upscaled.mp4 -i original.mp4 -map 0:v -map 1:a -c:v copy -shortest final.mp4
```

---

## Known limitations

Better stated here than discovered later:

- **No resume.** Every run re-submits every chunk, so if ComfyUI dies on chunk 9 of 10, you re-render
  all nine.
- **No polling deadline.** If ComfyUI crashes or its history is cleared mid-run, the CLI waits
  rather than failing. Ctrl-C and re-run.
- **Uploads buffer in memory.** Chunks are small by design, but a very large one needs the RAM.
- **The drag-and-drop workflow doesn't prep your clip** — no silent-audio fix, no 8n+1 trim, no
  splitting. That's what the CLI is for.
- **Mid-shot seams are possible** when VRAM forces a split with no cut nearby.
- **It overwrites your output file** without asking, and it uses `<output>_chunks/` as a scratch
  directory. It deletes only files it wrote there and leaves anything that was already present, but
  don't aim `--out` at a name whose `_chunks` folder you care about. `--keep-chunks` keeps the
  intermediates.

## Every flag

```
input                 video to upscale (omit with --setup)
--setup               check the install and exit
--scale     1.5       1.0 re-detail in place · 1.5 the sweet spot · 2.0 heaviest
--comfy               ComfyUI URL (default http://127.0.0.1:8188)
--out                 output file (default <input>_redetail.mp4)
--budget    850       frame-megapixels per chunk — the VRAM dial. Lower it if you OOM
--audio     original  'original' re-muxes your track; 'generated' keeps the model's
--keep-chunks         keep the per-chunk intermediates
--gguf                GGUF transformer in models/unet — required on non-Blackwell
--encoder             text encoder filename (bf16 on non-Blackwell)
--clip-device         'cpu' or 'default'; omitted leaves the workflow value alone
--decode-tile         VAEDecodeTiled tile_size — lower this for decode OOMs
--decode-temporal     VAEDecodeTiled temporal_size (default 128)
```

Tested against ComfyUI 0.32.0 with ComfyUI-LTXVideo, `kornia==0.7.4` and `comfy-kitchen 0.2.26`, on
an RTX PRO 6000 Blackwell and an RTX 4090, August 2026. We didn't record the node pack's commit —
if its nodes have changed since, that's the first thing to suspect.

---

## Licence

**The MIT grant in `LICENSE` covers this project's own code only** — `redetail.py` and
`build_ui_workflow.py`. It covers neither the weights nor the workflow files.

**The model weights are not MIT.** LTX-2.5 is released under the LTX-2-community-license and the
IC-LoRA under its own terms. Both Hugging Face repos are gated; read what you accept there before
any commercial use.

**The workflows are derived work.** Everything in `workflows/` comes from Lightricks' shipped
`example_workflows/2.5/LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json`, so
[ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)'s licence governs it, not ours.
Check its terms before you redistribute those files. We don't reproduce any upstream licence text
here — go read the originals.
