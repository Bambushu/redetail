#!/usr/bin/env python3
"""Build the ReDetail drag-and-drop ComfyUI workflow from Lightricks' V2V IC-LoRA example.

DERIVED, NOT AUTHORED. A ComfyUI UI graph is 18 nodes plus six subgraph definitions wired with
integer link ids — hand-writing or generating one produces a file that silently fails to load.
Patching the shipped example keeps every link valid and keeps this mergeable when Lightricks
updates theirs.

GRAPH CHANGES
  1. MODEL NAMES. The example names bf16 weights (42GB + 26GB) and, in the enhancer slot, a
     gemma4_e2b file that IS NOT IN THE LTX-2.5 REPO. ComfyUI validates every node upstream of an
     output even on a disabled branch, so that missing file fails the whole prompt. Both go to the
     int8_convrot quant.
  2. THE IC-LORA. Ships pointing at instant-shave; swapped for the pixel spatial upscaler. That one
     node is what makes this an upscaler.
  3. THE CANVAS LINK. EmptyLTXVLatentVideo's width/height are promoted to the Preprocess node but
     WIRED to GetImageSize on the source, so the graph renders 1:1 whatever you type. Those two
     links are cut, which makes the widgets live. `length` stays linked (auto-read from the clip) —
     verified by resolving the graph, which is why the notes say to trim the CLIP to 8n+1 rather
     than type a number.
  4. PLACEHOLDER IMAGE, so the empty LoadImage cannot throw Errno 21 on the first queue.

NOTE STRUCTURE — rewritten after a UX review (Kimi K3). The first version was ordered by the order
I hit the problems, which is not the order a stranger acts in. Now: install -> prep -> load ->
queue -> verify, with an error lookup table. Mechanisms and measurements moved to the README;
panels carry rules and commands only.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "workflows", "_base_ui.json")
OUT = os.path.join(HERE, "workflows", "ReDetail_LTX25_upscale.json")

TRANSFORMER = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
ENCODER = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
ICLORA = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
PLACEHOLDER = "redetail_replace_me.png"

if not os.path.exists(BASE):
    raise SystemExit(
        f"Missing {BASE}.\nIt is the upstream example this workflow is derived from — copy it from\n"
        "ComfyUI-LTXVideo/example_workflows/2.5/LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json")
d = json.load(open(BASE))
nodes = {n["id"]: n for n in d["nodes"]}
for _id in (5004, 5014, 2004, 9002, 5548):
    if _id not in nodes:
        raise SystemExit(f"Upstream example has changed: node {_id} is gone. Re-derive by hand "
                         f"rather than writing a workflow that loads but renders wrong.")

w = nodes[5004]["widgets_values"]
w[2], w[3], w[4], w[5], w[6] = TRANSFORMER, ENCODER, ENCODER, ICLORA, 1
nodes[5014]["widgets_values"][6] = TRANSFORMER
nodes[2004]["widgets_values"] = [PLACEHOLDER, "image"]

cut = set()
for inp in nodes[9002].get("inputs", []):
    if inp.get("name") in ("width", "height") and inp.get("link") is not None:
        cut.add(inp["link"])
        inp["link"] = None
# Assert, do not assume. If upstream rewires this, a silent no-op here produces a workflow that
# loads fine and renders 1:1 — the single most confusing failure this tool has.
if len(cut) != 2:
    raise SystemExit(f"Expected exactly 2 canvas links to sever, found {len(cut)}. "
                     f"Upstream graph changed — do not ship this.")
d["links"] = [l for l in d["links"] if l[0] not in cut]
pre = nodes[9002]["widgets_values"]
pre[2], pre[3] = 1152, 2112

# Reference resize, buried in the "Source Video" subgraph. The example hardcodes 544; it must equal
# the SOURCE clip's shorter side or the IC-LoRA guide is resampled to a different scale than the
# render was validated at. Default it to 768 (the 768x1408 row of the size table) and document it.
_resized = 0
for _sg in d["definitions"]["subgraphs"]:
    for _n in _sg.get("nodes", []):
        if _n.get("type") == "ResizeImageMaskNode" and _n.get("widgets_values"):
            if _n["widgets_values"][0] == "scale shorter dimension":
                _n["widgets_values"][1] = 768
                _resized += 1
if _resized != 1:
    raise SystemExit(f"Expected exactly 1 reference-resize node, found {_resized}.")

NOTES = {
    # 0 — feasibility and install FIRST. This is the only panel that can fail 100% of users.
    5526: """# ReDetail — generative video re-detailer

**It re-renders your clip with invented detail. A repaint, not a polish.**
Exactly what AI-generated footage, animation and soft low-detail sources need — and exactly wrong
for identity-critical faces, where it will change them.

## ⛔ START HERE — nothing works until this is done

**1. ✅ CORRECTION — `int8_convrot` is NOT Blackwell-only.** An earlier version of this panel said
it was. That was wrong: it runs on Ampere and Ada too (reported working on 3090, 4090 and 1070).
**Use the int8 weights below on any modern NVIDIA card.**

⚠️ **If int8 fails to load, suspect `comfy-kitchen` BEFORE your GPU.** Version 0.2.10, which
several ComfyUI images ship, fails on every convrot checkpoint and reports
`'NoneType' object has no attribute 'Params'` — which reads like unsupported weights. Step 2 pins
the fix.

Only if int8 genuinely will not run: install `city96/ComfyUI-GGUF`, put
`LTX-2.5-Distilled-Q4_K_M.gguf` (15GB, 🤗 `Abiray/LTX-2.5-Distilled-GGUF`) in `models/unet/`, and
swap UNETLoader for **UnetLoaderGGUF**. Then skip the text encoder entirely with the shipped cached
conditioning rather than downloading the 26GB bf16 one —
`python3 redetail.py clip.mp4 --gguf LTX-2.5-Distilled-Q4_K_M.gguf --cached-cond`

**2. Install the two pinned dependencies — into the SAME python ComfyUI runs from.**

```
# portable
python_embeded\\python.exe -m pip install "kornia==0.7.4" "comfy-kitchen>=0.2.26"
# venv / manual
./venv/bin/python -m pip install "kornia==0.7.4" "comfy-kitchen>=0.2.26"
```
Verify: `python -c "import kornia, comfy_kitchen; print(kornia.__version__)"` must print `0.7.4`.
A venv install is invisible to a system-python ComfyUI — this is the most common failure.

⚠️ **kornia 0.7.4 is older than some other node packs want.** If other workflows break after this,
this pin is why.

**3. Node pack:** `Lightricks/ComfyUI-LTXVideo`. **Restart ComfyUI after installing it** or its
nodes will not appear.

**4. Models.** Both HF repos are **gated** — accept the licence on *each*, including the base repo,
or the weights 403 while the README loads fine.

| file | folder | size |
|---|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `diffusion_models/` | 21.5GB |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `text_encoders/` | 15.4GB |
| `ltx-2.5-video-vae-bf16.safetensors` | `vae/` | 1.5GB |
| `ltx-2.5-audio-vae-bf16.safetensors` | `vae/` | 0.4GB |
| `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` | `loras/` | 0.3GB |

🤗 `Lightricks/LTX-2.5` · `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`

**5. Smoke test before anything real:** a **17-frame** clip at 640×384 → 960×576. It finishes in
about a minute and separates "my install is broken" from "my settings are wrong".

## Or skip all the arithmetic

`redetail.py` (shipped alongside) drives this graph over ComfyUI's HTTP API and does the
dimension solving, cut-based splitting, frame arithmetic and audio re-mux for you:

```
python3 redetail.py myclip.mp4 --scale 1.5
```
Use it for anything longer than one pass. The manual path below is for single short clips.""",

    # 1 — everything that happens OUTSIDE ComfyUI, which is chronologically first
    5527: """## 1 — Prep your clip (before you load it)

**a. It needs an audio track.** This graph encodes audio and video together and stops with
`VAEEncodeAudio: input audio is None` on a silent file. AI-generated clips are usually silent:

```
ffmpeg -i in.mp4 -f lavfi -i anullsrc=r=48000:cl=stereo -shortest -c:v copy -c:a aac ok.mp4
```

**b. Trim to 8n+1 frames.** Any other length **silently loses frames off the end**.
Valid: 9, 17, 25 … 121, 129, 137, 145, 249 … (`129 frames ≈ 5.4s @ 24fps`)

```
ffprobe -v error -count_frames -select_streams v:0 \\
        -show_entries stream=nb_read_frames -of csv=p=0 in.mp4
ffmpeg -i in.mp4 -frames:v 129 -c:a copy ok.mp4
```

`-count_frames` decodes and counts. Slower, but plenty of containers store no frame count and
report `N/A` — which is exactly when you need the number.

**c. Grab the first frame** — the Load Image node needs it:

```
ffmpeg -i ok.mp4 -frames:v 1 first_frame.png
```

Then load `ok.mp4` into **Source Video** and `first_frame.png` into **Load Image**
(it ships with a REPLACE ME placeholder).

### Ignore the API key box — the enhancer is now disconnected

**Input Parameters** exposes `LTX API key`, `enhance_seed` and an effective-prompt preview. Those
belong to the prompt-enhancer. This graph runs entirely locally and needs no account, no key and
no network. Leave them empty.

⚠️ **The API branch has been REMOVED, not just switched off,** because leaving it attached broke
the workflow for a whole class of users. ComfyUI validates every node upstream of an output *even
on a branch that is never taken*, and `GemmaAPITextEncode` reads its `ckpt_name` list from
`models/diffusion_models`. Anyone whose transformer lives elsewhere — everyone on the GGUF path —
got `ckpt_name: '…' not in []` and could not queue at all. Disabling the enhancer did not help,
because the failure was at validation, not execution.

The conditioning is now wired straight from `LTXVConditioning`, so those nodes are unreachable and
ComfyUI prunes them before validating. **The render is unchanged** (verified bit-identical, PSNR
inf): that branch only activated if your API key contained `ltxv_`, and it needed the local
checkpoint anyway to read its model id — so it could never have worked on the installs this broke.
If you wanted it to avoid the 15GB encoder download, use the shipped cached conditioning instead,
which does that offline and for free.""",

    # 2 — lookup table first, rule as a footnote
    9003: """## 2 — Set your target size

**Width and Height here are the OUTPUT size.** Find your source in the table and copy a column.

| your source | 1.5× | 2× |
|---|---|---|
| 640×384 | 960×576 | 1280×768 |
| 768×1408 | 1152×2112 | 1536×2816 |
| 1024×576 | **1472×832** (1.44×) | 2048×1152 |
| 1088×1920 | **1600×2816** (1.47×) | 2176×3840 |
| 1920×1088 | **2816×1600** (1.47×) | 3840×2176 |

Any scale works as long as both output numbers **divide by 64** — that is the only rule, and it is
why the bold rows have no exact 1.5× at all. 1920×1088 is the clearest case: 2880×1632 is off-grid
(1632 is not /64), so 2816×1600 is the answer, a 0.27% aspect difference you cannot see. 2× is
exact far more often, because doubling a /64 number stays /64. If a number is off-grid you get
`Error while processing rearrange-reduction pattern`.

**Length is read from your clip automatically.** That is why step 1b trims it.

### ⚠️ One buried setting you MUST change
Open the **Source Video** group and set **shorter_size** to your clip's *shorter side*
(640x384 -> `384`, 768x1408 -> `768`, 1920x1088 -> `1088`). It ships at 768.

It resizes the reference the upscaler guides from, so anything other than your source's own short
edge silently guides at the wrong scale. It is inside the subgraph, so you will not see it on the
canvas until you open that group.""",

    5529: """## 3 — Queue it

**8 steps, cfg 1.** Leave them unless you know why you are changing them.

Work out `frames × (W×H ÷ 1,000,000)` and keep it under your card's budget:

| your VRAM | keep under | example that fits |
|---|---|---|
| 96GB | ~850 | 129 frames @ 2816×1600 |
| 48GB | ~350 | 129 frames @ 1472×832 |
| 24GB | ~150 | 129 frames @ 960×576 |

Measured on an RTX PRO 6000: 243 frames at 1152×2112 takes about 7 minutes. Consumer-card times
are not measured yet — expect longer.

**Longer clip?** Split it, and split **only at real cuts** — invented texture differs between
pieces and shows inside a continuous shot. `redetail.py` does this automatically.""",

    5531: """## Errors

| you see | it means | do this |
|---|---|---|
| `Is a directory` on Load Image | no image loaded | load your first frame (step 1c) |
| `VAEEncodeAudio: input audio is None` | silent clip | add a silence track (step 1a) |
| `rearrange-reduction pattern` | a size is not /64 | use the table in step 2 |
| `'NoneType' has no attribute 'Params'` | comfy-kitchen too old | see START HERE step 2 |
| `cannot import name 'pad' from kornia…` | kornia 0.8.x | pin `kornia==0.7.4` |
| node missing / `no class_type` | pack not loaded | restart ComfyUI; check its console |
| OOM while decoding | decode spike | lower `tile_size` on this node |""",

    5532: """## 4 — When it finishes

Output lands in `ComfyUI/output/`.

**Re-mux your original audio.** The graph regenerates a soundtrack as a by-product; for finished
material you want your own:

```
ffmpeg -i upscaled.mp4 -i original.mp4 -map 0:v -map 1:a -c:v copy -shortest final.mp4
```

⚠️ **Check a FACE against your source, not overall sharpness.** This model invents detail rather
than recovering it — in informal tests on one person it repeatedly added freckles that were not in the original. There
is no fidelity dial: if the face drifts, the graph is not broken, that is the model.

Licence: LTX-2-community-license.""",
}
for nid, text in NOTES.items():
    if nid in nodes:
        nodes[nid]["widgets_values"] = [text]

TITLES = {
    5548: "① Source Video — your prepped clip",
    2004: "① First frame of that clip",
    9002: "② TARGET SIZE — width / height (length auto-reads from the clip)",
    5004: "Models — int8_convrot + upscaler IC-LoRA",
    5516: "③ Sampler — 8 steps, cfg 1",
    5518: "Decode — lower tile_size if you OOM here",
    4852: "④ Result",
}
for nid, t in TITLES.items():
    if nid in nodes:
        nodes[nid]["title"] = t

d["extra"] = d.get("extra", {})
d["extra"]["redetail"] = {"version": "1.0", "derived_from":
                          "ComfyUI-LTXVideo/example_workflows/2.5/"
                          "LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json"}
json.dump(d, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
print(f"  nodes {len(d['nodes'])}, links {len(d['links'])} (cut {len(cut)} canvas wires)")
print(f"  canvas {pre[2]}x{pre[3]} literal; placeholder image set on LoadImage")
