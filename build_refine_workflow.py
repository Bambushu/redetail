#!/usr/bin/env python3
"""Build the ReDetail refine workflow from Lightricks' TiledFusion upscale example.

DERIVED, NOT AUTHORED, for the reason build_ui_workflow.py gives: a ComfyUI UI graph is nodes plus
subgraph definitions wired with integer link ids, and patching the shipped example keeps every link
valid and keeps this mergeable when Lightricks updates theirs.

GRAPH CHANGES
  1. MODEL NAMES. The example names the bf16 pair (42GB + 26GB) and, in the enhancer slot, a
     gemma4_e2b file that is not in the LTX-2.5 repo. All go to the int8_convrot quants, and so do
     the download links behind ComfyUI's "Missing Models" panel, which would otherwise fetch 68GB
     of bf16 weights (plus an LTX-2.3 LoRA this graph never loads).
  2. THE TILE. The sampler's tile_width/height are wired from Get Tiling Sizes, whose presets follow
     the SOURCE, and any source within 200px of a preset keeps its own size: the "HD" preset turned
     a 768x1376 clip into ONE 768x1376 tile, 1.8x the area the LoRA trained on. Those two links are
     cut, so the sampler's own widgets are live at the card's trained tile, 1024x576.
  3. THE PROMPT-ENHANCER BRANCHES. Disconnected, for the reason ReDetail 1.1 disconnected the one in
     the pixel graph: ComfyUI validates every node upstream of an output even on a branch that is
     never taken, so a checkpoint the enhancer names but you do not have blocks the whole queue.
     This graph has two: the Gemma-API conditioning switches, and a TextGenerateLTX2Prompt enhancer
     on the positive prompt. Both are rewired around, which leaves them unreachable, and ComfyUI
     prunes unreachable nodes before it validates.
  4. THE PROMPTS. The example ships both prompt boxes empty. These carry the model card's own
     look-and-style prompts. They stay generic on purpose: every tile receives the whole prompt.
  5. OUTPUT SIZE defaults to FullHD instead of 4K, so a first run takes minutes rather than half an
     hour. The tiling-preview preset moves to qHD, the preset nearest the pinned tile, so the tile
     counts it prints are close to what the sampler actually runs.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "workflows", "_base_refine_ui.json")
OUT = os.path.join(HERE, "workflows", "ReDetail_LTX25_refine.json")

TRANSFORMER = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
ENCODER = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
REFINE = "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors"
HF = "https://huggingface.co/Lightricks"
URLS = {
    TRANSFORMER: (f"{HF}/LTX-2.5/resolve/main/diffusion_models/{TRANSFORMER}", "diffusion_models"),
    ENCODER: (f"{HF}/LTX-2.5/resolve/main/text_encoders/{ENCODER}", "text_encoders"),
}
# The model card's example prompts, verbatim.
POS = ("sharp photographic detail, crisp natural texture, fine surface detail, clean edges, "
       "natural film grain, high resolution footage")
NEG = "blurry, soft, plastic, smeared detail, oversharpened halos, warped faces"

if not os.path.exists(BASE):
    raise SystemExit(
        f"Missing {BASE}.\nIt is the upstream example this workflow is derived from. Copy it from\n"
        "ComfyUI-LTXVideo/example_workflows/2.5/LTX-2.5_V2V_TiledFusion_Upscale.json")
d = json.load(open(BASE))
nodes = {n["id"]: n for n in d["nodes"]}
for _id in (5004, 5014, 5508, 5509, 5516, 9002):
    if _id not in nodes:
        raise SystemExit(f"Upstream example has changed: node {_id} is gone. Re-derive by hand "
                         f"rather than writing a workflow that loads but renders wrong.")
subs = {s.get("name"): s for s in d["definitions"]["subgraphs"]}


def setw(node, name, value, expect=None):
    """Set a widget BY NAME, in both places the file stores it.

    Current frontends write every value twice: positionally in `widgets_values` and by name in
    `widgets_values_named`. Patching only the positional list leaves the stale copy beside it, and
    which one wins on load is the frontend's business. The two are not even guaranteed to agree
    upstream (one Gemma node here has them misaligned), so the old values are checked first.
    """
    named = node.get("widgets_values_named")
    if named is None or name not in named:
        raise SystemExit(f"node {node['id']} has no widget named {name!r}")
    i = list(named).index(name)
    if node["widgets_values"][i] != named[name]:
        raise SystemExit(f"node {node['id']} {name!r}: positional and named values disagree")
    if expect is not None and named[name] != expect:
        raise SystemExit(f"node {node['id']} {name!r} is {named[name]!r}, expected {expect!r}")
    node["widgets_values"][i] = named[name] = value


# 1. MODEL NAMES. The enhancer slot becomes unreachable below, but it still shows on the canvas and
# in the Missing Models panel, so it names a file you already have rather than one you cannot get.
_g = nodes[5004]
setw(_g, "ic_lora", REFINE, expect=REFINE)
setw(_g, "lora strength", 1, expect=1)
setw(_g, "ltx_2.5_model_checkpoint", TRANSFORMER)
setw(_g, "gemma4_e2b_enhancer", ENCODER)
setw(_g, "gemma4_12b_encoder", ENCODER)


def repoint(models, old, new):
    """Swap one entry of a node's properties.models download list for another."""
    for m in models:
        if m["name"] == old:
            m["name"], (m["url"], m["directory"]) = new, URLS[new]
            return True
    return False


_lm = {n["id"]: n for n in subs["Load Models"]["nodes"]}
for _nid, _field, _old, _new in (
        (5602, "unet_name", "ltx-2.5-22b-distilled-transformer-bf16.safetensors", TRANSFORMER),
        (5604, "clip_name", "gemma4_e2b_it_bf16.safetensors", ENCODER),
        (5605, "clip_name", "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", ENCODER)):
    _n = _lm[_nid]
    if not repoint(_n["properties"]["models"], _old, _new):
        raise SystemExit(f"Load Models node {_nid} no longer names {_old}")
    setw(_n, _field, _new, expect=_old)
# The group node carries the merged list the Missing Models panel reads. Rebuild it from what the
# graph actually loads: the enhancer entry and the LTX-2.3 deblur LoRA would only send people to
# downloads nothing here uses.
_agg = nodes[5004]["properties"]["models"]
_keep = [m for m in _agg if m["name"] not in ("gemma4_e2b_it_bf16.safetensors",
                                              "ltx-2.3-22b-ic-lora-deblur-0.9.safetensors")]
if not repoint(_keep, "ltx-2.5-22b-distilled-transformer-bf16.safetensors", TRANSFORMER) or \
        not repoint(_keep, "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", ENCODER):
    raise SystemExit("Load Models' download list changed upstream; re-derive it by hand")
nodes[5004]["properties"]["models"] = _keep

# 2. THE TILE. Cut the two links from Preprocess (Get Tiling Sizes) into the sampler group.
cut = set()
for inp in nodes[5516].get("inputs", []):
    if inp.get("name") in ("tile_width", "tile_height") and inp.get("link") is not None:
        cut.add(inp["link"])
        inp["link"] = None
# Assert, do not assume: a silent no-op here ships the 1.8x tile with nothing to show for it.
if len(cut) != 2:
    raise SystemExit(f"Expected exactly 2 tile links to sever, found {len(cut)}. Upstream graph "
                     f"changed; do not ship this.")
d["links"] = [l for l in d["links"] if l[0] not in cut]
for _o in nodes[9002].get("outputs", []):
    if _o.get("links"):
        _o["links"] = [x for x in _o["links"] if x not in cut]
# Now-live widgets: confirm they hold the card's tile (and rewrite both copies, which asserts so).
setw(nodes[5516], "tile_width", 1024, expect=1024)
setw(nodes[5516], "tile_height", 576, expect=576)

# 3. THE ENHANCER BRANCHES, inside Input Parameters.
ip = subs.get("Input Parameters")
if ip is None:
    raise SystemExit("Input Parameters subgraph is gone; do not ship unpatched enhancer branches")
_links = {l["id"]: l for l in ip["links"]}
# (link id, expected origin, new origin, new origin slot)
_rewire = ((13790, 5558, 1241, 0),       # positive_encoded: past the Gemma-API switch
           (13791, 5560, 1241, 1),       # negative_encoded: same
           (4, 9004, -10, 2))            # prompt switch's on_true: raw prompt, not the enhancer
for _lid, _was, _new, _slot in _rewire:
    _l = _links.get(_lid)
    if _l is None or _l["origin_id"] != _was:
        raise SystemExit(f"Link {_lid} no longer comes from node {_was}. Upstream graph changed; "
                         "fix this before shipping, or the enhancer blocks non-default installs.")
    _l["origin_id"], _l["origin_slot"] = _new, _slot
_inner = {n["id"]: n for n in ip["nodes"]}
for _nid, _gone in ((5558, 13790), (5560, 13791), (9004, 4)):
    for _o in _inner[_nid].get("outputs", []):
        _o["links"] = [x for x in (_o.get("links") or []) if x != _gone]
_inner[1241]["outputs"][0]["links"] = sorted(set(_inner[1241]["outputs"][0]["links"]) | {13790})
_inner[1241]["outputs"][1]["links"] = sorted(set(_inner[1241]["outputs"][1]["links"]) | {13791})
_prompt_in = ip["inputs"][2]
if _prompt_in.get("name") != "value_2":
    raise SystemExit(f"Input Parameters' third input is {_prompt_in.get('name')!r}, not the prompt")
_prompt_in["linkIds"] = sorted(set(_prompt_in.get("linkIds") or []) | {4})

# 4. THE PROMPTS.
setw(nodes[5508], "value", POS, expect="")
setw(nodes[5509], "value", NEG, expect="")

# 5. SIZES, on the Preprocess group.
setw(nodes[9002], "tile_size", "qHD", expect="HD")
setw(nodes[9002], "output_size", "FullHD", expect="4K")

NOTES = {
    5526: """# ReDetail 2.0: refine mode (the default)

**It rebuilds the fine detail a soft clip is missing and leaves everything else where it was.**
Lightricks' Refine-Details IC-LoRA re-renders every frame from your clip itself, in overlapping
1024x576 tiles fused at every step, so framing, colour, motion and faces stay locked to the source
while texture, edges and grain come back. Memory is bounded by the tile, not the frame: 4K fits a
32GB card.

The 1.1 behaviour, which repaints harder and invents more (including on faces), is still shipped as
**pixel mode**: `ReDetail_LTX25_upscale.json`.

## START HERE: nothing works until this is done

**1. Node pack:** `Lightricks/ComfyUI-LTXVideo`, **updated on or after 24 September 2026** (that is
when `LTXVTiledFusionSampler` landed). Install its `requirements.txt` into the python ComfyUI runs
from, then **restart ComfyUI**. `No module named 'colour'` in the boot log means the requirements
step was skipped.

**2. One pin, same python:** `pip install "comfy-kitchen>=0.2.26"`. Older versions fail on every
int8_convrot checkpoint with an error that looks like unsupported weights. The `kornia==0.7.4` pin
from 1.1 is no longer needed.

**3. Models.** Both HF repos are **gated**: accept the licence on *each*, including the base repo.
The **Missing Models** panel can fetch them once you have.

| file | folder | size |
|---|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `diffusion_models/` | 21.5GB |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `text_encoders/` | 15.4GB |
| `ltx-2.5-video-vae-bf16.safetensors` | `vae/` | 1.5GB |
| `ltx-2.5-audio-vae-bf16.safetensors` | `vae/` | 0.4GB |
| `ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors` | `loras/` | 1.3GB |

🤗 `Lightricks/LTX-2.5` · `Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details`

`int8_convrot` is not Blackwell-only; it runs on Ampere and Ada too.

**4. Smoke test first:** a 17-frame clip at the FullHD default. It separates "my install is
broken" from "my settings are wrong" in a few minutes.

## Or skip the arithmetic

`redetail.py` drives this graph over ComfyUI's HTTP API: exact scales (1.5x, not just presets),
silence for silent clips, the frame grid, the tail pad, cut-aware splitting and your original audio.

```
python3 redetail.py myclip.mp4 --scale 1.5
```""",

    5527: """## 1: Prep your clip (before you load it)

**a. It needs an audio track.** The graph carries your audio through, and stops with
`VAEEncodeAudio: input audio is None` on a silent file. AI clips usually are:

```
ffmpeg -i in.mp4 -f lavfi -i anullsrc=r=48000:cl=stereo -shortest -c:v copy -c:a aac ok.mp4
```

**b. Trim to 8n+1 frames** (9, 17, 25 ... 97, 121 ... 241). Any other length silently loses frames
off the end.

```
ffprobe -v error -count_frames -select_streams v:0 \\
        -show_entries stream=nb_read_frames -of csv=p=0 in.mp4
ffmpeg -i in.mp4 -frames:v 97 -c:a copy ok.mp4
```

**c. Optional:** the last one or two frames of any LTX render come out about 10% softer. Pad the
clip with 8 copies of its last frame and trim them off the result. The CLI does this for you.

Then load `ok.mp4` into **Load Video**. This mode needs no first-frame image.

**The prompt boxes** carry Lightricks' look-and-style prompts. Keep any edit about the *rendering*
(sharpness, texture, grain, light): every tile receives the whole prompt, so a subject you name can
be painted into tiles that do not contain it.

**Ignore the API key and enhancer inputs.** They are disconnected, as in 1.1: ComfyUI validated
them even switched off, and a missing enhancer file blocked the whole queue.""",

    9003: """## 2: Pick the output size

Set **output_size** on Preprocess: **FullHD** (1920 long edge), **4K** (3840) or 8K. The canvas
keeps your clip's aspect on a 32-pixel grid by itself; a source already within 200px of the target
keeps its own size.

Lightricks' card calls **4K the sweet spot for a 1080p source.** For an exact scale such as 1.5x, use the CLI
(`--scale 1.5`); the presets only know long edges.

**8K: not in one pass.** A 1024x576 tile covers 13% of an 8K frame, twice the scale the LoRA
trained at, and it starts inventing texture. Refine to 4K, then upscale that with lanczos.

The **Tiling sizes** preview prints the canvas, output and window plan. Its tile line shows the
nearest preset (qHD); the sampler itself runs the pinned 1024x576 tile.""",

    5529: """## 3: Queue it

**8 steps, cfg 1, euler.** Leave them unless you know why you are changing them.

**The tile is pinned to 1024x576**, the window the LoRA trained on. The stock graph derived it
from your source instead, and on a 768x1376 clip that produced a single tile 1.8x the trained size.
For a portrait clip you can swap the two numbers to **576x1024** (the other trained size). Keep
**overlap_frac at 0.5**; lower draws a grid on fine periodic texture.

**VRAM is bounded by the tile, not the canvas.** Measured on an RTX 5090: 97 frames to 2176x3904
in one pass, 14 min, 28.4GB of VRAM. Long clips run as 97-frame windows inside the sampler, so one
render covers a whole shot.

**System RAM is the other limit.** This graph holds the clip resized to the output and the result
as full float frames, about 33MB per output frame-megapixel on top of ~22GB of staged weights:
that 4s 4K pass peaked at 52GB of RAM. With less, render FullHD here, or use the CLI, whose
`--budget` splits long clips to fit.

**Guide strength** (`strength` on the IC-LoRA guide, inside Preprocess) is the fidelity dial.
1.0 locks to the source. In one test, 0.6 sharpened a face but drifted from it, and 0.4 replaced
it with a different person. Lower it for texture, not to fix a face.""",

    5531: """## Errors

| you see | it means | do this |
|---|---|---|
| `VAEEncodeAudio: input audio is None` | silent clip | add a silence track (1a) |
| `No module named 'colour'`, LTX nodes missing | pack requirements not installed | install its requirements.txt, restart |
| `LTXVTiledFusionSampler` missing | node pack too old | update ComfyUI-LTXVideo (24 Sep 2026+) |
| `'NoneType' has no attribute 'Params'` | comfy-kitchen too old | `>=0.2.26`, in ComfyUI's own python |
| a model name `not in list` | that file is missing or named differently | pick your file in Load Models |
| node missing / `no class_type` | a pack failed to import | read ComfyUI's **boot log** |
| OOM while decoding | decode spike | lower `tile_size` on Decode |""",

    5532: """## 4: When it finishes

Output lands in `ComfyUI/output/`. SaveVideo's mp4 understates the render (Lightricks measured
about 8 Mbps at 4K); the CLI saves at higher quality.

**Re-mux your original audio.** The graph passes yours through the audio VAE, which is close but
not bit-exact:

```
ffmpeg -i upscaled.mp4 -i original.mp4 -map 0:v -map 1:a -c:v copy -shortest final.mp4
```

**What it will not do:** repair a face that is malformed in the source (it rebuilds the detail of
the face that is there), recover very small lettering, or restore colour on archive footage.

Licence: LTX-2-community-license.""",
}
for nid, text in NOTES.items():
    if nid not in nodes:
        raise SystemExit(f"Note panel {nid} is gone upstream; re-place the notes by hand")
    setw(nodes[nid], "text", text)

TITLES = {
    5001: "① Load Video: your prepped clip",
    9002: "② Preprocess: output_size (FullHD / 4K)",
    5004: "Models: int8_convrot + Refine-Details IC-LoRA",
    5516: "③ Tiled Fusion: 8 steps, 1024x576 tile (pinned)",
    5518: "Decode: lower tile_size if you OOM here",
    4852: "④ Result",
}
for nid, t in TITLES.items():
    nodes[nid]["title"] = t

d["extra"] = d.get("extra", {})
d["extra"]["redetail"] = {"version": "2.0", "mode": "refine", "derived_from":
                          "ComfyUI-LTXVideo/example_workflows/2.5/"
                          "LTX-2.5_V2V_TiledFusion_Upscale.json"}
json.dump(d, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
print(f"  nodes {len(d['nodes'])}, links {len(d['links'])} (cut {len(cut)} tile wires, "
      f"rewired {len(_rewire)} enhancer links)")
print("  tile 1024x576 literal; output_size FullHD; prompts set")
