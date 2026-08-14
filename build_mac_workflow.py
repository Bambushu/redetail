#!/usr/bin/env python3
"""Build the Apple Silicon variant of the ReDetail workflow from the shipped one.

    python3 build_mac_workflow.py

WHY A SEPARATE FILE. The main workflow needs int8_convrot (Blackwell-only) and a text encoder. A
Mac needs the GGUF transformer instead, and cannot spare memory for a 26GB encoder. Both changes
are mechanical, so this derives the Mac file rather than maintaining a second copy by hand.

THE CONSTRAINT THAT SHAPES ALL OF THIS. A ComfyUI subgraph INSTANCE stores its promoted widget
values as a FLAT LIST, and the mapping from that list back to the definition's nodes is implicit —
it is not written down anywhere in the file. Delete a node whose widget is promoted and every later
value silently shifts onto the wrong widget. A workflow that loads and renders wrong is worse than
one that fails.

So this NEVER deletes a node inside a subgraph. Every change is a TYPE SWAP between nodes with the
same number of promoted widgets, which leaves the flat list aligned by construction:

    UNETLoader          -> UnetLoaderGGUF     (unet_name  -> unet_name)
    CLIPTextEncode      -> LoadConditioning   (text       -> cache name)
    GemmaAPITextEncode  -> LoadConditioning   (ckpt_name  -> cache name)

Swapping the Gemma nodes rather than deleting them solves a real problem: they sit on a switch
branch that is never taken, but ComfyUI VALIDATES disabled branches, and their ckpt_name enum is
EMPTY on a GGUF-only install (nothing in models/checkpoints or models/diffusion_models). Left
alone they fail the whole prompt; deleted, they shift the widget list. Swapped, they validate
trivially and the branch stays inert.

The only outright deletion is the top-level PreviewAny, which is not in a subgraph and therefore
promotes nothing. It has to go: it consumes `effective_prompt`, which keeps the two
TextGenerateLTX2Prompt nodes upstream of an output, and those take a CLIP — which would load the
very encoder this variant exists to avoid.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "workflows", "ReDetail_LTX25_upscale.json")
OUT = os.path.join(HERE, "workflows", "ReDetail_LTX25_upscale_MAC.json")

GGUF = "LTX-2.5-Distilled-Q4_K_M.gguf"
POS, NEG = "redetail_pos", "redetail_neg"

d = json.load(open(SRC))
nodes = {n["id"]: n for n in d["nodes"]}
subs = {s.get("name"): s for s in d["definitions"]["subgraphs"]}
for need in ("Load Models", "Input Parameters"):
    if need not in subs:
        sys.exit(f"subgraph {need!r} is gone — re-derive by hand rather than shipping a guess")


def swap(sg_name, node_id, new_type, widget):
    """Retype a node in place and clear its incoming links. Returns the links it orphaned."""
    sg = subs[sg_name]
    n = next((x for x in sg["nodes"] if x["id"] == node_id), None)
    if n is None:
        sys.exit(f"node {node_id} missing from {sg_name!r} — upstream graph changed")
    n["type"] = new_type
    n["widgets_values"] = [widget]
    dropped = {i["link"] for i in n.get("inputs", []) if i.get("link") is not None}
    n["inputs"] = []
    return sg, dropped


# 1. The transformer. int8_convrot needs Blackwell tensor layouts; GGUF dequantizes in the kernel.
sg, _ = swap("Load Models", 5602, "UnetLoaderGGUF", GGUF)
inst = nodes[5004]["widgets_values"]
if len(inst) != 7:
    sys.exit(f"Load Models instance has {len(inst)} promoted widgets, expected 7 — the slot "
             "surgery below is written against that exact layout, so do not ship a guess")
# DROP the transformer slot, do not overwrite it. Retyping the loader REMOVES its widget from the
# instance's promoted list (the GGUF loader's unet_name is not promoted; its value lives in the
# definition, which swap() already set). Setting inst[2] instead of deleting it leaves 7 values
# feeding 6 widgets, and every later value lands on the wrong one -- measured: the IC-LoRA node
# received the text encoder's filename as its lora_name and the LoRA filename as a FLOAT strength.
# ComfyUI rejected it, but a shift that happens to stay type-valid would have rendered wrong.
del inst[2]

# 2 & 3. Everything that would load the text encoder becomes a cached-conditioning load.
orphan = set()
sg_ip = None
for nid, name in ((2483, POS), (2612, NEG), (5504, POS), (5505, NEG)):
    sg_ip, dropped = swap("Input Parameters", nid, "LoadConditioning", name)
    orphan |= dropped
sg_ip["links"] = [l for l in sg_ip["links"] if l.get("id") not in orphan]
# Clear the outgoing-link bookkeeping on whatever fed them, or the file references dead links.
for n in sg_ip["nodes"]:
    for o in n.get("outputs", []):
        if o.get("links"):
            o["links"] = [l for l in o["links"] if l not in orphan]
# Same slot arithmetic on the other instance. Retyping the two CLIPTextEncode and two
# GemmaAPITextEncode nodes to LoadConditioning drops their promoted widgets (the cache names live
# in the definition), taking this list from 8 promoted values to 6. Confirmed against what
# ComfyUI's own serializer produces after loading the patched graph.
_ip = nodes[5014]["widgets_values"]
if len(_ip) != 8:
    sys.exit(f"Input Parameters instance has {len(_ip)} promoted widgets, expected 8 — do not ship")
del _ip[6:]

# 4. PreviewAny, top-level, promotes nothing. Removing it is what keeps the prompt-enhancer chain
#    (and therefore the CLIP loaders) out of the executed set.
prev = [n["id"] for n in d["nodes"] if n.get("type") == "PreviewAny"]
gone = {l[0] for l in d["links"] if l[3] in prev}
d["nodes"] = [n for n in d["nodes"] if n.get("type") != "PreviewAny"]
d["links"] = [l for l in d["links"] if l[3] not in prev]
# Whatever fed PreviewAny still lists that link on its output. Leaving it there is exactly the
# kind of stale reference this script refuses to ship.
for n in d["nodes"]:
    for o in n.get("outputs", []):
        if o.get("links"):
            o["links"] = [l for l in o["links"] if l not in gone]

# --- validate BEFORE writing: no input may reference a link that no longer exists -------------
def check(links, ns, where):
    have = {(l.get("id") if isinstance(l, dict) else l[0]) for l in links}
    bad = []
    for n in ns:
        for i in n.get("inputs", []):
            if i.get("link") is not None and i["link"] not in have:
                bad.append(f"{where} node {n['id']} input {i.get('name')} -> dead link {i['link']}")
        for o in n.get("outputs", []):
            for l in (o.get("links") or []):
                if l not in have:
                    bad.append(f"{where} node {n['id']} output -> dead link {l}")
    return bad


problems = check(d["links"], d["nodes"], "top")
for s in d["definitions"]["subgraphs"]:
    problems += check(s.get("links", []), s.get("nodes", []), s.get("name"))
# The SOURCE workflow already carries two stale output references — the canvas links that
# build_ui_workflow.py severs, which ComfyUI loads without complaint. Only fail on references THIS
# script introduced, or the check blocks on a condition it did not cause and cannot fix here.
_b = json.load(open(SRC))
baseline = set(check(_b["links"], _b["nodes"], "top"))
for s in _b["definitions"]["subgraphs"]:
    baseline |= set(check(s.get("links", []), s.get("nodes", []), s.get("name")))
if baseline:
    print(f"note: {len(baseline)} stale reference(s) inherited from the source workflow, ignored")
problems = [p for p in problems if p not in baseline]
if problems:
    print("REFUSING TO WRITE — dangling references:")
    for p in problems[:20]:
        print("  ", p)
    sys.exit(1)

# The inherited START HERE panel tells you to check for Blackwell and download a text encoder.
# Both are wrong here, and a note that contradicts the graph is worse than no note.
MAC_NOTE = """# ReDetail — Apple Silicon build

**Generative re-detailer.** It re-renders your clip and invents the fine detail. A repaint, not a
polish. Right for AI-generated and soft footage, wrong where a face or a logo must stay identical.

## ⛔ START HERE

**1. No text encoder is needed.** This build loads a cached conditioning instead. Copy the
`tools/comfyui_cond_cache/` **directory** from the repo into `ComfyUI/custom_nodes/` and restart.
The two `.pt` files inside are the whole encoder replacement (26KB each), so you skip a 26GB
download entirely.

**2. Models** — about 17GB total, no encoder:

| file | folder |
|---|---|
| `LTX-2.5-Distilled-Q4_K_M.gguf` | `models/unet/` |
| `ltx-2.5-video-vae-bf16.safetensors` | `models/vae/` |
| `ltx-2.5-audio-vae-bf16.safetensors` | `models/vae/` |
| `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` | `models/loras/` |

🤗 `Abiray/LTX-2.5-Distilled-GGUF` · `Lightricks/LTX-2.5` (**gated**, accept the licence)

**3. Node packs:** `Lightricks/ComfyUI-LTXVideo` and `city96/ComfyUI-GGUF`. Restart after both.

**4. Dependencies, into the SAME python ComfyUI runs from:**
```
./venv/bin/python -m pip install "kornia==0.7.4" "comfy-kitchen>=0.2.26"
```

**5. ⚠️ ONE ComfyUI CORE EDIT IS REQUIRED.** `comfy/ldm/lightricks/vae/na_diffusion_decoder.py`
builds RoPE frequencies in `float64`, which MPS does not support. Without this the clip samples all
the way through and then dies on the LAST node with
`Cannot convert a MPS Tensor to float64 dtype`. In `rope_inv_freqs`, compute on the CPU and move
the fp32 result to the device — the function already returns float32, so nothing else changes.

## Speed

Measured on an M5: **33 frames, 640×384 → 1280×768, 4.4 min.** Per frame-megapixel that is about
6× slower than an RTX 5090. A 10s clip is roughly 34 min at 2×, 19 min at 1.5×. Quality holds:
PSNR 37.4 dB against the same source rendered on the int8 path.

## Prep your clip

Audio track required (A/V encode jointly), length must be **8n+1 frames**, and both output
dimensions must divide by **64**. `redetail.py --cached-cond` does all of it for you.
"""
# BY ID, not by content. Matching on "START HERE" hit the ERRORS panel first, because it
# cross-references that section — so the intro panel kept telling Mac users to check for Blackwell
# and download a 26GB encoder, while an unrelated panel was overwritten.
_intro = nodes.get(5526)
if not _intro or _intro.get("type") != "MarkdownNote":
    sys.exit("node 5526 is not the intro MarkdownNote any more — do not ship stale instructions")
_intro["widgets_values"] = [MAC_NOTE]
_stale = [n["id"] for n in d["nodes"]
          if n.get("type") == "MarkdownNote"
          and any(k in str(n.get("widgets_values")) for k in ("Blackwell", "int8_convrot"))]
if _stale:
    sys.exit(f"panels {_stale} still describe the Blackwell path — fix them before shipping")

json.dump(d, open(OUT, "w"), indent=1)
print(f"wrote {os.path.relpath(OUT, HERE)}")
print(f"  transformer : {GGUF} (UnetLoaderGGUF)")
print(f"  conditioning: {POS} / {NEG} (LoadConditioning, no text encoder)")
print(f"  removed     : PreviewAny x{len(prev)}, {len(orphan)} orphaned links")
