#!/usr/bin/env python3
"""ReDetail — generative video upscaling with the LTX-2.5 IC-LoRA Pixel Spatial Upscaler.

    python3 redetail.py clip.mp4 --scale 1.5

Point it at any running ComfyUI (default http://127.0.0.1:8188). It works out the target
resolution, splits the clip on its own cuts, renders each piece, reassembles, and puts the
original audio back. One command, no browser automation, no manual graph surgery.

WHY A SCRIPT AND NOT JUST THE WORKFLOW: the .json in workflows/ is the whole render and you can
drag it straight into ComfyUI. But a real clip needs more than one render — the dimensions have to
land on a 64-pixel grid, long clips have to be split to fit VRAM, the splits have to land on the
source's own cuts, and each piece has to be an exact frame count or the result silently drifts out
of sync with the audio. That arithmetic is what this script does for you.

READ THIS BEFORE USING IT ON ANYTHING THAT MATTERS
--------------------------------------------------
This model SYNTHESIZES detail. It does not recover detail that was in the original. On faces it
will invent skin texture that was never there — measured on one persona, it consistently added
freckles. Judge the result by comparing the FACE against your source, not by how sharp it looks.
It is a creative step, not a restoration tool. Model licence: LTX-2-community-license.

REQUIREMENTS
    ComfyUI with Lightricks/ComfyUI-LTXVideo, kornia==0.7.4, comfy-kitchen>=0.2.26
    ffmpeg + ffprobe on PATH
    Models: see README.md
"""
import argparse, json, os, re, subprocess, sys, time, urllib.parse, urllib.request, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.join(HERE, "workflows", "ltx25_upscale_API.json")

# Node ids in the shipped API workflow. They are stable because the file is shipped resolved.
N_FRAME, N_VIDEO, N_SAVE = "2004", "5001", "4852"
N_REF_RESIZE, N_CANVAS = "5548:5026", "9002:3059"

# Output frame-megapixels per chunk == the VRAM dial. Measured on a 96GB card:
#   243f x 1.08MP -> 52GB | 243f x 2.43MP -> 65GB | 243f x 4.33MP -> 80.5GB
# i.e. roughly 42.5GB fixed + 0.036GB per frame-megapixel. 850 is comfortable on 96GB; drop it
# with --budget on a smaller card (24GB: try ~150, 48GB: try ~350) or if you hit an OOM.
FMP_BUDGET = 850.0

# No chunk shorter than this. Tiny chunks pay a full setup cost for a fraction of a second and
# starve the model of temporal context — splitting at every detected cut once produced ten chunks
# for a 14s clip, some of them 9 frames long.
MIN_SEC = 2.0


def run(cmd, timeout=3600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout


def probe(path):
    d = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_frames", "-show_entries",
                        "format=duration",
                        "-of", "json", path]))
    s = d["streams"][0]
    n, _, den = s["r_frame_rate"].partition("/")
    fps = int(n) / int(den or 1)
    # The video stream's FRAME COUNT is authoritative. The container duration includes the audio
    # track, which is often longer — a 17-frame clip reported 1.0s and asked for 24 frames.
    # ffprobe returns the STRING "N/A" when a container stores no frame count, and "N/A" is truthy
    # — so `int(x or 0)` raised ValueError and crashed before ever reaching the counting fallback
    # below. That is precisely the container the README tells people this handles.
    _raw = str(s.get("nb_frames") or "")
    nb = int(_raw) if _raw.isdigit() else 0
    if not nb:
        # NEVER fall back to the container duration — it includes the audio track, which is often
        # longer, and a 17-frame clip then plans for 24. Decode and count instead; slower, exact.
        # Same guarded parse: nb_read_frames can also come back "N/A" on a stream ffprobe cannot
        # decode, and an unguarded int() would turn "no frame count" into a crash.
        _c = run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                  "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path]).strip()
        nb = int(_c) if _c.isdigit() else 0
    if not nb:
        sys.exit(f"Could not determine a frame count for {path}")
    return int(s["width"]), int(s["height"]), fps, nb / fps


def nframes(path):
    n = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=nb_frames", "-of", "csv=p=0", path]).strip()
    if not n.isdigit():                       # container did not store it — count for real
        n = run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path]).strip()
    return int(n) if n.isdigit() else 0


# --------------------------------------------------------------------------------------------
# Geometry. Both of these exist because the guide node splits the latent into 2x2 patches, so the
# /32 latent has to be EVEN on both axes -> pixel dimensions must be divisible by 64, not 32.
# An 800x1440 clip fails with "Error while processing rearrange-reduction pattern".
# --------------------------------------------------------------------------------------------

def fit_source(w, h, max_crop=0.03):
    """Get the source onto the /64 grid, losing as little picture as possible.

    Cropping is simplest, but only cheap when the remainder is small: a 432x768 clip crops to
    384x768, throwing away 11% of the width. 432x768 is exactly 9:16, and 9:16 has an exact /64
    solution at 576x1024 — so it resizes instead and keeps the whole frame. 1280x714 is the
    opposite: no exact /64 aspect exists, and the crop costs 1.4%. Returns (mode, w, h).
    """
    cw, ch = w // 64 * 64, h // 64 * 64
    if (cw, ch) == (w, h):
        return "none", w, h
    if max(1 - cw / w, 1 - ch / h) <= max_crop:
        return "crop", cw, ch
    ar, best = w / h, None
    for b in range(1, 65):
        fh = b * 64
        fw = max(64, round(fh * ar / 64) * 64)
        if abs((fw / fh) / ar - 1) > 0.005:      # exact aspect only — a 3.7% stretch is worse
            continue                              # than resampling, never trade aspect for scale
        # Stay NEAR the source. Without this the search happily returned the first exact-aspect
        # /64 size at any scale: a 320x330 clip resized to 1792x1856, a 5.6x enlargement, before
        # the requested upscale even started. Cropping a few percent beats inventing 30x the
        # pixels, so anything outside ±35% is not a candidate.
        if not 0.65 <= fh / h <= 2.0:
            continue
        d = abs(fh / h - 1)
        if best is None or d < best[0]:
            best = (d, fw, fh)
    return ("resize", best[1], best[2]) if best else ("crop", cw, ch)


def solve_dims(w, h, scale):
    """Nearest /64 target to the requested scale that also holds the source aspect.

    The target is NOT scale x source. Exact 1.5x of 1920x1088 is 2880x1632, and 1632 is not /64,
    so that exact target is invalid and the answer is 2816x1600: 0.27% aspect error, invisible.
    Plenty of common sizes have no exact 1.5x on the grid; 2x is exact far more often, because
    doubling a /64 number stays /64. Aspect error is
    weighted ~8x scale error because a stretched face is obvious and 1.47x instead of 1.50x is not.
    """
    ar, best = w / h, None
    lo = max(64, int(scale * h * 0.85) // 64 * 64)
    for th in range(lo, int(scale * h * 1.15) // 64 * 64 + 65, 64):
        tw = max(64, round(th * ar / 64) * 64)
        aerr = abs((tw / th) / ar - 1)
        if aerr > 0.02:
            continue
        serr = ((tw / w) + (th / h)) / 2 / scale - 1
        score = aerr * 8 + abs(serr) * (1.3 if serr > 0 else 1.0)   # mild penalty for overshoot,
        if best is None or score < best[0]:                          # which costs unasked-for VRAM
            best = (score, tw, th, aerr, serr)
    if best is None:
        sys.exit(f"No /64 target within 2% of aspect {ar:.4f} near {scale}x. Try another scale.")
    return best[1], best[2], best[3], best[4]



def target_for(orig_w, orig_h, fit_w, fit_h, scale):
    """Output size = `scale` x the ORIGINAL clip, rendered at the FITTED source's aspect.

    TWO THINGS HAVE TO COME FROM DIFFERENT PLACES, which is why this is not just solve_dims().

    SCALE references the ORIGINAL clip. fit_source may already have enlarged a non-/64 source to
    reach the grid — 352x608 becomes 704x1216, itself exactly 2x — and applying the requested
    scale on top of that made `--scale 2.0` render 1408x2432, a silent 4x. Users mean "2x of my
    clip", so the requested scale is measured against what they handed in.

    ASPECT references the FITTED source. When fit_source CROPS (1280x714 -> 1280x704) the framing
    genuinely changes, and holding the original's aspect against a cropped input would stretch
    the picture.
    """
    eff = (orig_h * scale) / fit_h
    return solve_dims(fit_w, fit_h, eff)

def scene_cuts(path, threshold=0.3):
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-an", "-vf",
                        f"select='gt(scene,{threshold})',metadata=print", "-f", "null", "-"],
                       capture_output=True, text=True)
    return sorted(float(m) for m in re.findall(r"pts_time:([\d.]+)", r.stderr))


def segments(cuts, end, max_sec, fps):
    """Contiguous chunks, cut boundaries first, EVERY LENGTH 8n+1 FRAMES.

    Two things this has to get right, both learned the hard way on the SuperTrips trailer:

    1. Chunks must TILE the source contiguously in frames. Anything else desyncs the picture
       against the original audio, and a desynced file still plays perfectly and reports a clean
       duration — nothing downstream catches it.

    2. The model quantizes output length DOWN to the nearest 8n+1 (measured: 139->137, 132->129,
       108->105, 54->49). Feed it 132 frames and 3 frames vanish; over 17 chunks that is ~3% of
       the runtime and the picture drifts progressively EARLY. Feed a length that is already 8n+1
       and output length == input length.

    Boundaries therefore land on the nearest 8n+1 to each cut rather than exactly on it — a drift
    of at most 4 frames (0.17s), which just means a couple of frames of the next shot ride along
    in the previous chunk. Content stays at the correct timestamp, which is the thing that matters.
    """
    total = round(end * fps)
    maxf = max(9, int(max_sec * fps))
    cutf = [round(c * fps) for c in cuts if 0.1 < c < end - 0.1]

    # HOW MANY CHUNKS: purely the VRAM constraint. As FEW as possible, because every extra chunk
    # is another setup cost and another texture-mismatch seam.
    #
    # The model's 8n+1 grid is NOT handled here. Forcing the chunk COUNT to satisfy it (k must be
    # congruent to total mod 8, for all-8n+1 lengths summing to total) is arithmetically valid but
    # catastrophic in practice: a 184-frame clip has 184 % 8 == 0, so k would have to be a multiple
    # of 8 — turning a comfortable ONE-chunk render into EIGHT nine-frame renders. Instead the
    # caller pads each chunk's READ to the next 8n+1 using real frames from past the boundary and
    # trims the output back, which leaves chunk count free.
    # ...and never more chunks than MIN_SEC allows. A 9-frame chunk is pure waste: full setup cost
    # for 0.4s, and almost no temporal context for the model to work from.
    minf = max(9, int(MIN_SEC * fps))
    k_vram = max(1, -(-total // maxf))        # hard ceiling: never exceed the VRAM budget
    k_min = max(1, total // minf)             # soft floor: avoid pointless tiny chunks
    if k_vram > k_min and total > minf:
        # The budget demands smaller chunks than MIN_SEC allows. The memory limit wins — silently
        # exceeding it is an OOM, whereas a short chunk is only inefficient.
        print(f"  note: --budget forces chunks under {MIN_SEC}s "
              f"({total/k_vram/fps:.1f}s each). Memory limit wins.")
    k = k_vram

    # WHERE THE BOUNDARIES GO. Evenly spaced, then SNAPPED onto a nearby cut when one exists.
    # Deliberately NOT a boundary at every cut: chunking on cuts exists to hide texture mismatch
    # BETWEEN chunks, and there is no mismatch INSIDE a chunk — so merging consecutive short shots
    # is strictly better. Splitting at every cut gave one fast-cut clip ten chunks for 14.4s, some
    # of them 9 frames long, each paying full setup cost and giving the model almost no temporal
    # context to work with.
    bounds = [0]
    for i in range(1, k):
        t = round(i * total / k)
        # The distance test alone is NOT enough. It bounds how far the boundary moves from the
        # ideal, not how long the resulting chunk is — a cut 30% of maxf past an already-full
        # chunk still snapped, producing a 130-frame chunk against a 100-frame cap and OOMing
        # exactly where the budget promised it would not. Every candidate must also leave this
        # chunk within maxf AND leave enough room for the remaining chunks to fit theirs.
        remaining = k - i
        near = [c for c in cutf
                if abs(c - t) <= maxf * 0.35 and c >= bounds[-1] + minf and c <= total - minf
                and c - bounds[-1] <= maxf                 # this chunk fits the budget
                and total - c <= maxf * remaining]         # and so can everything after it
        bounds.append(min(near, key=lambda c: abs(c - t)) if near else t)
    bounds.append(total)
    # Natural lengths — they tile `total` exactly by construction.
    return [(a / fps, b / fps, b - a) for a, b in zip(bounds, bounds[1:])]


def render_len(n):
    """Smallest 8n+1 >= n. The model quantizes output length DOWN to 8n+1 (measured 139->137,
    132->129, 108->105, 54->49), so anything not already on the grid comes back short."""
    return n if n % 8 == 1 else ((n - 1) // 8 + 1) * 8 + 1


# --------------------------------------------------------------------------------------------
# ComfyUI over plain HTTP. No browser, no Playwright: the shipped workflow is already in API
# format with its subgraphs resolved, so it can be POSTed as-is.
# --------------------------------------------------------------------------------------------

class Comfy:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.cid = str(uuid.uuid4())
        try:
            urllib.request.urlopen(f"{self.url}/system_stats", timeout=10)
        except Exception:
            sys.exit(f"No ComfyUI at {self.url} — start it, or pass --comfy http://host:port")

    def upload(self, path):
        """POST to /upload/image. It handles video too; that is just the endpoint's name."""
        name = os.path.basename(path)
        bnd = "----redetail"
        body = b"".join([
            f'--{bnd}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode(),
            open(path, "rb").read(),
            f"\r\n--{bnd}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{bnd}--\r\n".encode()])
        req = urllib.request.Request(f"{self.url}/upload/image", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={bnd}"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read())["name"]

    def submit(self, prompt):
        req = urllib.request.Request(f"{self.url}/prompt",
                                     data=json.dumps({"prompt": prompt, "client_id": self.cid}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=120).read())["prompt_id"]
        except urllib.error.HTTPError as e:
            sys.exit(f"ComfyUI rejected the job:\n{e.read().decode()[:1500]}")

    def wait(self, pid, every=5):
        while True:
            time.sleep(every)
            h = json.loads(urllib.request.urlopen(f"{self.url}/history/{pid}", timeout=30).read())
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") != "success":
                    return None
                for out in h[pid].get("outputs", {}).values():
                    for k in ("video", "videos", "gifs", "images"):
                        if out.get(k):
                            return out[k][0]
                return None

    def download(self, item, dst):
        q = urllib.parse.urlencode({"filename": item["filename"],
                                    "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
        with urllib.request.urlopen(f"{self.url}/view?{q}", timeout=1800) as r, open(dst, "wb") as f:
            f.write(r.read())
        return os.path.getsize(dst)


# --------------------------------------------------------------------------------------------
# --setup: turn the error table into executable diagnostics.
# Everything here is checkable through ComfyUI's own HTTP API, including on a remote machine you
# cannot ls: each loader's enum in /object_info lists the filenames actually present in that model
# folder, and /system_stats reports the GPU.
# --------------------------------------------------------------------------------------------

NEEDED_FILES = [
    ("diffusion_models", "UNETLoader", "unet_name",
     "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "21.5GB",
     "Lightricks/LTX-2.5"),
    ("text_encoders", "CLIPLoader", "clip_name",
     "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "15.4GB",
     "Lightricks/LTX-2.5"),
    ("vae", "VAELoader", "vae_name", "ltx-2.5-video-vae-bf16.safetensors", "1.5GB",
     "Lightricks/LTX-2.5"),
    ("vae", "VAELoader", "vae_name", "ltx-2.5-audio-vae-bf16.safetensors", "0.4GB",
     "Lightricks/LTX-2.5"),
    ("loras", "LoraLoaderModelOnly", "lora_name",
     "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors", "0.3GB",
     "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"),
]
NEEDED_NODES = ["LTXAddVideoICLoRAGuide", "LTXICLoRALoaderModelOnly", "EmptyLTXVLatentVideo",
                "VAEDecodeTiled"]
# Blackwell is sm_120 / sm_100. int8_convrot needs those tensor layouts; on anything older the
# weights do not load AT ALL, which is an architecture limit and not a VRAM one.
BLACKWELL = ("5090", "5080", "5070", "RTX PRO 6000", "B200", "B300", "GB200")


def use_cached_cond(pr, name, out_node="4852"):
    """Replace the two CLIPTextEncode nodes with LoadConditioning and drop the encoder.

    Both prompt boxes are empty, so this conditioning is a constant and caching it changes nothing
    about the render. Verified: bit-identical output, PSNR inf against the encoder path.

    The two ComfySwitchNodes in between are bypassed rather than left alone. Their selector is
    StringContains('', 'ltxv_'), which is always False, so they always pick on_false — and the
    GemmaAPITextEncode nodes on their dead branch name a checkpoint that a GGUF-only install does
    not have. ComfyUI VALIDATES disabled branches, so those must go, not just be ignored.

    Nodes are then pruned by REACHABILITY from the video output. Pruning by eye is what produced a
    400 the first time; walking the graph from its output cannot pick the wrong set.
    """
    pr["5014:2483"] = {"class_type": "LoadConditioning", "inputs": {"name": f"{name}_pos"}}
    pr["5014:2612"] = {"class_type": "LoadConditioning", "inputs": {"name": f"{name}_neg"}}
    if "9002:9005" in pr:
        pr["9002:9005"]["inputs"]["positive"] = ["5014:1241", 0]
        pr["9002:9005"]["inputs"]["negative"] = ["5014:1241", 1]
    keep, stack = set(), [out_node]
    while stack:
        n = stack.pop()
        if n in keep or n not in pr:
            continue
        keep.add(n)
        for b in pr[n].get("inputs", {}).values():
            if isinstance(b, list) and b:
                stack.append(b[0])
    for dead in [k for k in pr if k not in keep]:
        pr.pop(dead)
    return pr


def bootstrap_cond(url, encoder, name, clip_device=None):
    """Load the encoder once, encode the empty prompt, save it, unload. Run this a single time."""
    if not encoder:
        print("--bootstrap-cond needs --encoder <filename> (the one time it IS loaded).")
        return False
    comfy = Comfy(url)
    oi = json.loads(urllib.request.urlopen(f"{url}/object_info", timeout=60).read())
    if "SaveConditioning" not in oi:
        print("SaveConditioning is missing. Put tools/cond_cache_node.py in ComfyUI/custom_nodes/ "
              "and RESTART ComfyUI, then re-run.")
        return False
    # A GGUF encoder needs CLIPLoaderGGUF; a safetensors one needs CLIPLoader. Pick by extension
    # rather than making the user say which, since the filename already states it.
    gguf_enc = encoder.lower().endswith(".gguf")
    loader = "CLIPLoaderGGUF" if gguf_enc else "CLIPLoader"
    if loader not in oi:
        print(f"{loader} is missing (needed for {encoder}).")
        return False
    ins = {"clip_name": encoder, "type": "ltxv"}
    if not gguf_enc and clip_device:
        ins["device"] = clip_device
    pr = {"1": {"class_type": loader, "inputs": ins},
          "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 0]}},
          "3": {"class_type": "SaveConditioning",
                "inputs": {"conditioning": ["2", 0], "name": f"{name}_pos"}},
          "4": {"class_type": "SaveConditioning",
                "inputs": {"conditioning": ["2", 0], "name": f"{name}_neg"}}}
    print(f"encoding the empty prompt with {encoder} (this is the only time it loads) ...")
    if comfy.wait(comfy.submit(pr)) is None:
        pass                      # wait() returns the output item; this graph has no video output
    print(f"saved '{name}_pos' and '{name}_neg' in ComfyUI/output/cond_cache/.\n"
          f"From now on: --cached-cond {name}   (and drop --encoder)")
    return True


def doctor(url, gguf=None, encoder=None, clip_device=None, cached_cond=None):
    """`--setup` must check the install the user is actually going to RUN.

    It used to take only the URL, so it always demanded the int8_convrot text encoder — which is
    Blackwell-only. A correctly configured 4090 (GGUF transformer + bf16 encoder on CPU) was told
    it had a missing model, and a GGUF install carrying an unusable int8 encoder was told it was
    fine. Both wrong, in opposite directions. The variant flags come in here now.
    """
    ok = True

    def say(good, msg, fix=""):
        nonlocal ok
        print(f"  {'PASS' if good else 'FAIL'}  {msg}")
        if not good:
            ok = False
            if fix:
                print(f"        -> {fix}")

    print("\nReDetail setup check\n" + "-" * 60)

    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], capture_output=True, check=True)
            say(True, f"{tool} on PATH")
        except Exception:
            say(False, f"{tool} NOT found", "install ffmpeg and put it on PATH")

    try:
        stats = json.loads(urllib.request.urlopen(f"{url}/system_stats", timeout=10).read())
    except Exception as e:
        say(False, f"ComfyUI unreachable at {url} ({e})",
            "start ComfyUI, or pass --comfy http://host:port")
        print("-" * 60 + "\nCannot check further without ComfyUI.\n")
        return False
    say(True, f"ComfyUI reachable at {url}")

    gpu = ""
    for d in stats.get("devices", []):
        gpu = d.get("name", "")
        vram = d.get("vram_total", 0) / 1e9
        say(True, f"GPU: {gpu}  ({vram:.0f}GB)")

    try:
        oi = json.loads(urllib.request.urlopen(f"{url}/object_info", timeout=60).read())
    except Exception as e:
        say(False, f"could not read /object_info ({e})")
        return False

    for n in NEEDED_NODES:
        say(n in oi, f"node {n}",
            "install github.com/Lightricks/ComfyUI-LTXVideo and RESTART ComfyUI. If it is already "
            "installed, its import failed - that is only a WARNING in the boot log, so read the "
            "log. Usual cause: kornia 0.8.x (pin kornia==0.7.4) or comfy-kitchen < 0.2.26, "
            "installed into a DIFFERENT python than ComfyUI runs from.")

    def enum_for(cls, field):
        try:
            v = oi[cls]["input"]["required"][field][0]
            return v if isinstance(v, list) else []
        except Exception:
            return []

    # NOT named `gguf` — that is the parameter holding the user's chosen filename,
    # and shadowing it made the file check demand a model literally called "True".
    gguf_node = "UnetLoaderGGUF" in oi
    present_unet = enum_for("UNETLoader", "unet_name")
    gguf_files = enum_for("UnetLoaderGGUF", "unet_name") if gguf_node else []
    int8_installed = any("int8-convrot" in f for f in present_unet)
    is_blackwell = any(b.lower() in gpu.lower() for b in BLACKWELL)

    # Substitute the two files the low-spec path legitimately replaces, so the check matches the
    # command the user will actually run rather than the default int8 set.
    wanted = []
    for folder, cls, field, fname, size, repo in NEEDED_FILES:
        if cached_cond and folder == "text_encoders":
            continue          # no encoder is loaded at all, so requiring one is just wrong
        if gguf and folder == "diffusion_models":
            wanted.append(("unet", "UnetLoaderGGUF", "unet_name", gguf, "~13GB", repo))
        elif encoder and folder == "text_encoders":
            wanted.append((folder, cls, field, encoder, "26GB (bf16)", repo))
        else:
            wanted.append((folder, cls, field, fname, size, repo))

    # If the user asked for a GGUF, the node pack that loads it must be present — otherwise the
    # file check below reports a MISSING MODEL and sends them to the Lightricks repo, when the real
    # problem is an uninstalled node pack and a different download entirely.
    if gguf:
        say(gguf_node, "ComfyUI-GGUF node pack (UnetLoaderGGUF)",
            "install github.com/city96/ComfyUI-GGUF and RESTART ComfyUI. The GGUF transformer "
            "itself comes from https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF, NOT the "
            "Lightricks repo.")

    for folder, cls, field, fname, size, repo in wanted:
        have = fname in enum_for(cls, field)
        # An installed GGUF is NOT a substitute for the default transformer unless it was actually
        # selected: without --gguf the run never swaps in UnetLoaderGGUF, so passing here told
        # people their install was fine and then failed at render time on the missing int8 file.
        if not have and folder == "diffusion_models" and gguf_files and not gguf:
            say(False, f"{folder}/{fname}  ({size})",
                f"a GGUF transformer is installed ({', '.join(gguf_files[:2])}) but you did not "
                f"select it — re-run with --gguf <filename>, or download the int8 transformer.")
            continue
        say(have, f"{folder}/{fname}  ({size})",
            f"download from https://huggingface.co/{repo} into ComfyUI/models/{folder}/ . "
            f"The repo is GATED: if you get 403 while the README loads fine, you have not "
            f"accepted its licence at https://huggingface.co/{repo}")

    # Only a complaint if the run would actually TOUCH the int8 weights. With --gguf and a bf16
    # --encoder selected, leftover int8 files on disk are irrelevant.
    # Derive this from what the user SELECTED, not from which files happen to be on disk.
    # `int8_installed` only inspected the transformer folder, so `--gguf` without `--encoder` on a
    # non-Blackwell card reached the PASS branch — while at runtime the encoder stays int8 and the
    # render cannot load. Either unselected component is enough to make this an int8 run.
    # --cached-cond means no encoder is loaded AT ALL, so "you selected no encoder" stops being
    # evidence of an int8 run. Without this, the correct Mac setup gets told it is broken.
    using_int8 = (gguf is None) or (encoder is None and not cached_cond)
    if cached_cond:
        say("LoadConditioning" in oi, f"cond cache node (for --cached-cond {cached_cond})",
            "copy tools/cond_cache_node.py into ComfyUI/custom_nodes/ and RESTART ComfyUI")
        say(True, f"encoder will NOT be loaded — using cached '{cached_cond}_pos/_neg'",
            "")
        if "mps" in gpu.lower() or "apple" in gpu.lower():
            # Found the hard way: everything sampled fine on an M5 and died on the last node.
            say(True, "Apple Silicon detected — see the MPS note below", "")
            print("        NOTE  comfy/ldm/lightricks/vae/na_diffusion_decoder.py builds RoPE\n"
                  "              frequencies in float64, which MPS does not support, so\n"
                  "              VAEDecodeTiled raises 'Cannot convert a MPS Tensor to float64'.\n"
                  "              Compute it on CPU and move the fp32 result to the device.\n"
                  "              The function already returns float32, so nothing else changes.")
    if not is_blackwell and using_int8 and (int8_installed or encoder is None):
        say(False, f"int8_convrot weights present but '{gpu}' is not Blackwell",
            "int8_convrot needs Blackwell tensor layouts - these weights will NOT load on this "
            "card at any resolution. Use the GGUF path: install github.com/city96/ComfyUI-GGUF, "
            "put LTX-2.5-Distilled-Q4_K_M.gguf in models/unet/, and run with "
            "--gguf LTX-2.5-Distilled-Q4_K_M.gguf "
            "--encoder gemma4-12b-with-proj-ltx-2.5-bf16.safetensors --clip-device cpu "
            "--budget 150 --decode-tile 256")
    elif not is_blackwell and gguf_node:
        say(True, f"'{gpu}' is not Blackwell, and ComfyUI-GGUF is installed - use --gguf")

    print("-" * 60)
    print("All checks passed. Try the smoke test:\n"
          "  ffmpeg -i yourclip.mp4 -frames:v 17 -c:a copy smoke.mp4\n"
          "  python3 redetail.py smoke.mp4 --scale 1.5\n" if ok else
          "Fix the FAIL lines above, then run --setup again.\n")
    return ok


def main():
    p = argparse.ArgumentParser(description="Generative video upscaling with LTX-2.5.")
    p.add_argument("input", nargs="?",
                   help="video to upscale (not needed with --setup)")
    p.add_argument("--setup", action="store_true",
                   help="check your install and print exactly what is missing, then exit")
    p.add_argument("--scale", type=float, default=1.5,
                   help="1.0 re-detail at same size, 1.5 the sweet spot, 2.0 heaviest (default 1.5)")
    p.add_argument("--comfy", default="http://127.0.0.1:8188")
    p.add_argument("--out", default=None, help="output file (default <name>_redetail.mp4)")
    p.add_argument("--budget", type=float, default=FMP_BUDGET,
                   help="VRAM dial: output frame-megapixels per chunk. Lower it if you OOM.")
    p.add_argument("--audio", choices=["original", "generated"], default="original",
                   help="'original' re-muxes your audio (default). The model regenerates a track "
                        "per chunk, which you do not want on finished material.")
    p.add_argument("--keep-chunks", action="store_true")
    # ---- low-spec / non-Blackwell -----------------------------------------------------------
    # int8_convrot needs Blackwell tensor layouts. That is what rules out a 4090 — not its VRAM.
    # A GGUF transformer dequantizes in the compute kernel and runs on any architecture. Measured
    # on an RTX 4090: 8 sampling steps in 70s, 21.8GB peak of 24.5GB.
    p.add_argument("--gguf", default=None,
                   help="GGUF transformer in models/unet (e.g. LTX-2.5-Distilled-Q4_K_M.gguf). "
                        "Required on non-Blackwell cards; also 6.4GB lighter and ~16%% faster.")
    p.add_argument("--encoder", default=None,
                   help="text encoder filename. The int8_convrot encoder is ALSO Blackwell-only, "
                        "so on other cards use gemma4-12b-with-proj-ltx-2.5-bf16.safetensors "
                        "together with --clip-device cpu (it is 26GB and will not fit beside the "
                        "transformer).")
    p.add_argument("--clip-device", default=None, choices=["cpu", "default"],
                   help="run the text encoder on CPU. It encodes once then idles through every "
                        "sampling step, so this buys VRAM for a one-off CPU cost.")
    p.add_argument("--decode-tile", type=int, default=None,
                   help="VAEDecodeTiled tile_size. DECODE is a SEPARATE memory spike from "
                        "sampling — a 4090 sampled 8/8 steps cleanly and then OOMed here. "
                        "256 works on 24GB.")
    p.add_argument("--decode-temporal", type=int, default=None,
                   help="VAEDecodeTiled temporal_size (default 128). Lower with --decode-tile.")
    # BOTH prompt boxes in this graph are empty, so the text conditioning is a constant. Encode it
    # once, cache ~26KB, and no render ever loads the encoder again. Measured on an RTX 5090:
    # peak 30.4GB -> 24.8GB and 29.2s -> 24.0s, output bit-identical (PSNR inf).
    p.add_argument("--cached-cond", nargs="?", const="redetail", default=None, metavar="NAME",
                   help="skip the text encoder entirely and load a cached conditioning written by "
                        "--bootstrap-cond. This is what makes the Mac path viable: the only "
                        "non-Blackwell encoder is 26GB bf16, and dropping it frees that memory "
                        "for the transformer.")
    p.add_argument("--bootstrap-cond", nargs="?", const="redetail", default=None, metavar="NAME",
                   help="load the encoder ONCE, encode the empty prompt, save it under NAME, and "
                        "exit. Needs --encoder. Run this a single time, then use --cached-cond.")
    a = p.parse_args()

    if a.bootstrap_cond:
        sys.exit(0 if bootstrap_cond(a.comfy.rstrip("/"), a.encoder, a.bootstrap_cond,
                                     a.clip_device) else 1)
    if a.setup:
        sys.exit(0 if doctor(a.comfy.rstrip("/"), a.gguf, a.encoder, a.clip_device,
                             a.cached_cond) else 1)
    if not a.input:
        sys.exit("Give me a video, or run with --setup to check your install.")
    src = os.path.abspath(a.input)
    if not os.path.exists(src):
        sys.exit(f"No such file: {src}")
    out = a.out or os.path.splitext(src)[0] + "_redetail.mp4"
    work = os.path.splitext(out)[0] + "_chunks"
    os.makedirs(work, exist_ok=True)
    # Cleanup used to empty this directory unconditionally. `<out>_chunks` is derived from --out,
    # so a user pointing --out at a name that collides with an existing folder lost its contents.
    #
    # A NAME SNAPSHOT IS NOT ENOUGH, which is what the first fix used. Two fixed scratch names
    # (concat.txt, joined.mp4) were written with `w`/`-y`, so a pre-existing file with either name
    # was clobbered and then KEPT because its name was in the snapshot — destroying data while
    # appearing to protect it. And anything created in here by another process mid-run was deleted
    # as though we had made it. Track the exact paths this run writes; touch nothing else.
    RID = uuid.uuid4().hex[:8]
    created = []

    def mine(path):
        """Register a scratch path as ours, so cleanup may delete it."""
        created.append(path)
        return path

    w, h, fps, dur = probe(src)
    # `-c:a aac` TRANSCODES an existing stream; it does not create one. A silent source therefore
    # still reaches VAEEncodeAudio with audio=None and the graph errors. AI-generated clips — the
    # main input for a generative re-detailer — are usually silent, so this is the common case.
    has_audio = bool(run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "csv=p=0", src]).strip())
    fit, cw, ch = fit_source(w, h)
    vf = f"scale={cw}:{ch}:flags=lanczos" if fit == "resize" else f"crop={cw}:{ch}"
    tw, th, aerr, serr = target_for(w, h, cw, ch, a.scale)
    max_sec = a.budget / (tw * th / 1e6) / fps
    segs = segments(scene_cuts(src), dur, max_sec, fps)
    src_frames = round(dur * fps)

    print(f"\n  {os.path.basename(src)}  {w}x{h}  {dur:.1f}s  {fps:g}fps")
    if fit == "resize":
        print(f"  source -> {cw}x{ch} (resized; exact aspect, no framing lost)")
    elif fit == "crop":
        print(f"  source -> {cw}x{ch} (cropped {max(1-cw/w, 1-ch/h)*100:.1f}% to reach the /64 grid)")
    print(f"  target -> {tw}x{th}  ({tw/cw:.2f}x, {serr*100:+.1f}% off {a.scale}x, "
          f"aspect error {aerr*100:.2f}%)")
    print(f"  {len(segs)} chunk(s), {sum(L for _, _, L in segs)}/{src_frames} frames, "
          f"cap {max_sec:.1f}s at {a.budget:g} frame-MP\n")

    base = json.load(open(WF))
    comfy = Comfy(a.comfy)
    parts = []
    for i, (s, _e, L) in enumerate(segs):
        tag = f"rd_{abs(hash((src, s, L, tw, th))) % 10**8:08d}_{i:03d}"
        seg, frm = mine(f"{work}/{tag}.mp4"), mine(f"{work}/{tag}.png")
        # -ss AFTER -i and NO -to. As an INPUT option -ss is a keyframe seek: it silently starts
        # early and runs long (a 4.00s request came back 7.92s). -to is an exclusive end on a
        # float timestamp and can clip the last frame. -frames:v alone pins the count. Seeking
        # half a frame early keeps float error from deciding whether frame N survives.
        # NEVER -an: this graph encodes audio and video together and errors on a silent input.
        # Read RLEN frames — the next 8n+1 at or above this chunk's natural length. The extra
        # frames are REAL content from past the boundary, so the model gets genuine context rather
        # than duplicated padding, and the output is trimmed back to L after download. Forcing the
        # chunk COUNT to satisfy 8n+1 instead is arithmetically valid but turns a comfortable
        # one-chunk render into eight nine-frame ones.
        rlen = render_len(L)
        pad = f",tpad=stop_mode=clone:stop_duration={rlen/fps:.3f}" if i == len(segs) - 1 else ""
        cut = ["ffmpeg", "-y", "-v", "error", "-i", src]
        if not has_audio:
            cut += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        cut += ["-ss", f"{max(0.0, s - 0.5/fps):.6f}", "-frames:v", str(rlen), "-vf", vf + pad,
                "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k"]
        if not has_audio:
            cut += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
        subprocess.run(cut + [seg], check=True)
        got = nframes(seg)
        if got != rlen:
            sys.exit(f"Chunk {i} came out {got} frames instead of {rlen}. Refusing to continue — "
                     f"the result would be out of sync with the audio.")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", seg, "-frames:v", "1", "-update", "1",
                        frm], check=True)

        pr = json.loads(json.dumps(base))
        pr[N_VIDEO]["inputs"]["file"] = comfy.upload(seg)
        pr[N_FRAME]["inputs"]["image"] = comfy.upload(frm)   # ships EMPTY; an empty filename makes
        pr[N_SAVE]["inputs"]["filename_prefix"] = tag        # ComfyUI open the input DIRECTORY
        pr[N_REF_RESIZE]["inputs"]["resize_type.shorter_size"] = min(cw, ch)
        # THE CANVAS. In the stock graph these are LINKED from GetImageSize on the resized source,
        # so it renders 1:1 no matter what scale you asked for — the single most confusing failure
        # this tool has. Literal ints sever that link.
        if a.gguf:
            # Same output socket, so nothing downstream needs rewiring.
            pr["5004:5602"] = {"class_type": "UnetLoaderGGUF",
                               "inputs": {"unet_name": a.gguf}}
            # The DISABLED prompt-enhancer branch is still VALIDATED, and its ckpt_name must name a
            # file that exists — on a GGUF-only install diffusion_models/ is empty and the whole
            # prompt is rejected against an empty list. Any real filename there will do.
            for _n in ("5014:5504", "5014:5505"):
                if _n in pr and "ckpt_name" in pr[_n]["inputs"]:
                    pr[_n]["inputs"]["ckpt_name"] = a.gguf
        for _n in ("5004:5604", "5004:5605"):
            if _n in pr:
                if a.encoder:
                    pr[_n]["inputs"]["clip_name"] = a.encoder
                if a.clip_device and "device" in pr[_n]["inputs"]:
                    pr[_n]["inputs"]["device"] = a.clip_device
        # LAST, because it prunes: any node this drops must already have been patched above, and
        # anything it keeps still carries those edits.
        if a.cached_cond:
            pr = use_cached_cond(pr, a.cached_cond, N_SAVE)
        if a.decode_tile or a.decode_temporal:
            _d = pr.get("5518:5538", {}).get("inputs", {})
            if a.decode_tile:
                _d["tile_size"] = a.decode_tile
                _d["overlap"] = min(_d.get("overlap", 64), max(16, a.decode_tile // 8))
            if a.decode_temporal:
                _d["temporal_size"] = a.decode_temporal
                _d["temporal_overlap"] = min(_d.get("temporal_overlap", 32),
                                             max(8, a.decode_temporal // 4))
        pr[N_CANVAS]["inputs"]["width"], pr[N_CANVAS]["inputs"]["height"] = tw, th

        t0 = time.time()
        print(f"  chunk {i+1}/{len(segs)}  {L} frames ...", end="", flush=True)
        item = comfy.wait(comfy.submit(pr))
        if not item:
            sys.exit(f"\nChunk {i} failed. Check the ComfyUI console for the error.")
        dst = mine(f"{work}/out_{tag}.mp4")
        mb = comfy.download(item, dst) / 1e6
        # A SHORT return is unrecoverable: ffmpeg cannot invent the missing frames, and -frames:v
        # would silently accept the shortfall. Only an EXCESS is trimmable.
        got_out = nframes(dst)
        if got_out < L:
            sys.exit(f"Chunk {i} came back {got_out} frames, needed {L}. Refusing to assemble — "
                     f"the result would be short and out of sync.")
        if got_out != L:
            t = mine(dst.replace(".mp4", "_t.mp4"))
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", dst, "-frames:v", str(L),
                            "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p",
                            "-c:a", "copy", t], check=True)
            os.replace(t, dst)
            if nframes(dst) != L:
                sys.exit(f"Chunk {i} is {nframes(dst)} frames after trimming, needed {L}.")
        parts.append(dst)
        print(f" {(time.time()-t0)/60:.1f} min, {mb:.1f}MB")

    # Re-time every chunk to the source's exact rational rate before concatenating, so the joined
    # stream has uniform timestamps without any frame being duplicated or dropped.
    rate = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=r_frame_rate", "-of", "csv=p=0", src]).strip() or "24/1"
    retimed = []
    for x in parts:
        rt = mine(x.replace(".mp4", "_rt.mp4"))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", x,
                        "-vf", f"setpts=N/(({rate})*TB)", "-fps_mode", "passthrough",
                        "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p",
                        *([] if a.audio == "generated" else ["-an"]),
                        *(["-c:a", "aac", "-b:a", "192k"] if a.audio == "generated" else []),
                        rt], check=True)
        retimed.append(rt)
    parts = retimed

    # RID-suffixed: these are the two FIXED names that could collide with a user's file.
    lst = mine(f"{work}/concat_{RID}.txt")
    # ffconcat quoting: a single quote inside a path terminates the quoted string. The escape is
    # to close, emit an escaped quote, and reopen. Without it any path like O'Brien/clip.mp4
    # fails at assembly, after every chunk has already been rendered.
    open(lst, "w").write("".join(
        "file '%s'\n" % os.path.abspath(x).replace("'", "'\\''") for x in parts))
    joined = mine(f"{work}/joined_{RID}.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:v", "libx264", "-crf", "16", "-fps_mode", "passthrough", "-pix_fmt",
                    "yuv420p",
                    # -an here used to be unconditional, which silently made --audio generated
                    # produce a silent file.
                    *(["-c:a", "aac", "-b:a", "192k"] if a.audio == "generated" else ["-an"]),
                    joined], check=True)

    # Length gate. A long or short chunk shifts everything after it against the original audio, and
    # the -shortest below would then quietly trim the tail — leaving a file that looks perfect and
    # is out of sync. Better to stop and say so.
    # Force the assembled video to EXACTLY the source frame count. Trimming here is safe because
    # the chunks already tile the source; the discrepancy is a rounding artifact of the CFR concat,
    # not missing content. Checking-with-a-tolerance let an 18-frame assembly of a 17-frame clip
    # through, because the tolerance floor was 4 frames.
    if nframes(joined) > src_frames:
        _t = mine(joined.replace(".mp4", "_x.mp4"))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", joined, "-frames:v", str(src_frames),
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an", _t],
                       check=True)
        os.replace(_t, joined)
    jf = nframes(joined)
    print(f"\n  assembled {jf} frames vs source {src_frames}")
    # EXACT, not a tolerance. An absolute floor of 4 frames is 23% of a 17-frame clip, which let an
    # 18-frame assembly of a 17-frame source through.
    if src_frames and jf != src_frames:
        sys.exit(f"Length mismatch: {jf} vs {src_frames} frames. Refusing to mux.")
    jd = probe(joined)[3]
    if abs(jd - dur) > max(0.15, dur * 0.01):
        sys.exit(f"Duration mismatch: {jd:.2f}s vs source {dur:.2f}s. Frame COUNT can match while "
                 f"the frame RATE does not, and -shortest would then silently trim the picture.")

    if a.audio == "original":
        # `-af apad` BEFORE `-shortest`, and the two must go together.
        #
        # A source's audio track is routinely a few milliseconds shorter than its video: here a
        # 107-frame clip carried 4.448s of audio against 4.458s of picture. `-shortest` alone then
        # stops at the AUDIO end and silently trims the picture — 107 frames in, 105 out, after
        # every length gate in this script had already passed. apad makes the audio effectively
        # infinite, so `-shortest` now stops at the VIDEO end and pads the tail with silence.
        # Losing 8ms of silence is nothing; losing two frames of picture is a desync.
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", joined, "-i", src, "-map", "0:v",
                        "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        *(["-af", "apad"] if has_audio else []), "-shortest",
                        "-movflags", "+faststart", out], check=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", joined, "-c", "copy",
                        "-movflags", "+faststart", out], check=True)

    # Re-check AFTER the mux. Every other gate in this script runs before it, which is exactly how
    # a two-frame loss shipped: the assembly was verified correct and then the mux quietly undid it.
    final_frames = nframes(out)
    if src_frames and final_frames != src_frames:
        sys.exit(f"Muxing changed the frame count: {final_frames} vs {src_frames}. The file at "
                 f"{out} is out of sync with its audio — do not use it.")

    if not a.keep_chunks:
        for p in created:
            if os.path.exists(p):
                os.remove(p)
        # Only if WE emptied it. A directory that still holds someone else's files stays.
        if not os.listdir(work):
            os.rmdir(work)

    print(f"\n  -> {out}  ({os.path.getsize(out)/1e6:.1f}MB, {tw}x{th})")
    print("     Compare a FACE against your source before you ship it: this model invents "
          "skin detail.\n")


if __name__ == "__main__":
    main()
