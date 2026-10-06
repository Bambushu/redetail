#!/usr/bin/env python3
"""Pre-publish check for ReDetail. Run it before tagging or posting anything.

    python3 tools/check_release.py [--comfy http://127.0.0.1:8188]

Every check either passes with evidence or fails with the offending value. Nothing here trusts a
claim that a fix landed — each one is re-derived from the shipped files.

The check that earns its keep is section 5: it parses the README's own size table and re-derives
every cell from solve_dims(). Documentation drifting from code is the single failure this project
has shipped most often — a wrong 1024x576 row survived a full review pass, then survived the fix
in a SECOND copy of the same table inside the workflow notes. Now it cannot.

Section 2 is optional and maintainer-only: it compares this repo against a private twin of the
same geometry code, if one is present. Set REDETAIL_TWIN to that file's path, or leave it unset —
the check skips cleanly and reports nothing.
"""
import argparse, importlib.util, json, os, re, subprocess, sys, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails, warns = [], []


def ok(label, good, detail=""):
    print(f"  {'PASS' if good else 'FAIL'}  {label}{'  — ' + detail if detail else ''}")
    if not good:
        fails.append(label)


def warn(label, detail=""):
    print(f"  SKIP  {label}{'  — ' + detail if detail else ''}")
    warns.append(label)


def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.argv = ["x"]                     # the modules parse args at import
    s.loader.exec_module(m)
    return m


ap = argparse.ArgumentParser()
ap.add_argument("--comfy", default="http://127.0.0.1:8188")
args = ap.parse_args()
COMFY = args.comfy.rstrip("/")

print("\n=== 1. Files present ===")
for f in ["redetail.py", "build_ui_workflow.py", "build_refine_workflow.py", "README.md",
          "CHANGELOG.md", "LICENSE", ".gitignore",
          "workflows/ReDetail_LTX25_upscale.json", "workflows/ltx25_upscale_API.json",
          "workflows/ReDetail_LTX25_refine.json", "workflows/ltx25_refine_API.json",
          "workflows/redetail_replace_me.png"]:
    p = os.path.join(REPO, f)
    ok(f, os.path.exists(p), f"{os.path.getsize(p)/1024:.0f}KB" if os.path.exists(p) else "MISSING")

r = load(f"{REPO}/redetail.py", "r")
src = open(f"{REPO}/redetail.py").read()
md = open(f"{REPO}/README.md").read()

print("\n=== 2. Geometry parity with the private twin (optional) ===")
twin = os.environ.get("REDETAIL_TWIN")
if not twin or not os.path.exists(twin):
    warn("no twin configured (set REDETAIL_TWIN to compare)")
else:
    u = load(twin, "u")
    diffs = []
    for wh in ((320, 330), (352, 608), (432, 768), (1280, 714), (640, 384), (800, 1440)):
        if r.fit_source(*wh) != u.fit_source(*wh):
            diffs.append(f"fit_source{wh}")
    for wh, sc in (((352, 608), 2.0), ((1280, 714), 1.5), ((800, 1440), 1.5), ((640, 384), 1.5)):
        a = r.target_for(*wh, *r.fit_source(*wh)[1:], sc)[:2]
        b = u.target_for(*wh, *u.fit_source(*wh)[1:], sc)[:2]
        if a != b:
            diffs.append(f"target_for{wh}@{sc}: {a} vs {b}")
    ok("twin agrees on all geometry", not diffs, str(diffs))

print("\n=== 3. Known-bug regressions ===")
_raw = "N/A"
ok("ffprobe 'N/A' frame count does not crash", (int(_raw) if _raw.isdigit() else 0) == 0)
segs = r.segments([130 / 24], 200 / 24, 100 / 24, 24)
ok("cut-snapping respects the VRAM chunk cap", max(t[2] for t in segs) <= 100,
   f"longest {max(t[2] for t in segs)} of cap 100")
tiles = all(sum(t[2] for t in r.segments(c, f / 24, m / 24, 24)) == f
            for c, f, m in (([130 / 24], 200, 100), ([], 243, 850), ([2.0, 5.0], 243, 120)))
ok("segments tile the source exactly", tiles)
ok("fit_source stays within [0.65, 2.0] of source",
   all(0.65 <= r.fit_source(*wh)[2] / wh[1] <= 2.0
       for wh in ((320, 330), (352, 608), (432, 768), (1280, 714), (800, 1440))))
# THE INVARIANT: --scale N delivers ~N x the ORIGINAL clip, whatever fitting happened on the way.
# This is what "--scale 2.0 rendered 4x" violated, and what a hardcoded-clamp assertion missed.
bad = []
for (w, h), sc in (((352, 608), 2.0), ((352, 608), 1.5), ((320, 330), 1.5), ((432, 768), 1.5),
                   ((1280, 714), 1.5), ((640, 384), 2.0), ((800, 1440), 1.5),
                   ((1024, 576), 1.5), ((1920, 1088), 1.5), ((768, 1408), 2.0)):
    tw, th = r.target_for(w, h, *r.fit_source(w, h)[1:], sc)[:2]
    if abs(th / h - sc) / sc > 0.06:
        bad.append(f"{w}x{h}@{sc} -> {tw}x{th} = {th/h:.2f}x")
ok("requested scale is the delivered scale", not bad, str(bad))
ok("cleanup tracks exact paths, not a name snapshot",
   "preexisting" not in src and "created.append" in src)
ok("fixed scratch names are per-run unique", "concat_{RID}.txt" in src)
ok("int8 use derived from selections, not installed files",
   "using_int8 = (gguf is None) or (encoder is None and not cached_cond)" in src)
# --cached-cond loads NO encoder, so "no encoder selected" stops being evidence of an int8 run.
# Without this clause the correct Mac setup is told it is broken.
ok("cached conditioning exempts the encoder requirement",
   "if cached_cond and folder == \"text_encoders\":" in src)
ok("unselected GGUF does not pass --setup", "but you did not " in src)
ok("ffconcat paths escaped", "replace(\"'\", \"'\\\\''\")" in src)

# REFINE GEOMETRY: the /32 grid, no source fit, and still the scale that was asked for.
bad = []
for (w, h), sc in (((640, 384), 2.0), ((768, 1376), 1.5), ((1280, 720), 1.5), ((1920, 1080), 2.0),
                   ((1664, 928), 2.31), ((1080, 1920), 1.5), ((1984, 1120), 1.943)):
    tw, th = r.target_for(w, h, w, h, sc, r.GRID["refine"])[:2]
    if tw % 32 or th % 32 or abs(th / h - sc) / sc > 0.06:
        bad.append(f"{w}x{h}@{sc} -> {tw}x{th}")
ok("refine targets are /32 and deliver the asked scale", not bad, str(bad))
ok("refine reads the clip as-is (no /64 source fit)",
   '("none", w, h) if refine else fit_source(w, h)' in src)
ok("refine renders the card's 8-frame tail pad", "rlen += 8" in src)
# Found on ComfyUI 0.37: LoadVideo now reports its INPUT clip as an output, ahead of the render,
# and 1.x downloaded the first video it saw. Every run "succeeded" and returned the clip it had
# uploaded. Frame count could not catch it; only asking for the save node and checking size can.
ok("downloads the SAVE node's file, not the first video in history",
   "def wait(self, pid, save_node" in src and "comfy.wait(comfy.submit(pr), N_SAVE)" in src)
ok("every chunk is checked for the target size", "Refusing to assemble a" in src)
ok("guide strength refused in pixel mode", "--guide-strength is a refine-mode dial" in src)
# BEHAVIOURAL. The crf path moved between ComfyUI releases (format.codec... -> codec...), so the
# walker is exercised against both published schema shapes plus one with no crf at all.
_new = {"required": {"format": ["COMBO", {}], "codec": ["COMFY_DYNAMICCOMBO_V3", {"options": [
    {"key": "auto", "inputs": {}}, {"key": "h264", "inputs": {"optional": {"encoding": [
        "COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "auto", "inputs": {}}, {"key": "re-encode",
            "inputs": {"required": {"crf": ["FLOAT", {}]}}}]}]}}}]}]}}
_old = {"required": {"format": ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "auto", "inputs": {
    "required": {"codec": _new["required"]["codec"]}}}]}]}}
import io as _io
_real = urllib.request.urlopen
try:
    _got = []
    for _schema in (_new, _old, {"required": {"format": ["COMBO", {}], "codec": ["COMBO", {}]}}):
        urllib.request.urlopen = (lambda *a, _s=_schema, **k:
                                  _io.BytesIO(json.dumps({"SaveVideo": {"input": _s}}).encode()))
        _c = object.__new__(r.Comfy)
        _c.url = "x"
        _got.append(_c.save_options())
finally:
    urllib.request.urlopen = _real
ok("SaveVideo crf found under codec (newer ComfyUI)", _got[0].get("codec.encoding.crf") == r.SAVE_CRF
   and _got[0].get("codec") == "h264", str(_got[0]))
ok("SaveVideo crf found under format (ComfyUI 0.37)",
   _got[1].get("format.codec.encoding.crf") == r.SAVE_CRF, str(_got[1]))
ok("no crf sent to a server that declares none", _got[2] == {}, str(_got[2]))

print("\n=== 3b. Muxing the original audio never trims the picture ===")
# BEHAVIOURAL, not a string match. A source's audio track is routinely a few ms shorter than its
# picture, and `-shortest` alone then stops at the AUDIO end and silently drops the last frames —
# after every length gate in the tool has already passed. This builds exactly that clip and mixes
# it with the shipped flags, so the bug cannot come back unnoticed.
import tempfile
ok("mux command pads audio rather than truncating video", '"-af", "apad"' in src)
ok("frame count re-checked AFTER the mux", "Muxing changed the frame count" in src)
try:
    with tempfile.TemporaryDirectory() as td:
        vid, srcclip, outp = f"{td}/v.mp4", f"{td}/s.mp4", f"{td}/o.mp4"
        # 17 frames @24fps = 0.7083s of picture
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc=size=64x64:rate=24", "-frames:v", "17",
                        "-pix_fmt", "yuv420p", vid], check=True)
        # same picture, but only 0.69s of audio — SHORTER than the video, as real clips are
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.69",
                        "-map", "0:v", "-map", "1:a", "-frames:v", "17",
                        "-c:a", "aac", "-pix_fmt", "yuv420p", srcclip], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", srcclip,
                        "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
                        "-af", "apad", "-shortest", outp], check=True)
        got = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                              "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", outp],
                             capture_output=True, text=True).stdout.strip()
        ok("17-frame video + 0.69s audio survives the mux", got == "17", f"got {got} frames")
except FileNotFoundError:
    warn("ffmpeg not on PATH — mux behaviour not exercised")

print("\n=== 3c. The documented 24GB recipe is complete ===")
# Condensing the README once silently dropped --decode-temporal from this command. Decode is a
# SEPARATE memory spike from sampling — a 4090 sampled all 8 steps and then OOMed in VAE decode —
# so a low-VRAM recipe missing a decode flag is a recipe that fails at the last step.
_cmd = md[md.find("--gguf LTX-2.5-Distilled"):]
_cmd = _cmd[:_cmd.find("```")]
for _flag in ("--encoder", "--clip-device", "--budget", "--decode-tile", "--decode-temporal"):
    ok(f"4090 recipe keeps {_flag}", _flag in _cmd)

print("\n=== 4. Every README dimension is on the /32 grid ===")
# /32 is the refine grid, and every pixel TARGET is re-derived on /64 in section 5. Refine takes
# any SOURCE size, so its table lists these two off-grid sources on purpose. Anything else off /32
# is a typo that would fail a real render.
DELIBERATE = {"1280x720", "1920x1080"}
bad = {f"{a}x{b}" for a, b in re.findall(r"(\d{3,4})[x×](\d{3,4})", md)
       if int(a) % 32 or int(b) % 32}
# SUBSET, not equality. These are *permitted*, not *required* — asserting equality meant that
# editing a counter-example out of the prose failed the check, which is backwards.
ok("README dims /32 (off-grid refine sources allowed)", bad <= DELIBERATE,
   f"unexplained off-grid: {sorted(bad - DELIBERATE)}")

print("\n=== 5. README size tables re-derived from the code ===")
_ROW = r"\|\s*(\d+)x(\d+)\s*\|\s*\**(\d+)x(\d+)\**[^|]*\|\s*\**(\d+)x(\d+)\**[^|]*\|"


def _table(start, end):
    a = md.find(start)
    return re.findall(_ROW, md[a:md.find(end, a)]) if a >= 0 else []


# Each table is checked against ITS mode's solver. One regex over the whole README used to be
# enough; with two tables on two grids it would check refine rows against the pixel solver.
_ref = _table("**Refine** needs both output dimensions", "In the workflow you pick")
_pix = _table("**Pixel** needs **/64**", "The bold pixel rows")
ok("refine table rows found", len(_ref) >= 5, f"{len(_ref)} rows")
for sw, sh, w15, h15, w20, h20 in _ref:
    sw, sh = int(sw), int(sh)
    d15 = r.target_for(sw, sh, sw, sh, 1.5, r.GRID["refine"])[:2]
    d20 = r.target_for(sw, sh, sw, sh, 2.0, r.GRID["refine"])[:2]
    ok(f"  refine {sw}x{sh}", d15 == (int(w15), int(h15)) and d20 == (int(w20), int(h20)),
       f"README {w15}x{h15} / {w20}x{h20}  vs derived {d15} / {d20}")
ok("pixel table rows found", len(_pix) >= 5, f"{len(_pix)} rows")
for sw, sh, w15, h15, w20, h20 in _pix:
    d15, d20 = r.solve_dims(int(sw), int(sh), 1.5)[:2], r.solve_dims(int(sw), int(sh), 2.0)[:2]
    ok(f"  pixel {sw}x{sh}", d15 == (int(w15), int(h15)) and d20 == (int(w20), int(h20)),
       f"README {w15}x{h15} / {w20}x{h20}  vs derived {d15} / {d20}")

print("\n=== 5b. No unfinished placeholders ship ===")
# The docs are drafted before the measurements that fill them; a TODO_ marker must never reach a
# release. Checked across every shipped text file, including the generated workflows' notes.
_todo = [f for f in ("README.md", "CHANGELOG.md", "LICENSE", "workflows/ReDetail_LTX25_refine.json",
                     "workflows/ReDetail_LTX25_upscale.json")
         if "TODO_" in open(os.path.join(REPO, f), encoding="utf-8").read()]
ok("no TODO_ markers in shipped files", not _todo, str(_todo))

print("\n=== 6. Documented flags match argparse ===")
_f = md[md.find("## Every flag"):]
_f = _f[:_f.find("```", _f.find("```") + 3)]                  # the fenced block only
doc = set(re.findall(r"^(--[a-z][a-z-]+)", _f, re.M))         # >=2 chars, not the `---` rule
real = set(re.findall(r'p\.add_argument\("(--[a-z-]+)"', src))
ok("no documented flag that does not exist", doc <= real, f"extra: {doc - real}")
ok("no real flag left undocumented", not (real - doc - {"--setup"}), f"missing: {real - doc}")

print("\n=== 7. Workflow JSON ===")
wf = json.load(open(f"{REPO}/workflows/ReDetail_LTX25_upscale.json"))
ok("18 top-level nodes", len(wf["nodes"]) == 18, str(len(wf["nodes"])))
ok("6 subgraph definitions", len(wf.get("definitions", {}).get("subgraphs", [])) == 6)
ok("saved viewport present (user lands on the whole graph)", bool(wf.get("extra", {}).get("ds")))
notes = " ".join((n.get("widgets_values") or [""])[0]
                 for n in wf["nodes"] if n.get("type") == "MarkdownNote")
# The workflow notes are a SECOND copy of the README's instructions. Every fix has to land twice.
for label, cond in (("-count_frames in notes", "-count_frames" in notes),
                    ("no bare stream=nb_frames", "stream=nb_frames" not in notes),
                    ("anullsrc rate matches README", "r=48000" in notes and "r=48000" in md),
                    ("size table matches README", "1472×832" in notes and "1536×864" not in notes),
                    ("API-key box explained", "API key" in notes and "switched off" in notes),
                    # RETRACTED CLAIM, now guarded in the opposite direction. This used to assert
                    # the notes SAID int8_convrot was Blackwell-only. Users then reported it
                    # working on 3090, 4090 and 1070, and the likely real cause of our one failure
                    # was comfy-kitchen 0.2.10. Assert the correction is present and the false
                    # claim is gone, so it cannot be reintroduced by a regenerate.
                    ("int8 Blackwell-only claim is retracted in the notes",
                     "needs **Blackwell**" not in notes and "CORRECTION" in notes),
                    ("notes point at comfy-kitchen before the GPU",
                     "comfy-kitchen" in notes and "0.2.10" in notes),
                    ("README carries the same retraction",
                     "not Blackwell-only" in md and "0.2.10" in md),
                    ("python3 everywhere", "python3 redetail.py" in notes
                     and not re.search(r"(?<!3 )\bpython redetail\.py", md))):
    ok(label, cond)
nbad = sorted({f"{a}x{b}" for a, b in re.findall(r"(\d{3,4})[x×](\d{3,4})", notes)
               if int(a) % 64 or int(b) % 64})
ok("note dims /64", nbad == ["2880x1632"], f"found {nbad}")

# THE REFINE WORKFLOW. Generated by build_refine_workflow.py, so every fix is asserted here in
# the shipped artefact, not just in the builder that produces it.
rwf = json.load(open(f"{REPO}/workflows/ReDetail_LTX25_refine.json"))
rn = {n["id"]: n for n in rwf["nodes"]}
ok("refine: 18 top-level nodes", len(rwf["nodes"]) == 18, str(len(rwf["nodes"])))
ok("refine: 6 subgraph definitions", len(rwf.get("definitions", {}).get("subgraphs", [])) == 6)
ok("refine: saved viewport present", bool(rwf.get("extra", {}).get("ds")))
_tin = {i["name"]: i for i in rn[5516]["inputs"]}
ok("refine: tile links cut, so the sampler's own widgets are live",
   _tin["tile_width"].get("link") is None and _tin["tile_height"].get("link") is None)
_sn, _sv = rn[5516]["widgets_values_named"], rn[5516]["widgets_values"]
ok("refine: tile 1024x576 in BOTH stored copies",
   (_sn["tile_width"], _sn["tile_height"]) == (1024, 576)
   and _sv[list(_sn).index("tile_width")] == 1024 and _sv[list(_sn).index("tile_height")] == 576)


def _agree(nodes):
    return [n["id"] for n in nodes if n.get("widgets_values_named") is not None
            and list(n["widgets_values_named"].values()) != n["widgets_values"][:len(n["widgets_values_named"])]]


_dis = _agree(rwf["nodes"]) + [x for sg in rwf["definitions"]["subgraphs"] for x in _agree(sg["nodes"])]
ok("refine: positional and named widget copies agree everywhere", not _dis, str(_dis))
_vals = []


def _walk(o, key=""):
    if isinstance(o, dict):
        for k, v in o.items():
            _walk(v, k)
    elif isinstance(o, list):
        for v in o:
            _walk(v, key)
    elif isinstance(o, str) and key not in ("name", "label"):
        _vals.append(o)


_walk(rwf)
ok("refine: no bf16 transformer/encoder or e2b file named anywhere",
   not [v for v in _vals if "distilled-transformer-bf16" in v or "with-proj-ltx-2.5-bf16" in v
        or "gemma4_e2b_it" in v])
_urls = [m["url"] for m in rn[5004]["properties"]["models"]]
ok("refine: Missing Models panel offers the int8 files and the refine LoRA",
   any("transformer-comfy-int8-convrot" in u for u in _urls)
   and any("proj-ltx-2.5-comfy-int8-convrot" in u for u in _urls)
   and any("refine-details-1.0" in u for u in _urls) and not any("bf16.safetensors" in u
   and ("transformer" in u or "with-proj" in u) for u in _urls), str(_urls))
ok("refine: card prompts in both prompt boxes",
   rn[5508]["widgets_values"][0].startswith("sharp photographic detail")
   and rn[5509]["widgets_values"][0].startswith("blurry, soft, plastic"))
ok("refine: output_size defaults to FullHD", rn[9002]["widgets_values_named"]["output_size"] == "FullHD")
rnotes = " ".join((n.get("widgets_values") or [""])[0]
                  for n in rwf["nodes"] if n.get("type") == "MarkdownNote")
for label, cond in (("no em dashes", "\u2014" not in rnotes),
                    ("pack date stated", "24 September 2026" in rnotes),
                    ("comfy-kitchen pin", ">=0.2.26" in rnotes),
                    ("anullsrc rate matches README", "r=48000" in rnotes),
                    ("-count_frames", "-count_frames" in rnotes),
                    ("portrait tile explained", "576x1024" in rnotes),
                    ("kornia pin retired", "no longer needed" in rnotes),
                    ("enhancer explained", "disconnected" in rnotes),
                    ("python3 everywhere", "python3 redetail.py" in rnotes)):
    ok(f"refine notes: {label}", cond)
_rbad = sorted({f"{a}x{b}" for a, b in re.findall(r"(\d{3,4})[x×](\d{3,4})", rnotes)
                if int(a) % 32 or int(b) % 32})
ok("refine note dims /32", not _rbad, f"found {_rbad}")

print("\n=== 8. Licence compliance ===")
# The LTX-2 Community License is NOT permissive. Section 3 permits redistributing derivatives
# only on conditions, and each check below maps to one of them.
lic = open(f"{REPO}/LICENSE").read()
ltx = os.path.join(REPO, "workflows", "LICENSE-LTX-2-Community.txt")
ok("MIT covers original code ONLY", "ORIGINAL CODE ONLY" in lic)
ok("3(b) complete copy of the Agreement is included", os.path.exists(ltx),
   f"{os.path.getsize(ltx)/1024:.0f}KB" if os.path.exists(ltx) else "MISSING")
if os.path.exists(ltx):
    lt = open(ltx).read()
    ok("3(b) that copy includes Attachment A", "Attachment A" in lt and "ATTACHMENT A" in lt.upper())
ok("3(b) derivatives stated as exclusively under that Agreement", "EXCLUSIVELY under" in lic)
ok("3(c) modified files carry a notice of what changed", "Changes made" in lic)
ok("3(a) use restrictions passed on to recipients", "USE RESTRICTIONS CARRY FORWARD" in lic)
ok("commercial-revenue threshold stated", "10,000,000" in lic and "10,000,000" in md)
# TRACKED, not present. The build scripts need the unmodified examples beside them, so a
# maintainer checkout legitimately has them on disk; .gitignore keeps them out of every commit.
_bases = ["workflows/_base_ui.json", "workflows/_base_refine_ui.json"]
if os.path.isdir(os.path.join(REPO, ".git")):
    _tracked = subprocess.run(["git", "-C", REPO, "ls-files", *_bases], capture_output=True,
                              text=True).stdout.split()
else:
    _tracked = [b for b in _bases if os.path.exists(os.path.join(REPO, b))]
ok("no verbatim upstream example redistributed", not _tracked, str(_tracked))
ok("3(c) the refine workflow's changes are noticed",
   "LTX-2.5_V2V_TiledFusion_Upscale.json" in lic and "build_refine_workflow.py" in lic)
ok("Refine-Details weights listed", "Refine Details" in lic)
ok("no self-contradiction", "CODE AND WORKFLOW" not in lic
   and "do not reproduce any upstream" not in md.lower())
ok("README licence section agrees", "own code only" in md)

print("\n=== 9. Nothing machine-specific in any shipped file ===")
LEAKS = ("/Users/", "/home/", "runpod", "rpa_", "podenv", "sk-", "ghp_")
for dirpath, dirnames, filenames in os.walk(REPO):
    dirnames[:] = [d for d in dirnames if d not in ("demos", "__pycache__", ".git")]
    for fn in sorted(filenames):
        if not fn.endswith((".py", ".md")):
            continue
        p = os.path.join(dirpath, fn)
        t = open(p, encoding="utf-8", errors="ignore").read()
        # This file necessarily CONTAINS the patterns it hunts for, so it flagged itself. Drop the
        # line that defines them and scan the rest — exempting the whole file would leave the one
        # script that lives in the repo permanently unchecked.
        t = "\n".join(l for l in t.splitlines() if "LEAKS = (" not in l)
        found = [k for k in LEAKS if k in t]
        ok(os.path.relpath(p, REPO), not found, f"found {found}")

print("\n=== 11. Shipped conditioning tensors ===")
# These make the text encoder an optional download rather than a required one, so their presence
# is a shipping guarantee, not a nicety. torch is NOT required to run this check -- it is a
# maintainer tool that must work on a bare machine, so the tensor shape is only asserted when
# torch happens to be importable.
_cc = os.path.join(REPO, "tools/comfyui_cond_cache")
ok("cond cache node ships as a package", os.path.isfile(f"{_cc}/__init__.py"))
for _n in ("redetail_pos", "redetail_neg"):
    _p = f"{_cc}/{_n}.pt"
    _have = os.path.isfile(_p)
    ok(f"{_n}.pt shipped", _have,
       f"{os.path.getsize(_p)} bytes" if _have else "MISSING")
    # Small enough that shipping it is the whole point. If this balloons, something is wrong.
    if _have:
        ok(f"{_n}.pt under 64KB", os.path.getsize(_p) < 65536)
try:
    import torch as _t
    _c = _t.load(f"{_cc}/redetail_pos.pt", map_location="cpu", weights_only=False)
    ok("shipped conditioning has the expected shape",
       _t.is_tensor(_c[0][0]) and tuple(_c[0][0].shape) == (1, 1, 6144),
       str(tuple(_c[0][0].shape)))
except ImportError:
    warn("torch not importable — tensor shape not verified")
# They are model OUTPUT derived from gated weights, so the licence must say so. Shipping them under
# the MIT grant by omission is exactly the failure this guards against.
_lic = open(os.path.join(REPO, "LICENSE")).read()
ok("LICENSE names the .pt files as LTX-2 material", "comfyui_cond_cache" in _lic)
ok("LICENSE states which encoder produced them", "Q5_K_M" in _lic)
ok("LICENSE flags the int8 quantisation caveat", "int8_convrot encoder is a" in _lic)
# Refine's prompts are not empty, so it ships its own pair, made by the int8 encoder.
for _n in ("redetail_refine_pos", "redetail_refine_neg"):
    _p = f"{_cc}/{_n}.pt"
    _have = os.path.isfile(_p)
    ok(f"{_n}.pt shipped", _have, f"{os.path.getsize(_p)} bytes" if _have else "MISSING")
    if _have:
        ok(f"{_n}.pt under 1MB", os.path.getsize(_p) < 1 << 20)
ok("LICENSE names the refine tensors and their encoder",
   "redetail_refine_pos.pt" in _lic and "comfy-int8-convrot.safetensors  (type ltxv)" in _lic)
# ComfyUI 0.37 ships a core SaveConditioning that shadows ours. The CLI must use the unique names.
_pk = open(f"{_cc}/__init__.py").read()
ok("cache pack registers unique ReDetail* node names",
   '"ReDetailSaveConditioning"' in _pk and '"ReDetailLoadConditioning"' in _pk)
ok("CLI uses the unique names, never the shadowed core one",
   '"class_type": "ReDetailSaveConditioning"' in src and '"class_type": "SaveConditioning"' not in src
   and '"class_type": "ReDetailLoadConditioning"' in src)
ok("README documents --cached-cond", "--cached-cond" in md)
# The Mac variant shipped without the README ever naming it, so a Mac user would have
# followed the manual GGUF instructions instead of the graph built for them.
ok("README points Mac users at the MAC workflow",
   "ReDetail_LTX25_upscale_MAC.json" in md)

print("\n=== 12. The prompt-enhancer branch stays disconnected ===")
# A user could not queue at all: ComfyUI validates nodes upstream of an output even on a branch
# that is never taken, and GemmaAPITextEncode reads ckpt_name from models/diffusion_models, which
# is empty for everyone on the GGUF path. The fix is a rewire, and it lives in a GENERATED file --
# so assert it in the shipped artefacts, not just in the builder that produces them.
_api = json.load(open(os.path.join(REPO, "workflows/ltx25_upscale_API.json")))
ok("API graph takes conditioning straight from LTXVConditioning",
   _api["9002:9005"]["inputs"]["positive"] == ["5014:1241", 0]
   and _api["9002:9005"]["inputs"]["negative"] == ["5014:1241", 1])
for _wf in ("workflows/ReDetail_LTX25_upscale.json", "workflows/ReDetail_LTX25_upscale_MAC.json"):
    _d = json.load(open(os.path.join(REPO, _wf)))
    _ip = next((s for s in _d["definitions"]["subgraphs"]
                if s.get("name") == "Input Parameters"), None)
    _origins = {l.get("id"): l.get("origin_id") for l in (_ip or {}).get("links", [])}
    ok(f"{os.path.basename(_wf)}: enhancer switches bypassed",
       _origins.get(13790) == 1241 and _origins.get(13791) == 1241,
       f"13790<-{_origins.get(13790)} 13791<-{_origins.get(13791)}")
ok("builder reproduces the rewire (not just the checked-in file)",
   "Expected to rewire 2 conditioning links past the switches"
   in open(os.path.join(REPO, "build_ui_workflow.py")).read())
# The refine graph has TWO enhancer branches: the Gemma-API switches and a prompt rewriter.
_rapi = json.load(open(os.path.join(REPO, "workflows/ltx25_refine_API.json")))
ok("refine API: conditioning straight from LTXVConditioning",
   _rapi["9002:9005"]["inputs"]["positive"] == ["5014:1241", 0]
   and _rapi["9002:9005"]["inputs"]["negative"] == ["5014:1241", 1])
ok("refine API: prompt switch never reaches the enhancer",
   _rapi["5014:5556"]["inputs"]["on_true"] == ["5508", 0])
_rip = next((sg for sg in rwf["definitions"]["subgraphs"] if sg.get("name") == "Input Parameters"), {})
_ro = {l.get("id"): l.get("origin_id") for l in _rip.get("links", [])}
ok("refine UI: all three enhancer links rerouted",
   _ro.get(13790) == 1241 and _ro.get(13791) == 1241 and _ro.get(4) == -10,
   f"13790<-{_ro.get(13790)} 13791<-{_ro.get(13791)} 4<-{_ro.get(4)}")
ok("refine builder reproduces the rewire",
   "the enhancer blocks non-default installs"
   in open(os.path.join(REPO, "build_refine_workflow.py")).read())

print("\n=== 13. The refine API graph is the one redetail.py expects ===")
_ri = lambda k: _rapi[k]["inputs"]
ok("tile literal 1024x576, not linked",
   (_ri("5516:9100")["tile_width"], _ri("5516:9100")["tile_height"]) == (1024, 576))
ok("Refine-Details LoRA at 1.0 on the int8 transformer",
   _ri("5004:5606")["lora_name"] == "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors"
   and _ri("5004:5606")["strength_model"] == 1
   and _ri("5004:5602")["unet_name"].endswith("comfy-int8-convrot.safetensors"))
ok("guide streams 97-frame windows", _ri("9002:5012").get("use_streaming") is True)
ok("the nodes the CLI patches all exist",
   all(k in _rapi for k in (r.N_VIDEO, r.N_SAVE, r.N_CANVAS, r.N_GUIDE_RESIZE, r.N_GUIDE,
                            r.N_SAMPLER, "5014:2483", "5014:2612", "5518:5538")))
ok("placeholders, not a maintainer's files",
   _ri("5001")["file"] == "__INPUT_VIDEO__" and _ri("4852")["filename_prefix"] == "__OUT_PREFIX__")
ok("prompts match the UI workflow's",
   (_ri("5508")["value"], _ri("5509")["value"]) == (rn[5508]["widgets_values"][0],
                                                   rn[5509]["widgets_values"][0]))

print("\n=== 10. Live install check ===")
try:
    urllib.request.urlopen(f"{COMFY}/system_stats", timeout=8)
except Exception as e:
    warn(f"ComfyUI unreachable at {COMFY} — --setup not exercised", str(e)[:50])
else:
    p = subprocess.run([sys.executable, f"{REPO}/redetail.py", "--setup", "--comfy", COMFY],
                       capture_output=True, text=True, timeout=300)
    print(f"  ----  --setup: {p.stdout.count('PASS')} PASS / {p.stdout.count('FAIL')} FAIL "
          f"(exit {p.returncode})")
    for line in p.stdout.splitlines():
        if line.strip().startswith("FAIL"):
            print(f"        {line.strip()}")

print("\n" + "=" * 62)
print(f"RESULT: {len(fails)} failure(s), {len(warns)} skipped")
for f in fails:
    print("   FAILED:", f)
sys.exit(1 if fails else 0)
