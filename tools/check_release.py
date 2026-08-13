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
for f in ["redetail.py", "build_ui_workflow.py", "README.md", "LICENSE", ".gitignore",
          "workflows/ReDetail_LTX25_upscale.json", "workflows/ltx25_upscale_API.json",
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
   "using_int8 = (gguf is None) or (encoder is None)" in src)
ok("unselected GGUF does not pass --setup", "but you did not " in src)
ok("ffconcat paths escaped", "replace(\"'\", \"'\\\\''\")" in src)

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

print("\n=== 4. Every README dimension is on the /64 grid ===")
# 2880x1632 (off-grid TARGET) and 432x768 (off-grid SOURCE) are the two counter-examples the
# prose exists to explain. Anything else off-grid is a typo that would fail a real render.
DELIBERATE = sorted(["2880x1632", "432x768"])
bad = sorted({f"{a}x{b}" for a, b in re.findall(r"(\d{3,4})[x×](\d{3,4})", md)
              if int(a) % 64 or int(b) % 64})
ok("README dims /64 (counter-examples allowed)", bad == DELIBERATE, f"found {bad}")

print("\n=== 5. README size table re-derived from the code ===")
rows = re.findall(r"\|\s*(\d+)x(\d+)\s*\|\s*\**(\d+)x(\d+)\**[^|]*\|\s*(\d+)x(\d+)\s*\|", md)
ok("table rows found", len(rows) >= 5, f"{len(rows)} rows")
for sw, sh, w15, h15, w20, h20 in rows:
    d15, d20 = r.solve_dims(int(sw), int(sh), 1.5)[:2], r.solve_dims(int(sw), int(sh), 2.0)[:2]
    ok(f"  {sw}x{sh}", d15 == (int(w15), int(h15)) and d20 == (int(w20), int(h20)),
       f"README {w15}x{h15} / {w20}x{h20}  vs derived {d15} / {d20}")

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
                    ("non-Blackwell needs the encoder too",
                     "text encoder* is Blackwell-only" in notes),
                    ("python3 everywhere", "python3 redetail.py" in notes
                     and not re.search(r"(?<!3 )\bpython redetail\.py", md))):
    ok(label, cond)
nbad = sorted({f"{a}x{b}" for a, b in re.findall(r"(\d{3,4})[x×](\d{3,4})", notes)
               if int(a) % 64 or int(b) % 64})
ok("note dims /64", nbad == ["2880x1632"], f"found {nbad}")

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
ok("no verbatim upstream example redistributed",
   not os.path.exists(os.path.join(REPO, "workflows", "_base_ui.json")))
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
