#!/usr/bin/env python3
"""Build the CivitAI / Reddit asset set for ReDetail from the demo render triples.

METHODOLOGY — read this before you trust any image it makes.

The naive comparison (640x384 source next to a 1280x768 result) is not a comparison, it is a
resolution difference. Anything looks better when it is twice as big. Every pairing here upscales
the SOURCE with Lanczos to the exact output size first, so both halves are the same pixel count
and the only variable left is where the pixels came from — resampled vs synthesized.

Crop regions are chosen by scoring the SOURCE frame, never the ReDetail frame. Scoring the result
would pick whichever tile the model invented the most detail in, which is cherry-picking dressed up
as automation.

Every annotated PNG is written beside an identical unlabelled `_clean` copy. Labels burned into a
crop have faked a finding here before — the clean file is what you re-measure from.

    python3 make_assets.py                # everything
    python3 make_assets.py --scene rust   # one scene
    python3 make_assets.py --no-video     # stills only (fast)
"""
import argparse, os, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REND = f"{HERE}/renders"
OUT = f"{HERE}/post"
FONTS = "/System/Library/Fonts/Supplemental"
BLACK = f"{FONTS}/Arial Black.ttf"
BOLD = f"{FONTS}/Arial Bold.ttf"
REG = f"{FONTS}/Arial.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"          # settings box — values must column-align

# The upscale side of every render, read out of workflows/ltx25_upscale_API.json rather than typed
# from memory: 9 manual sigmas = 8 steps, CFG 1.0 (the model is distilled, so guidance is off),
# IC-LoRA and guide both at strength 1.0. If the shipped graph changes, re-read it — a settings box
# carrying stale numbers is worse than no settings box.
UPSCALE_SETTINGS = [
    ("MODEL", "LTX-2.5 22B distilled, int8_convrot"),
    ("IC-LORA", "pixel-spatial-upscaler-x2-1.0 @ 1.0, guide 1.0"),
    ("SAMPLER", "8 steps, euler_ancestral, CFG 1.0"),
]

# The six complete source/1.5x/2.0x triples. `t` is the sample second — hand-picked per scene so
# the frame is mid-motion rather than the first frame, which on H3 is often the calmest one.
SCENES = [
    ("forest", 10, 4.0, "Forest canopy, backlit", "DEMO_forest_10st"),
    ("forest", 15, 4.0, "Forest canopy, backlit", "DEMO_forest_15st"),
    ("portrait", 10, 5.0, "Portrait, window light", "DEMO_portrait_10st"),
    ("portrait", 15, 5.0, "Portrait, window light", "DEMO_portrait_15st"),
    ("rain", 15, 5.0, "Rain on glass, night", "DEMO_rain_15st"),
    ("rust", 15, 6.0, "Rusted metal, macro", "DEMO_rust_15st"),
    # Fast motion. The stem carries a _spec suffix, which is why it is stated rather than built
    # from name+steps like the others.
    ("action", 25, 5.0, "Motocross, dirt spray", "DEMO_action_25st_spec"),
]

# Scenes where a single still is the wrong test. On fast motion the question is not "is this frame
# sharper" but "does the invented detail hold still between frames", so these also get a filmstrip.
MOTION = {"action"}

FG, DIM, ACCENT, BG = (255, 255, 255), (150, 150, 150), (255, 138, 76), (14, 14, 16)
PAD, GAP = 28, 14
CROP = (600, 450)          # 100% pixel window. Overridable with --crop.

# The scenes that actually READ in a comparison, best first. ACTION LEADS EVERYWHERE (Mike's
# call): fast motion is the question people actually ask of a video upscaler, and it is the one
# case a still-image upscaler cannot answer at all. Foliage and hair follow because their
# difference lands on structure you can name — a frond edge, a single strand. Rusted metal scored
# highest for raw high-frequency energy and still read worst, because its difference is grain on
# grain with no structure to anchor it. Detail energy is not the same as legibility.
HERO = [("action", 25, "2.0"), ("forest", 15, "2.0"),
        ("portrait", 15, "2.0"), ("portrait", 10, "2.0")]


def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


def sh_out(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def font(path, size):
    return ImageFont.truetype(path, size)


def grab(video, t, dst):
    """One frame at time t. -ss AFTER -i: as an input option it is a keyframe seek and lands
    somewhere else entirely, which desynced a whole trailer once."""
    sh("ffmpeg", "-y", "-v", "error", "-i", video, "-ss", str(t), "-frames:v", "1", dst)


def lanczos(src_png, w, h, dst):
    sh("ffmpeg", "-y", "-v", "error", "-i", src_png,
       "-vf", f"scale={w}:{h}:flags=lanczos", dst)


def pick_roi(src_img, out_w, out_h, crop_w, crop_h):
    """Highest-detail tile of the SOURCE, returned in OUTPUT coordinates.

    Score = mean absolute difference between the tile and a blurred copy of itself, i.e. how much
    high-frequency energy is there. Edges of frame are excluded — H3 puts its worst artifacts in
    the outer 8%, and a crop that lands there shows off the generator's flaws, not the upscaler's.
    """
    from PIL import ImageFilter, ImageChops, ImageStat
    sw, sh_ = src_img.size
    cw, ch = int(crop_w * sw / out_w), int(crop_h * sh_ / out_h)
    m = 0.08
    best, best_score = (int(sw * 0.5 - cw / 2), int(sh_ * 0.5 - ch / 2)), -1e9
    step_x, step_y = max(8, (sw - cw) // 12), max(8, (sh_ - ch) // 12)
    for y in range(int(sh_ * m), max(int(sh_ * m) + 1, int(sh_ * (1 - m)) - ch + 1), step_y):
        for x in range(int(sw * m), max(int(sw * m) + 1, int(sw * (1 - m)) - cw + 1), step_x):
            tile = src_img.crop((x, y, x + cw, y + ch)).convert("L")
            d = ImageChops.difference(tile, tile.filter(ImageFilter.GaussianBlur(2)))
            s = ImageStat.Stat(d).mean[0]
            # Penalise a tile whose two halves differ a lot in brightness. These crops get shown
            # split down the middle, and a crop straddling a hard light/dark boundary reads as
            # "the two halves are different pictures" even when it is one continuous frame — which
            # is exactly what a forest tile did on the first cover. Detail selection is unchanged;
            # this only breaks ties away from edges that wreck the split presentation.
            half = cw // 2
            imb = abs(ImageStat.Stat(tile.crop((0, 0, half, ch))).mean[0]
                      - ImageStat.Stat(tile.crop((half, 0, cw, ch))).mean[0])
            s -= 0.06 * imb
            if s > best_score:
                best_score, best = s, (x, y)
    return (round(best[0] * out_w / sw), round(best[1] * out_h / sh_))


def fit(draw, text, path, size, maxw, xy, color):
    """Draw `text` at `size`, shrinking until it fits `maxw`. The cover is sized by its crops, so
    header strings that fit one grid silently ran off the edge of another."""
    f = font(path, size)
    while size > 9 and draw.textlength(text, font=f) > maxw:
        size -= 1
        f = font(path, size)
    draw.text(xy, text, font=f, fill=color)


def label_bar(draw, x, y, w, text, sub=None, color=FG):
    draw.text((x, y), text, font=font(BLACK, 22), fill=color)
    if sub:
        draw.text((x, y + 28), sub, font=font(REG, 15), fill=DIM)


def settings_box(d, x, y, w, rows, title="SETTINGS"):
    """Bordered panel of LABEL / value rows in mono, so values column-align down the panel."""
    lab, val, lh = font(MONO, 12), font(MONO, 12), 19
    h = 26 + len(rows) * lh + 8
    d.rectangle([x, y, x + w, y + h], outline=(52, 52, 58), fill=(22, 22, 25))
    d.text((x + 12, y + 8), title, font=font(BLACK, 11), fill=ACCENT)
    lw = max(d.textlength(k, font=lab) for k, _ in rows) + 14
    for i, (k, v) in enumerate(rows):
        yy = y + 26 + i * lh
        d.text((x + 12, yy), k, font=lab, fill=DIM)
        d.text((x + 12 + lw, yy), v, font=val, fill=FG)
    return h


def pair_png(left, right, ltitle, lsub, rtitle, rsub, caption, dst, settings=None):
    """Two same-size crops side by side, on the dark card. Also writes `_clean`."""
    w, h = left.size
    W = PAD * 2 + w * 2 + GAP
    head = 132
    # Footer holds the methodology line plus the settings panel, so it grows with the row count.
    foot = 58 if not settings else 58 + 26 + len(settings) * 19 + 8 + 14
    H = head + h + foot
    card = Image.new("RGB", (W, H), BG)
    card.paste(left, (PAD, head))
    card.paste(right, (PAD + w + GAP, head))
    clean = card.copy()

    d = ImageDraw.Draw(card)
    ft = font(BLACK, 30)
    d.text((PAD, 22), "ReDetail", font=ft, fill=ACCENT)
    # Measure the wordmark rather than guessing an offset — "ReDetail" in Arial Black is wider
    # than it looks and a fixed 132px indent ran the caption straight through it.
    d.text((PAD + d.textlength("ReDetail", font=ft) + 22, 29), caption, font=font(REG, 17),
           fill=DIM)
    label_bar(d, PAD, head - 58, w, ltitle, lsub, DIM)
    label_bar(d, PAD + w + GAP, head - 58, w, rtitle, rsub, FG)
    d.rectangle([PAD + w + GAP - 1, head - 1, PAD + w * 2 + GAP, head + h], outline=ACCENT, width=2)
    d.text((PAD, head + h + 16), "100% pixel crop. Both halves identical size — source Lanczos-"
           "resampled to match. No sharpening applied to either.", font=font(REG, 14), fill=DIM)
    if settings:
        settings_box(d, PAD, head + h + 44, W - PAD * 2, settings)
    card.save(dst)
    clean.save(dst.replace(".png", "_clean.png"))
    return card


def title_card(text, sub, w, h, dst):
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.text((w // 2, h // 2 - 46), "ReDetail", font=font(BLACK, int(h * 0.13)),
           fill=ACCENT, anchor="mm")
    d.text((w // 2, h // 2 + 26), text, font=font(BOLD, int(h * 0.055)), fill=FG, anchor="mm")
    d.text((w // 2, h // 2 + 74), sub, font=font(REG, int(h * 0.036)), fill=DIM, anchor="mm")
    img.save(dst)
    return img


def filmstrip(src, up, sw, sh_, t, box, caption, dst, n=4, step=2):
    """N consecutive frames of the same crop: Lanczos on top, ReDetail below.

    This is the only honest way to show a generative upscaler on fast motion. A single frame just
    says "sharper"; what matters is whether the detail it invents is the SAME detail next frame, or
    whether it re-invents from scratch and boils. Consecutive frames put that on one page.
    """
    fps = float(sh_out("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "stream=r_frame_rate", "-of", "csv=p=0", up).split("/")[0] or 24) / 1.0
    rate = sh_out("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                  "stream=r_frame_rate", "-of", "csv=p=0", up) or "24/1"
    nu, _, de = rate.partition("/")
    fps = float(nu) / float(de or 1)

    rows = [[], []]
    for i in range(n):
        tt = t + (i * step) / fps
        a, b = f"{OUT}/.fs_a.png", f"{OUT}/.fs_b.png"
        grab(src, tt, a)
        lz = f"{OUT}/.fs_lz.png"
        lanczos(a, sw, sh_, lz)
        grab(up, tt, b)
        rows[0].append(Image.open(lz).crop(box).copy())
        rows[1].append(Image.open(b).crop(box).copy())

    cw, ch = rows[0][0].size
    head, lab, foot, g = 108, 26, 70, 8
    W = PAD * 2 + cw * n + g * (n - 1)
    H = head + ch * 2 + lab * 2 + g + foot
    card = Image.new("RGB", (W, H), BG)
    for r, row in enumerate(rows):
        y = head + lab + r * (ch + lab + g)
        for i, im in enumerate(row):
            card.paste(im, (PAD + i * (cw + g), y))
    clean = card.copy()

    d = ImageDraw.Draw(card)
    ft = font(BLACK, 30)
    d.text((PAD, 22), "ReDetail", font=ft, fill=ACCENT)
    d.text((PAD + d.textlength("ReDetail", font=ft) + 22, 29), caption, font=font(REG, 17),
           fill=DIM)
    d.text((PAD, 60), f"{n} frames, {step} apart. Same crop, same output size.",
           font=font(BOLD, 16), fill=FG)
    for r, (nm, col) in enumerate((("LANCZOS", DIM), ("REDETAIL", FG))):
        d.text((PAD, head + r * (ch + lab + g) + 2), nm, font=font(BLACK, 19), fill=col)
    d.text((PAD, H - foot + 14), "Watch whether the invented texture stays put across the row. "
           "Detail that changes every frame is the failure mode on motion, not softness.",
           font=font(REG, 14), fill=DIM)
    card.save(dst)
    clean.save(dst.replace(".png", "_clean.png"))
    for f in (".fs_a.png", ".fs_b.png", ".fs_lz.png"):
        if os.path.exists(f"{OUT}/{f}"):
            os.remove(f"{OUT}/{f}")


def build_scene(name, steps, t, desc, stem, do_video):
    src = f"{REND}/{stem}.mp4"
    tag = f"{name}_{steps}st"
    made = []
    for scale, sw, sh_ in (("1.5", 960, 576), ("2.0", 1280, 768)):
        up = f"{REND}/up_{scale}x/{stem}_{scale}x.mp4"
        if not (os.path.exists(src) and os.path.exists(up)):
            print(f"  skip {tag} @{scale}x — missing input")
            continue
        s_png, u_png = f"{OUT}/.{tag}_src.png", f"{OUT}/.{tag}_{scale}x_re.png"
        grab(src, t, s_png)
        grab(up, t, u_png)
        base_png = f"{OUT}/.{tag}_{scale}x_lanczos.png"
        lanczos(s_png, sw, sh_, base_png)

        base, re = Image.open(base_png), Image.open(u_png)
        # 440x330 was too small to read. At 100% pixels a small window shows you grain but not the
        # STRUCTURE the grain sits on, which is what makes a comparison legible — fronds, hair
        # strands, a lip edge. Bigger window, same 100% scale, nothing else changes.
        cw, ch = CROP
        x, y = pick_roi(Image.open(s_png), sw, sh_, cw, ch)
        x, y = min(x, sw - cw), min(y, sh_ - ch)
        box = (x, y, x + cw, y + ch)
        nf = sh_out("ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                    "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", src) or "?"
        gen = f"MiniMax H3 t2v, 640x384, {nf}f @24fps, {steps} steps"
        if "_spec" in stem:
            gen += ", Spectrum"
        rows = [("SOURCE", gen)] + UPSCALE_SETTINGS + [
            ("OUTPUT", f"{sw}x{sh_}  ({scale}x), single chunk, original audio re-muxed")]
        # Raw crops cached unadorned — the cover grid composes from THESE, not from the finished
        # cards. Reusing the cards meant inheriting their header/footer padding as dead space.
        base.crop(box).save(f"{OUT}/.raw_{tag}_{scale}x_L.png")
        re.crop(box).save(f"{OUT}/.raw_{tag}_{scale}x_R.png")
        dst = f"{OUT}/crop_{tag}_{scale}x.png"
        pair_png(base.crop(box), re.crop(box),
                 "LANCZOS", f"640x384 resampled to {sw}x{sh_}",
                 "REDETAIL", f"{sw}x{sh_} generated",
                 f"{desc} · {steps} steps · {scale}x", dst, settings=rows)
        made.append(dst)
        print(f"  crop_{tag}_{scale}x.png  roi=({x},{y})")

        if name in MOTION:
            fdst = f"{OUT}/strip_{tag}_{scale}x.png"
            filmstrip(src, up, sw, sh_, t, box, f"{desc} · {steps} steps · {scale}x", fdst)
            made.append(fdst)
            print(f"  strip_{tag}_{scale}x.png")

        if do_video:
            v = f"{OUT}/sbs_{tag}_{scale}x.mp4"
            # Left half is the source Lanczos'd to the same size, so the split is honest.
            sh("ffmpeg", "-y", "-v", "error", "-i", src, "-i", up, "-filter_complex",
               f"[0:v]scale={sw}:{sh_}:flags=lanczos,drawtext=fontfile={BOLD}:text='LANCZOS':"
               f"x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=10[l];"
               f"[1:v]drawtext=fontfile={BOLD}:text='REDETAIL {scale}x':x=20:y=20:fontsize=30:"
               f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=10[r];[l][r]hstack",
               "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an", v)
            print(f"  sbs_{tag}_{scale}x.mp4")
            made.append(v)

            # Wipe reveal. `blend` with an all_expr keeps the frame size constant — an animated
            # `crop` feeding `overlay` changes the top layer's dimensions every frame, which some
            # encoders accept and others quietly mangle. X/Y/T are blend's own variables.
            dur = float(sh_out("ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "csv=p=0", up) or 10)
            wv = f"{OUT}/wipe_{tag}_{scale}x.mp4"
            # The divider is drawn INSIDE all_expr, not with drawbox: drawbox's `t` is thickness,
            # so `x='iw*t/dur'` silently evaluates against the line width and never moves.
            # rgb24 makes 255 on every plane a white line; in yuv it would come out green.
            sh("ffmpeg", "-y", "-v", "error", "-i", src, "-i", up, "-filter_complex",
               # gbrp is PINNED, not incidental. Asking for rgb24 let blend renegotiate to planar
               # GBR anyway, and c0/c1/c2 are plane indices — so an orange (255,138,76) written as
               # R,G,B landed in G,B,R and came out cyan. Pin the format, then order the channels
               # to match it: c0=G, c1=B, c2=R.
               f"[0:v]scale={sw}:{sh_}:flags=lanczos,setsar=1,format=gbrp[l];"
               f"[1:v]setsar=1,format=gbrp[r];"
               # Per-channel so the divider is the accent orange. all_expr forces one value across
               # all planes, and a thin neutral line comes back magenta after 4:2:0 subsampling.
               f"[l][r]blend="
               f"c0_expr='if(lt(abs(X-W*T/{dur:.3f}),3),138,if(lt(X,W*T/{dur:.3f}),B,A))':"
               f"c1_expr='if(lt(abs(X-W*T/{dur:.3f}),3),76,if(lt(X,W*T/{dur:.3f}),B,A))':"
               f"c2_expr='if(lt(abs(X-W*T/{dur:.3f}),3),255,if(lt(X,W*T/{dur:.3f}),B,A))',"
               f"drawtext=fontfile={BOLD}:text='REDETAIL':x=20:y=20:fontsize=28:fontcolor=white:"
               f"box=1:boxcolor=black@0.55:boxborderw=10,"
               f"drawtext=fontfile={BOLD}:text='LANCZOS':x=w-tw-20:y=20:fontsize=28:"
               f"fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10[v]",
               "-map", "[v]", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an", wv)
            print(f"  wipe_{tag}_{scale}x.mp4")
            made.append(wv)
    return made


def wipe_segment(left_v, right_v, w, h, start, dur, label, lname, rname, dst,
                 left_needs_scale=True):
    """One wipe segment: `left_v` on the left, `right_v` revealed from the left edge rightwards.

    Trimming happens INSIDE the filter graph with `trim`, not with `-ss`. As an input option `-ss`
    is a keyframe seek that silently lands somewhere else, which desynced a whole trailer once;
    here it would also slide the two inputs against each other by different amounts.
    """
    lscale = f"scale={w}:{h}:flags=lanczos," if left_needs_scale else ""
    sh("ffmpeg", "-y", "-v", "error", "-i", left_v, "-i", right_v, "-filter_complex",
       f"[0:v]trim=start={start}:duration={dur},setpts=PTS-STARTPTS,{lscale}setsar=1,"
       f"format=gbrp[l];"
       f"[1:v]trim=start={start}:duration={dur},setpts=PTS-STARTPTS,scale={w}:{h},setsar=1,"
       f"format=gbrp[r];"
       # gbrp pinned and channels ordered G,B,R — see the note on the single-clip wipe.
       f"[l][r]blend="
       f"c0_expr='if(lt(abs(X-W*T/{dur:.3f}),3),138,if(lt(X,W*T/{dur:.3f}),B,A))':"
       f"c1_expr='if(lt(abs(X-W*T/{dur:.3f}),3),76,if(lt(X,W*T/{dur:.3f}),B,A))':"
       f"c2_expr='if(lt(abs(X-W*T/{dur:.3f}),3),255,if(lt(X,W*T/{dur:.3f}),B,A))',"
       f"drawtext=fontfile={BOLD}:text='{rname}':x=20:y=20:fontsize=30:fontcolor=white:"
       f"box=1:boxcolor=black@0.55:boxborderw=10,"
       f"drawtext=fontfile={BOLD}:text='{lname}':x=w-tw-20:y=20:fontsize=30:fontcolor=white:"
       f"box=1:boxcolor=black@0.55:boxborderw=10,"
       # ffmpeg drawtext takes 0xRRGGBB. 0x4C8AFF is the accent with its bytes reversed, which
       # renders blue instead of orange — easy to write, hard to spot without looking at a frame.
       f"drawtext=fontfile={BLACK}:text='{label}':x=20:y=h-th-20:fontsize=26:fontcolor=0xFF8A4C:"
       f"box=1:boxcolor=black@0.55:boxborderw=10[v]",
       "-map", "[v]", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-r", "24",
       "-an", dst)


def concat(parts, dst):
    lst = f"{OUT}/.concat.txt"
    open(lst, "w").write("".join(f"file '{os.path.abspath(p)}'\n" for p in parts))
    sh("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
       "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst)
    os.remove(lst)
    for p in parts:
        os.remove(p)


# Reel order: ACTION FIRST. It is the strongest opener and the only clip that answers the question
# a video upscaler is actually judged on — whether fast motion survives.
REEL = [("action", 25, "MOTOCROSS"), ("portrait", 15, "PORTRAIT"), ("forest", 15, "FOREST"),
        ("rain", 15, "RAIN"), ("rust", 15, "RUST")]
SEG = 6.0          # per scene. Long enough for the sweep to read, short enough to hold attention.


def reel_lanczos(scale="2.0", w=1280, h=768):
    """Lanczos vs ReDetail, every scene in a row, one file."""
    parts = []
    for name, steps, label in REEL:
        stem = next((s[4] for s in SCENES if s[0] == name and s[1] == steps), None)
        src, up = f"{REND}/{stem}.mp4", f"{REND}/up_{scale}x/{stem}_{scale}x.mp4"
        if not (stem and os.path.exists(src) and os.path.exists(up)):
            print(f"  reel: skipping {name} (missing input)")
            continue
        p = f"{OUT}/.seg_{name}.mp4"
        wipe_segment(src, up, w, h, 1.0, SEG, f"{label}  ·  {scale}x",
                     "LANCZOS", "REDETAIL", p)
        parts.append(p)
        print(f"  reel segment: {name}")
    if not parts:
        return None
    dst = f"{OUT}/reel_lanczos_vs_redetail_{scale}x.mp4"
    concat(parts, dst)
    print(f"  {os.path.basename(dst)}  ({len(parts)} scenes, {len(parts)*SEG:.0f}s)")
    return dst


def reel_scales(w=1280, h=768):
    """1.5x vs 2.0x, every scene in a row.

    The 1.5x result is 960x576 and is Lanczos-enlarged to 1280x768 so both sides display at the
    same size. That is the honest framing ONLY if you read it as "which do I pick when delivering
    at 1280x768" — it is not 1.5x at its own native size, and the enlargement costs it something.
    Stated on the video itself so nobody has to take it on trust.
    """
    parts = []
    for name, steps, label in REEL:
        stem = next((s[4] for s in SCENES if s[0] == name and s[1] == steps), None)
        a, b = f"{REND}/up_1.5x/{stem}_1.5x.mp4", f"{REND}/up_2.0x/{stem}_2.0x.mp4"
        if not (stem and os.path.exists(a) and os.path.exists(b)):
            print(f"  scale reel: skipping {name} (missing input)")
            continue
        p = f"{OUT}/.sseg_{name}.mp4"
        wipe_segment(a, b, w, h, 1.0, SEG, f"{label}  ·  1.5x enlarged to match",
                     "1.5x", "2.0x", p)
        parts.append(p)
        print(f"  scale segment: {name}")
    if not parts:
        return None
    dst = f"{OUT}/reel_1.5x_vs_2.0x.mp4"
    concat(parts, dst)
    print(f"  {os.path.basename(dst)}  ({len(parts)} scenes, {len(parts)*SEG:.0f}s)")
    return dst


def scale_ab(name, steps, t, desc):
    """1.5x vs 2.0x with NO Lanczos anywhere. Two ReDetail outputs, nothing else.

    The earlier version enlarged the 1.5x result with Lanczos so it could sit beside the 2.0x at
    1280x768 — which put resampling blur INTO the number being measured. Here both sides are
    compared at 960x576, the 1.5x's own native size: 1.5x is untouched and 2.0x is downscaled.
    Downscaling discards, it does not invent, and it is what you would actually do to deliver at
    this size. A Lanczos panel is not shown because Lanczos is not one of the options in this
    decision — the choice is between two renders you already paid for.
    """
    from PIL import ImageStat, ImageFilter, ImageChops
    stem = next((s[4] for s in SCENES if s[0] == name and s[1] == steps), None)
    a15, a20 = f"{REND}/up_1.5x/{stem}_1.5x.mp4", f"{REND}/up_2.0x/{stem}_2.0x.mp4"
    if not (stem and os.path.exists(a15) and os.path.exists(a20)):
        print(f"  scale a/b: skipping {name} (missing input)")
        return None
    W, H = 960, 576
    grab(a15, t, f"{OUT}/.a_15.png")
    grab(a20, t, f"{OUT}/.a_20n.png")
    lanczos(f"{OUT}/.a_20n.png", W, H, f"{OUT}/.a_20.png")     # DOWN-scale only

    cw, ch = CROP
    x, y = pick_roi(Image.open(f"{OUT}/.a_15.png"), W, H, cw, ch)
    x, y = min(x, W - cw), min(y, H - ch)
    box = (x, y, x + cw, y + ch)
    c15 = Image.open(f"{OUT}/.a_15.png").crop(box).convert("RGB")
    c20 = Image.open(f"{OUT}/.a_20.png").crop(box).convert("RGB")

    def energy(im):
        g = im.convert("L")
        return ImageStat.Stat(ImageChops.difference(
            g, g.filter(ImageFilter.GaussianBlur(1.2)))).mean[0]
    e15, e20 = energy(c15), energy(c20)
    dd = ImageStat.Stat(ImageChops.difference(c15, c20)).mean[0]
    AMP = 8
    panels = [(c15, "1.5x", "960x576 native, untouched"),
              (c20, "2.0x", "1280x768 downscaled to 960x576"),
              (ImageChops.difference(c15, c20).point(lambda v: min(255, v * AMP)),
               "DIFFERENCE", f"amplified {AMP}x")]

    n, pad, gap, head, lab = len(panels), 24, 12, 118, 46
    foot = 14 + 26 + (1 + len(UPSCALE_SETTINGS)) * 19 + 8 + 20
    Wc, Hc = pad * 2 + cw * n + gap * (n - 1), head + lab + ch + foot
    card = Image.new("RGB", (Wc, Hc), BG)
    for i, (im, _, _) in enumerate(panels):
        card.paste(im, (pad + i * (cw + gap), head + lab))
    clean = card.copy()
    d = ImageDraw.Draw(card)
    ft = font(BLACK, 30)
    d.text((pad, 22), "ReDetail", font=ft, fill=ACCENT)
    d.text((pad + d.textlength("ReDetail", font=ft) + 22, 29),
           f"{desc} · {steps} steps · which scale?", font=font(REG, 17), fill=DIM)
    fit(d, f"Both compared at 960x576. No Lanczos on either side.    detail energy  1.5x "
        f"{e15:.2f}  vs  2.0x {e20:.2f}  ({(e20/e15-1)*100:+.0f}%)    mean |delta| {dd:.1f}/255",
        BOLD, 16, Wc - pad * 2, (pad, 62), FG)
    fit(d, "Measurably different, but at this magnitude the two are hard to tell apart by eye. "
        "Detail energy is density, not quality — more invented texture is not automatically "
        "better, least of all on skin.", REG, 14, Wc - pad * 2, (pad, 86), DIM)
    for i, (_, tt, sub) in enumerate(panels):
        px = pad + i * (cw + gap)
        d.text((px, head), tt, font=font(BLACK, 21), fill=FG)
        d.text((px, head + 26), sub, font=font(REG, 14), fill=DIM)
    settings_box(d, pad, head + lab + ch + 14, Wc - pad * 2,
                 [("SOURCE", f"MiniMax H3 t2v, 640x384, {steps} steps")] + UPSCALE_SETTINGS)
    dst = f"{OUT}/scale_{name}_{steps}st.png"
    card.save(dst)
    clean.save(dst.replace(".png", "_clean.png"))
    for f in (".a_15.png", ".a_20.png", ".a_20n.png"):
        if os.path.exists(f"{OUT}/{f}"):
            os.remove(f"{OUT}/{f}")
    print(f"  scale_{name}_{steps}st.png  1.5x {e15:.2f} vs 2.0x {e20:.2f} "
          f"({(e20/e15-1)*100:+.0f}%), |d|={dd:.1f}")
    return dst


def scale_toggle(name, steps, t, desc, hold=0.8, cycles=5):
    """A/B flicker between 1.5x and 2.0x on ONE frozen frame and ONE crop.

    Toggling in place is how you compare two nearly-identical images: the eye is poor at absolute
    sharpness and very good at spotting what MOVES between two states. The frame is frozen on
    purpose — flickering live video just reads as video.
    """
    stem = next((s[4] for s in SCENES if s[0] == name and s[1] == steps), None)
    src = f"{REND}/{stem}.mp4"
    a15, a20 = f"{REND}/up_1.5x/{stem}_1.5x.mp4", f"{REND}/up_2.0x/{stem}_2.0x.mp4"
    if not all(os.path.exists(p) for p in (src, a15, a20)):
        print(f"  toggle: skipping {name} (missing input)")
        return None
    W, H = 1280, 768
    s_png = f"{OUT}/.t_s.png"
    grab(src, t, s_png)
    grab(a15, t, f"{OUT}/.t_15n.png")
    grab(a20, t, f"{OUT}/.t_20.png")
    lanczos(f"{OUT}/.t_15n.png", W, H, f"{OUT}/.t_15.png")

    cw, ch = CROP
    x, y = pick_roi(Image.open(s_png), W, H, cw, ch)
    x, y = min(x, W - cw), min(y, H - ch)
    box = (x, y, x + cw, y + ch)
    frames = []
    for key, lbl in (("15", "REDETAIL 1.5x"), ("20", "REDETAIL 2.0x")):
        im = Image.open(f"{OUT}/.t_{key}.png").crop(box).convert("RGB")
        card = Image.new("RGB", (cw, ch + 92), BG)
        card.paste(im, (0, 72))
        d = ImageDraw.Draw(card)
        d.text((14, 12), lbl, font=font(BLACK, 26), fill=ACCENT if key == "20" else FG)
        d.text((14, 44), f"{desc} · same frame, same crop, 100% pixels",
               font=font(REG, 14), fill=DIM)
        p = f"{OUT}/.tog_{key}.png"
        card.save(p)
        frames.append(p)

    lst = f"{OUT}/.tog.txt"
    with open(lst, "w") as fh:
        for _ in range(cycles):
            for p in frames:
                fh.write(f"file '{os.path.abspath(p)}'\nduration {hold}\n")
        fh.write(f"file '{os.path.abspath(frames[-1])}'\n")
    dst = f"{OUT}/toggle_{name}_{steps}st_1.5x_vs_2.0x.mp4"
    sh("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
       "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-crf", "14",
       "-movflags", "+faststart", dst)
    for f in (lst, *frames, f"{OUT}/.t_s.png", f"{OUT}/.t_15.png", f"{OUT}/.t_15n.png",
              f"{OUT}/.t_20.png"):
        if os.path.exists(f):
            os.remove(f)
    print(f"  {os.path.basename(dst)}")
    return dst


def cover():
    """CivitAI cover: 2x2 grid, each tile a Lanczos|ReDetail split of one scene.

    Composed from the cached raw crops so there is no inherited padding, and each tile is
    half-and-half of a single crop — the split line is the comparison, no gap between halves.
    """
    caps = {"forest": "ferns, backlit", "portrait": "skin + hair",
            "rain": "rain on glass", "rust": "rusted metal", "action": "motocross, dirt"}
    picks = [(f"{n}_{st}st", sc, f"{caps.get(n, n)} · {st} steps") for n, st, sc in HERO]
    tiles = []
    for tag, sc, cap in picks:
        L, R = f"{OUT}/.raw_{tag}_{sc}x_L.png", f"{OUT}/.raw_{tag}_{sc}x_R.png"
        if not (os.path.exists(L) and os.path.exists(R)):
            continue
        l, r = Image.open(L), Image.open(R)
        w, h = l.size
        half = w // 2
        t = Image.new("RGB", (w, h))
        t.paste(l.crop((0, 0, half, h)), (0, 0))
        t.paste(r.crop((half, 0, w, h)), (half, 0))
        ImageDraw.Draw(t).line([(half, 0), (half, h)], fill=ACCENT, width=2)
        tiles.append((t, cap))
    if len(tiles) < 4:
        print("  cover: need 4 raw crop pairs, have", len(tiles))
        return None

    tw, th = tiles[0][0].size
    head, capH, G = 132, 34, 16
    W, H = tw * 2 + G * 3, head + (th + capH) * 2 + G * 3
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for i, (t, cap) in enumerate(tiles):
        x = G + (i % 2) * (tw + G)
        y = head + (i // 2) * (th + capH + G)
        img.paste(t, (x, y))
        d.rectangle([x - 1, y - 1, x + tw, y + th], outline=(48, 48, 52))
        d.text((x, y + th + 8), cap, font=font(REG, 17), fill=DIM)
    ft = font(BLACK, 48)
    d.text((G, 16), "ReDetail", font=ft, fill=ACCENT)
    tagx = G + d.textlength("ReDetail", font=ft) + 22
    fit(d, "LTX-2.5 video upscaler", REG, 24, W - tagx - G, (tagx, 34), DIM)
    fit(d, "it synthesizes detail — it does not interpolate", REG, 19, W - G * 2, (G, 72), DIM)
    fit(d, "each tile: LEFT half Lanczos  ·  RIGHT half ReDetail  ·  identical output size, "
        "2.0x, 100% pixels", BOLD, 17, W - G * 2, (G, 100), FG)
    p = f"{OUT}/cover.png"
    img.save(p)
    print(f"  cover.png {W}x{H}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--crop", default=None, help="100%% crop window, e.g. 600x450")
    ap.add_argument("--reels", action="store_true", help="build the two reels and nothing else")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.crop:
        CROP = tuple(int(v) for v in a.crop.lower().split("x"))

    if a.reels:
        print("reel: Lanczos vs ReDetail")
        reel_lanczos()
        # NOT a wipe for the scale comparison — a wipe compares across a moving boundary, which
        # cannot show a small difference that is spread evenly over the whole frame.
        print("scale comparison: ladders + toggles")
        for nm, st, t, desc, _ in SCENES:
            if (nm, st) in [(r[0], r[1]) for r in REEL]:
                scale_ab(nm, st, t, desc)
                scale_toggle(nm, st, t, desc)
        sys.exit(0)

    todo = [s for s in SCENES if not a.scene or s[0] == a.scene]
    for name, steps, t, desc, stem in todo:
        print(f"{name} {steps}st @{t}s")
        build_scene(name, steps, t, desc, stem, not a.no_video)

    print("cover")
    cover()
    title_card("LTX-2.5 pixel upscaler", "one command, any ComfyUI", 1280, 720,
               f"{OUT}/title.png")
    print(f"\n-> {OUT}")
