"""
DroneScanner - Photogrammetry Tool
====================================
Warm UI + live visualisation of every calculation step

pip install opencv-python numpy open3d matplotlib
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import time
import math

cv2 = np = o3d = None
plt = FigureCanvasTkAgg = Figure = None

def lazy_import():
    global cv2, np, o3d, plt, FigureCanvasTkAgg, Figure
    import cv2 as _cv2
    import numpy as _np
    import open3d as _o3d
    import matplotlib.pyplot as _plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FCA
    from matplotlib.figure import Figure as _Fig
    cv2, np, o3d = _cv2, _np, _o3d
    plt, FigureCanvasTkAgg, Figure = _plt, _FCA, _Fig


# ══════════════════════════════════════════
#  WARM PALETTE
# ══════════════════════════════════════════
BG        = "#140e08"
SURFACE   = "#1e1510"
CARD      = "#27190f"
CARD2     = "#2f1f12"
BORDER    = "#4a2e18"
ACCENT    = "#e8935a"
ACCENT2   = "#f0c060"
TEXT      = "#f5e6c8"
MUTED     = "#9a7a5a"
GREEN_W   = "#98c88a"
HIGHLIGHT = "#3d2515"

MPL_BG    = "#1c1208"
MPL_AX    = "#221608"
MPL_GRID  = "#3a2410"
MPL_TEXT  = "#c8a878"

STEPS = [
    ("load images"),
    ("detect features"),
    ("match pairs"),
    ("reconstruct 3D"),
    ("save cloud"),
]
STAGE_MAP = {
    "loading":   0,
    "keypoints": 1,
    "matches":   2,
    "cloud":     3,
}


# ══════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════

def load_images(folder, log):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    paths = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])
    log(f"  found {len(paths)} images in folder")
    if len(paths) < 3:
        raise ValueError("Need at least 3 images!")
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            imgs.append((p, img))
    return imgs


def detect_features(imgs, log, on_keypoints=None):
    sift = cv2.SIFT_create(nfeatures=3000)
    kp_desc = []
    for i, (path, img) in enumerate(imgs):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        kp_desc.append((kp, des))
        log(f"  [{i+1}/{len(imgs)}]  {os.path.basename(path)}")
        log(f"         {len(kp):,} keypoints detected")
        if on_keypoints:
            on_keypoints(img, kp, i, len(imgs))
    return kp_desc


def match_features(kp_desc, imgs, log, on_match=None):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    n  = len(kp_desc)

    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    log(f"  {len(all_pairs):,} pairs to match")

    pairs = []
    lock  = threading.Lock()

    match_counter = [0]

    def match_pair(args):
        i, j = args
        des1, des2 = kp_desc[i][1], kp_desc[j][1]
        if des1 is None or des2 is None:
            return
        raw  = bf.knnMatch(des1, des2, k=2)
        good = [m for m, nn in raw if m.distance < 0.75 * nn.distance]
        if len(good) >= 8:
            with lock:
                pairs.append((i, j, good))
                match_counter[0] += 1
                cnt = match_counter[0]
            log(f"  pair ({i+1},{j+1})  →  {len(good):,} good matches")
            # only redraw viz every 20 matches to avoid choking the GUI
            if on_match and cnt % 20 == 1:
                on_match(imgs[i][1], imgs[j][1],
                         kp_desc[i][0], kp_desc[j][0], good)

    workers = max(1, __import__('os').cpu_count() - 1)
    log(f"  running on {workers} CPU cores in parallel")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(match_pair, all_pairs))

    pairs.sort(key=lambda x: (x[0], x[1]))
    log(f"  total: {len(pairs)} usable pairs")
    return pairs


def estimate_camera(img_shape):
    h, w = img_shape[:2]
    f = max(w, h) * 1.2
    return np.array([[f, 0, w/2],
                     [0, f, h/2],
                     [0, 0,   1]], dtype=np.float64)


def triangulate_gpu(P1, P2, pts1, pts2, xp):
    """Triangulate points using GPU (cupy) or CPU (numpy) depending on xp."""
    # build 4x4 DLT system per point on GPU in batch
    n   = pts1.shape[1]
    A   = xp.zeros((n, 4, 4), dtype=xp.float64)
    A[:, 0] = pts1[0:1].T * P1[2] - P1[0]
    A[:, 1] = pts1[1:2].T * P1[2] - P1[1]
    A[:, 2] = pts2[0:1].T * P2[2] - P2[0]
    A[:, 3] = pts2[1:2].T * P2[2] - P2[1]
    # SVD per point — last singular vector is the solution
    _, _, Vt = xp.linalg.svd(A)
    X = Vt[:, -1, :]          # (n, 4)
    X = X / X[:, 3:4]
    return X[:, :3]            # (n, 3)


def reconstruct(kp_desc, pairs, imgs, log, on_points=None):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    # try to use GPU via cupy, fall back to numpy
    try:
        import cupy as xp
        xp.zeros(1)  # test GPU is accessible
        gpu = True
        log("  🎮  GPU detected — using RTX 4070 for triangulation")
    except Exception:
        xp = np
        gpu = False
        log("  💻  GPU not available — using CPU")

    K = estimate_camera(imgs[0][1].shape)
    log(f"  camera intrinsics  (focal={K[0,0]:.1f}px)")
    log(f"  K = [{K[0,0]:.0f}  0  {K[0,2]:.0f}]")
    log(f"      [  0  {K[1,1]:.0f}  {K[1,2]:.0f}]")
    log(f"      [  0    0    1  ]")

    # upload K to GPU once
    K_xp = xp.array(K)

    pts3d_all = []
    col_all   = []
    lock      = threading.Lock()
    counter   = [0]

    def process_pair(args):
        idx, (i, j, good) = args
        kp1, _ = kp_desc[i]
        kp2, _ = kp_desc[j]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

        # essential matrix on CPU (opencv doesnt support GPU here)
        E, mask = cv2.findEssentialMat(
            src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None:
            return

        _, R, t, mask2 = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)
        angle = math.degrees(math.acos(max(-1, min(1, (np.trace(R) - 1) / 2))))

        inliers = mask2.ravel() == 255
        if inliers.sum() < 8:
            return

        # build projection matrices on GPU
        P1_xp = K_xp @ xp.array(np.hstack([np.eye(3),   np.zeros((3,1))]))
        P2_xp = K_xp @ xp.array(np.hstack([R,            t              ]))

        # upload inlier points to GPU and triangulate there
        s = xp.array(src_pts[inliers].T.astype(np.float64))  # (2, n)
        d = xp.array(dst_pts[inliers].T.astype(np.float64))  # (2, n)

        pts3d_gpu = triangulate_gpu(P1_xp, P2_xp, s, d, xp)

        # pull back to CPU
        pts3d = pts3d_gpu.get() if gpu else pts3d_gpu

        # colour lookup on CPU
        img_bgr = imgs[i][1]
        h, w    = img_bgr.shape[:2]
        colours = []
        for (px, py) in src_pts[inliers]:
            px = int(np.clip(px, 0, w - 1))
            py = int(np.clip(py, 0, h - 1))
            b, g, r = img_bgr[py, px]
            colours.append([r/255.0, g/255.0, b/255.0])

        depths = pts3d[:, 2]
        valid  = depths > 0
        if not valid.any():
            return
        keep = valid & (depths < np.percentile(depths[valid], 95))

        batch_pts = pts3d[keep]
        batch_col = np.array(colours)[keep]

        with lock:
            pts3d_all.append(batch_pts)
            col_all.append(batch_col)
            counter[0] += 1
            cnt = counter[0]

        log(f"  pair {idx+1}: {angle:.1f}°  {len(batch_pts):,} pts  ({mask.sum()} inliers)")

        if on_points and cnt % 50 == 1:
            with lock:
                if pts3d_all:
                    on_points(np.vstack(pts3d_all), np.vstack(col_all))

    workers = max(1, __import__('os').cpu_count() - 1)
    log(f"  reconstructing {len(pairs)} pairs on {workers} CPU cores + GPU triangulation…")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(process_pair, enumerate(pairs)))

    if not pts3d_all:
        raise RuntimeError("Reconstruction failed — need more overlapping photos!")

    log(f"  merged {len(pts3d_all)} batches into final cloud")
    return np.vstack(pts3d_all), np.vstack(col_all)


def save_cloud(pts3d, colours, out_path, log):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colours, 0, 1))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(out_path, pcd)
    log(f"  saved  →  {out_path}")
    log(f"  {len(pcd.points):,} points after outlier cleanup")
    return pcd


def open_viewer(pcd):
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="DroneScanner — 3D Point Cloud",
        width=1200, height=800
    )


def run_pipeline(folder, out_path, log, vis, done_cb, err_cb):
    try:
        lazy_import()
        t0 = time.time()

        log("─── loading images ───────────────────────")
        vis.set_stage("loading")
        imgs = load_images(folder, log)

        log("\n─── detecting features ───────────────────")
        vis.set_stage("keypoints")
        kp_desc = detect_features(imgs, log, on_keypoints=vis.show_keypoints)

        log("\n─── matching features ────────────────────")
        vis.set_stage("matches")
        pairs = match_features(kp_desc, imgs, log, on_match=vis.show_matches)

        log("\n─── 3D reconstruction ────────────────────")
        vis.set_stage("cloud")
        pts3d, colours = reconstruct(
            kp_desc, pairs, imgs, log, on_points=vis.show_cloud
        )

        log("\n─── saving ───────────────────────────────")
        pcd = save_cloud(pts3d, colours, out_path, log)

        log(f"\n  done in {time.time()-t0:.1f}s")
        done_cb(out_path, pcd)

    except Exception as e:
        import traceback
        err_cb(str(e), traceback.format_exc())


# ══════════════════════════════════════════
#  LIVE VIS PANEL
# ══════════════════════════════════════════

class VisPanel:
    def __init__(self, parent):
        self.parent = parent
        self.fig    = Figure(figsize=(5.8, 5), facecolor=MPL_BG)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.06)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_idle()

    def _warm_ax(self, ax, title=""):
        ax.set_facecolor(MPL_AX)
        ax.tick_params(colors=MPL_TEXT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(MPL_GRID)
        ax.set_title(title, color=ACCENT2, fontsize=9, pad=6,
                     fontfamily="monospace")

    def _draw_idle(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._warm_ax(ax, "waiting…")
        ax.text(0.5, 0.5, "🛸", fontsize=52,
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.34, "live calculations will appear here",
                ha="center", va="center", transform=ax.transAxes,
                color=MUTED, fontsize=9, fontfamily="monospace")
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw_idle()

    def set_stage(self, stage):
        pass  # stepbar handles this

    def show_keypoints(self, img_bgr, keypoints, idx, total):
        def _draw():
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            self._warm_ax(
                ax,
                f"keypoint detection  [{idx+1}/{total}]  —  {len(keypoints):,} features"
            )
            rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w  = rgb.shape[:2]
            scale = min(1.0, 900 / max(h, w))
            small = cv2.resize(rgb, (int(w*scale), int(h*scale)))
            ax.imshow(small, aspect="auto")
            sample = keypoints[::max(1, len(keypoints)//300)]
            xs = [kp.pt[0]*scale for kp in sample]
            ys = [kp.pt[1]*scale for kp in sample]
            ax.scatter(xs, ys, c=ACCENT, s=6, alpha=0.75,
                       linewidths=0, zorder=3)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw_idle()

        self.parent.after(0, _draw)

    def show_matches(self, img1, img2, kp1, kp2, good_matches):
        def _draw():
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            self._warm_ax(
                ax,
                f"feature matching  —  {len(good_matches):,} matches"
            )
            rgb1  = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            rgb2  = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            scale = min(1.0, 420 / max(rgb1.shape[:2]))
            s1    = cv2.resize(rgb1, (int(rgb1.shape[1]*scale), int(rgb1.shape[0]*scale)))
            s2    = cv2.resize(rgb2, (int(rgb2.shape[1]*scale), int(rgb2.shape[0]*scale)))
            h1, w1 = s1.shape[:2]
            h2, w2 = s2.shape[:2]
            canvas = np.zeros((max(h1,h2), w1+w2, 3), dtype=np.uint8)
            canvas[:h1, :w1]      = s1
            canvas[:h2, w1:w1+w2] = s2
            ax.imshow(canvas, aspect="auto")
            for m in good_matches[::max(1, len(good_matches)//80)]:
                x1, y1 = kp1[m.queryIdx].pt
                x2, y2 = kp2[m.trainIdx].pt
                ax.plot([x1*scale, x2*scale+w1], [y1*scale, y2*scale],
                        color=GREEN_W, alpha=0.3, linewidth=0.6)
                ax.plot(x1*scale, y1*scale, 'o',
                        color=ACCENT, markersize=2, markeredgewidth=0)
                ax.plot(x2*scale+w1, y2*scale, 'o',
                        color=ACCENT2, markersize=2, markeredgewidth=0)
            ax.axvline(w1, color=BORDER, linewidth=1.5, alpha=0.9)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw_idle()

        self.parent.after(0, _draw)

    def show_cloud(self, pts3d, colours):
        def _draw():
            self.fig.clear()
            ax = self.fig.add_subplot(111, projection="3d")
            ax.set_facecolor(MPL_AX)
            ax.set_title(
                f"3D reconstruction  —  {len(pts3d):,} points",
                color=ACCENT2, fontsize=9, pad=6, fontfamily="monospace"
            )
            step = max(1, len(pts3d) // 5000)
            dp   = pts3d[::step]
            dc   = np.clip(colours[::step], 0, 1)
            ax.scatter(dp[:,0], dp[:,1], dp[:,2],
                       c=dc, s=0.9, alpha=0.75, linewidths=0)
            ax.set_xlabel("X", color=MUTED, fontsize=7)
            ax.set_ylabel("Y", color=MUTED, fontsize=7)
            ax.set_zlabel("Z", color=MUTED, fontsize=7)
            ax.tick_params(colors=MPL_TEXT, labelsize=6)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor(MPL_GRID)
            self.fig.patch.set_facecolor(MPL_BG)
            self.canvas.draw_idle()

        self.parent.after(0, _draw)


# ══════════════════════════════════════════
#  STEP BAR
# ══════════════════════════════════════════

class StepBar(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        self.dots   = []
        self.labels = []
        self.lines  = []
        self._build()

    def _build(self):
        for i, label in enumerate(STEPS):
            col = tk.Frame(self, bg=SURFACE)
            col.pack(side="left", expand=True)
            dot = tk.Label(col, text="○", font=("Consolas", 13),
                           bg=SURFACE, fg=MUTED)
            dot.pack()
            lbl = tk.Label(col, text=label, font=("Consolas", 7),
                           bg=SURFACE, fg=MUTED)
            lbl.pack()
            self.dots.append(dot)
            self.labels.append(lbl)
            if i < len(STEPS) - 1:
                ln = tk.Label(self, text="──", font=("Consolas", 9),
                              bg=SURFACE, fg=MUTED)
                ln.pack(side="left", expand=True)
                self.lines.append(ln)

    def set_step(self, idx):
        for i, (dot, lbl) in enumerate(zip(self.dots, self.labels)):
            if i < idx:
                dot.config(text="●", fg=GREEN_W)
                lbl.config(fg=GREEN_W)
                if i < len(self.lines):
                    self.lines[i].config(fg=GREEN_W)
            elif i == idx:
                dot.config(text="◉", fg=ACCENT)
                lbl.config(fg=ACCENT)
            else:
                dot.config(text="○", fg=MUTED)
                lbl.config(fg=MUTED)


# ══════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DroneScanner")
        self.configure(bg=BG)
        self.minsize(1080, 680)
        self.resizable(True, True)
        self.folder_var = tk.StringVar()
        self.out_var    = tk.StringVar()
        self.status_var = tk.StringVar(value="ready")
        self.timer_var  = tk.StringVar(value="")
        self.running    = False
        self._pcd       = None
        self.vis        = None
        self._timer_start = None
        self._timer_job   = None
        self._build()
        self._center()

    def _build(self):
        # header
        hdr = tk.Frame(self, bg=SURFACE, padx=28, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🛸  DroneScanner",
                 font=("Georgia", 19, "bold"), bg=SURFACE, fg=TEXT).pack(side="left")
        tk.Label(hdr, text="photogrammetry · structure from motion",
                 font=("Consolas", 9), bg=SURFACE, fg=MUTED).pack(
                     side="left", padx=18, pady=(4,0))
        tk.Label(hdr, textvariable=self.status_var,
                 font=("Consolas", 9), bg=SURFACE, fg=ACCENT).pack(side="right")
        tk.Label(hdr, textvariable=self.timer_var,
                 font=("Consolas", 9), bg=SURFACE, fg=ACCENT2).pack(side="right", padx=(0,16))

        # step bar
        self.stepbar = StepBar(self, pady=8)
        self.stepbar.pack(fill="x", padx=28)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # paned window
        panes = tk.PanedWindow(self, orient="horizontal", bg=BORDER,
                               sashwidth=4, bd=0)
        panes.pack(fill="both", expand=True)

        self._left_frame = tk.Frame(panes, bg=MPL_BG)
        panes.add(self._left_frame, minsize=480)

        right = tk.Frame(panes, bg=BG, padx=20, pady=16)
        panes.add(right, minsize=340)
        self._build_controls(right)

        # draw idle placeholder right away
        self._init_vis_placeholder()

    def _init_vis_placeholder(self):
        # show placeholder without importing matplotlib
        lbl = tk.Label(
            self._left_frame,
            text="🛸\n\nlive calculations\nwill appear here",
            font=("Georgia", 13), bg=MPL_BG, fg=MUTED,
            justify="center"
        )
        lbl.place(relx=0.5, rely=0.5, anchor="center")
        self._placeholder_lbl = lbl

    def _build_controls(self, parent):
        # run btn — always at top so it's always visible
        self.run_btn = tk.Button(
            parent, text="▶  run scan",
            font=("Georgia", 13, "bold"),
            bg=ACCENT, fg=BG,
            activebackground=ACCENT2, activeforeground=BG,
            relief="flat", bd=0, padx=0, pady=12,
            cursor="hand2", command=self._start
        )
        self.run_btn.pack(fill="x", pady=(0, 12))

        # view btn (hidden until done)
        self.view_btn = tk.Button(
            parent, text="🔍  open 3D viewer",
            font=("Georgia", 11),
            bg=CARD2, fg=ACCENT2,
            activebackground=HIGHLIGHT, activeforeground=ACCENT2,
            relief="flat", bd=0, padx=0, pady=8,
            cursor="hand2", command=self._open_viewer
        )

        self._input_card(parent, "📂  image folder",
                         "folder containing your drone photos",
                         self.folder_var, self._pick_folder)
        self._input_card(parent, "💾  output .ply",
                         "where to save the 3D point cloud",
                         self.out_var, self._pick_output)

        # tips
        tips = tk.Frame(parent, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1,
                        padx=14, pady=10)
        tips.pack(fill="x", pady=(0, 14))
        tk.Label(tips, text="tips for a good scan",
                 font=("Georgia", 10, "italic"), bg=CARD, fg=ACCENT2).pack(anchor="w")
        for tip in [
            "50–200 photos  ·  >60% overlap",
            "avoid reflective or textureless surfaces",
            "fly a grid or orbit the subject",
            "consistent lighting, no harsh shadows",
        ]:
            tk.Label(tips, text=f"  · {tip}",
                     font=("Consolas", 8), bg=CARD, fg=MUTED).pack(anchor="w", pady=1)

        # log
        tk.Label(parent, text="calculation log",
                 font=("Georgia", 10, "italic"), bg=BG, fg=ACCENT2).pack(
                     anchor="w", pady=(0, 4))
        log_frame = tk.Frame(parent, bg=CARD,
                             highlightbackground=BORDER, highlightthickness=1)
        log_frame.pack(fill="both", expand=True)
        sb = tk.Scrollbar(log_frame, bg=CARD, troughcolor=CARD)
        sb.pack(side="right", fill="y")
        self.log_box = tk.Text(
            log_frame, bg="#110c06", fg="#c8a060",
            font=("Consolas", 8), relief="flat", bd=0,
            padx=10, pady=8,
            yscrollcommand=sb.set, state="disabled", wrap="word"
        )
        self.log_box.pack(fill="both", expand=True)
        sb.config(command=self.log_box.yview)

    def _input_card(self, parent, title, subtitle, var, cmd):
        card = tk.Frame(parent, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1,
                        padx=14, pady=10)
        card.pack(fill="x", pady=(0, 10))
        top = tk.Frame(card, bg=CARD)
        top.pack(fill="x")
        tk.Label(top, text=title,
                 font=("Georgia", 10, "bold"), bg=CARD, fg=TEXT).pack(side="left")
        tk.Button(top, text="browse",
                  font=("Consolas", 8),
                  bg=HIGHLIGHT, fg=ACCENT,
                  activebackground=BORDER, activeforeground=ACCENT2,
                  relief="flat", bd=0, padx=8, pady=3,
                  cursor="hand2", command=cmd).pack(side="right")
        tk.Label(card, text=subtitle,
                 font=("Consolas", 8), bg=CARD, fg=MUTED).pack(anchor="w")
        tk.Label(card, textvariable=var,
                 font=("Consolas", 8), bg=CARD, fg=ACCENT2,
                 wraplength=360, justify="left").pack(anchor="w")

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if folder:
            self.folder_var.set(folder)
            if not self.out_var.get():
                self.out_var.set(os.path.join(folder, "pointcloud.ply"))

    def _pick_output(self):
        path = filedialog.asksaveasfilename(
            title="Save point cloud",
            defaultextension=".ply",
            filetypes=[("Point Cloud", "*.ply"), ("All", "*.*")]
        )
        if path:
            self.out_var.set(path)

    def _start(self):
        if self.running:
            return
        folder = self.folder_var.get().strip()
        out    = self.out_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("no folder", "select a valid image folder first")
            return
        if not out:
            messagebox.showerror("no output", "choose an output path")
            return

        # init vis panel on first run
        if self.vis is None:
            lazy_import()
            if hasattr(self, '_placeholder_lbl'):
                self._placeholder_lbl.destroy()
            self.vis = VisPanel(self._left_frame)

        self.running = True
        self.run_btn.config(state="disabled", bg=MUTED, fg=SURFACE)
        self.view_btn.pack_forget()
        self.status_var.set("processing…")
        self._clear_log()
        self.stepbar.set_step(0)
        self._start_timer()

        threading.Thread(
            target=self._thread_run,
            args=(folder, out),
            daemon=True
        ).start()

    def _thread_run(self, folder, out):
        app = self

        class VisProxy:
            def set_stage(self, stage):
                idx = STAGE_MAP.get(stage, 0)
                app.after(0, lambda: app.stepbar.set_step(idx))

            def show_keypoints(self, *a, **kw):
                app.vis.show_keypoints(*a, **kw)

            def show_matches(self, *a, **kw):
                app.vis.show_matches(*a, **kw)

            def show_cloud(self, *a, **kw):
                app.vis.show_cloud(*a, **kw)

        run_pipeline(folder, out, app._write, VisProxy(),
                     app._done, app._err)

    def _done(self, out_path, pcd):
        self._pcd = pcd
        def update():
            self.running = False
            self._stop_timer()
            self.run_btn.config(state="normal", bg=ACCENT, fg=BG)
            self.stepbar.set_step(len(STEPS) - 1)
            self.status_var.set("✓  done")
            self.view_btn.pack(fill="x", pady=(8, 0))
        self.after(0, update)

    def _err(self, msg, tb=""):
        def update():
            self.running = False
            self._stop_timer()
            self.run_btn.config(state="normal", bg=ACCENT, fg=BG)
            self.status_var.set("✗  error")
            self._write(f"\n✗  {msg}")
            if tb:
                self._write(tb)
            messagebox.showerror("error", msg)
        self.after(0, update)

    def _open_viewer(self):
        if self._pcd is not None:
            threading.Thread(target=open_viewer,
                             args=(self._pcd,), daemon=True).start()

    def _start_timer(self):
        import time as _time
        self._timer_start = _time.time()
        self._tick_timer()

    def _tick_timer(self):
        if self._timer_start is None:
            return
        import time as _time
        elapsed = int(_time.time() - self._timer_start)
        m, s = divmod(elapsed, 60)
        self.timer_var.set(f"⏱  {m:02d}:{s:02d}")
        self._timer_job = self.after(1000, self._tick_timer)

    def _stop_timer(self):
        if self._timer_job:
            self.after_cancel(self._timer_job)
            self._timer_job = None

    def _write(self, msg):
        def update():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, update)

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    def _center(self):
        self.update_idletasks()
        w, h = 1120, 700
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")


if __name__ == "__main__":
    App().mainloop()
