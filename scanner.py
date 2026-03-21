"""
==============================================
  DroneScanner - Photogrammetry Tool
  powered by OpenCV + Open3D

  INSTALL FIRST (run in terminal):
  pip install opencv-python numpy open3d
==============================================
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading, os, time

cv2 = np = o3d = None

def lazy_import():
    global cv2, np, o3d
    import cv2 as _cv2; import numpy as _np; import open3d as _o3d
    cv2, np, o3d = _cv2, _np, _o3d

def load_images(folder, log):
    exts = (".jpg",".jpeg",".png",".bmp",".tiff")
    paths = sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(exts)])
    log(f"📂  Found {len(paths)} images")
    if len(paths) < 3: raise ValueError("Need at least 3 images!")
    return [(p, cv2.imread(p)) for p in paths if cv2.imread(p) is not None]

def detect_features(imgs, log):
    sift = cv2.SIFT_create(nfeatures=3000)
    result = []
    for i,(path,img) in enumerate(imgs):
        kp,des = sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), None)
        result.append((kp,des))
        log(f"🔍  [{i+1}/{len(imgs)}] {os.path.basename(path)} → {len(kp)} keypoints")
    return result

def match_features(kp_desc, log):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    pairs = []
    for i in range(len(kp_desc)):
        for j in range(i+1, len(kp_desc)):
            d1,d2 = kp_desc[i][1], kp_desc[j][1]
            if d1 is None or d2 is None: continue
            good = [m for m,n in bf.knnMatch(d1,d2,k=2) if m.distance < 0.75*n.distance]
            if len(good) >= 8: pairs.append((i,j,good))
    log(f"🔗  {len(pairs)} pairs matched"); return pairs

def reconstruct(kp_desc, pairs, imgs, log):
    K = (lambda s: np.array([[max(s[1],s[0])*1.2,0,s[1]/2],[0,max(s[1],s[0])*1.2,s[0]/2],[0,0,1]],dtype=np.float64))(imgs[0][1].shape)
    pts3d_all, col_all = [], []
    for (i,j,good) in pairs:
        src = np.float32([kp_desc[i][0][m.queryIdx].pt for m in good])
        dst = np.float32([kp_desc[j][0][m.trainIdx].pt for m in good])
        E,mask = cv2.findEssentialMat(src,dst,K,method=cv2.RANSAC,prob=0.999,threshold=1.0)
        if E is None: continue
        _,R,t,mask2 = cv2.recoverPose(E,src,dst,K,mask=mask)
        inliers = mask2.ravel()==255
        pts4d = cv2.triangulatePoints(K@np.hstack([np.eye(3),np.zeros((3,1))]),K@np.hstack([R,t]),src[inliers].T,dst[inliers].T)
        pts4d /= pts4d[3]; pts3d = pts4d[:3].T
        img_bgr = imgs[i][1]; h,w = img_bgr.shape[:2]
        colours = [[img_bgr[np.clip(int(py),0,h-1),np.clip(int(px),0,w-1)][2]/255,
                    img_bgr[np.clip(int(py),0,h-1),np.clip(int(px),0,w-1)][1]/255,
                    img_bgr[np.clip(int(py),0,h-1),np.clip(int(px),0,w-1)][0]/255]
                   for px,py in src[inliers]]
        d = pts3d[:,2]; valid = d>0
        if not valid.any(): continue
        keep = valid & (d < np.percentile(d[valid],95))
        pts3d_all.append(pts3d[keep]); col_all.append(np.array(colours)[keep])
    if not pts3d_all: raise RuntimeError("Reconstruction failed — need more overlapping photos!")
    log(f"📐  Triangulated from {len(pts3d_all)} pairs")
    return np.vstack(pts3d_all), np.vstack(col_all)

def save_and_view(pts3d, colours, out_path, log):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colours,0,1))
    pcd,_ = pcd.remove_statistical_outlier(20,2.0)
    o3d.io.write_point_cloud(out_path, pcd)
    log(f"💾  Saved → {out_path}"); log(f"✅  {len(pcd.points):,} points")
    log("🖥️  Opening viewer…")
    o3d.visualization.draw_geometries([pcd], window_name="DroneScanner", width=1100, height=700)

def run_pipeline(folder, out_path, log, done_cb, err_cb):
    try:
        lazy_import(); log("🚀  Starting…\n"); t0=time.time()
        imgs=load_images(folder,log); kp=detect_features(imgs,log)
        pairs=match_features(kp,log); pts,col=reconstruct(kp,pairs,imgs,log)
        save_and_view(pts,col,out_path,log); log(f"\n🎉  Done in {time.time()-t0:.1f}s"); done_cb(out_path)
    except Exception as e: err_cb(str(e))

# ── GUI ─────────────────────────────────────────────────────────────────────
BG="#0d0f14"; SURFACE="#161b24"; CARD="#1e2535"; ACCENT="#4f8ef7"
TEXT="#e8eaf0"; MUTED="#6b7394"; GREEN="#3ddc97"; BORDER="#2a3352"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DroneScanner"); self.configure(bg=BG); self.minsize(720,580)
        self.folder_var=tk.StringVar(); self.out_var=tk.StringVar()
        self.status_var=tk.StringVar(value="Idle"); self.running=False
        self._ui(); self._center()

    def _ui(self):
        hdr=tk.Frame(self,bg=SURFACE,pady=18); hdr.pack(fill="x")
        tk.Label(hdr,text="🛸  DroneScanner",font=("Segoe UI",20,"bold"),bg=SURFACE,fg=TEXT).pack()
        tk.Label(hdr,text="Photogrammetry · Point Cloud Generator",font=("Segoe UI",9),bg=SURFACE,fg=MUTED).pack()
        body=tk.Frame(self,bg=BG,padx=28,pady=20); body.pack(fill="both",expand=True)
        self._row(body,"📂  Image Folder","Folder with your drone photos",self.folder_var,self._pick_folder)
        self._row(body,"💾  Output .ply","Where to save the 3D point cloud",self.out_var,self._pick_out)
        tips=tk.Frame(body,bg=CARD,highlightbackground=BORDER,highlightthickness=1,padx=16,pady=12)
        tips.pack(fill="x",pady=(0,16))
        tk.Label(tips,text="💡 Tips",font=("Segoe UI",11,"bold"),bg=CARD,fg=ACCENT).pack(anchor="w")
        for t in ["50–200 overlapping photos (>60% overlap)","Consistent lighting, no reflective surfaces","Walk around object or fly a grid","Higher resolution = more detail"]:
            tk.Label(tips,text=f"  · {t}",font=("Segoe UI",9),bg=CARD,fg=MUTED).pack(anchor="w")
        lf=tk.Frame(body,bg=CARD,highlightbackground=BORDER,highlightthickness=1); lf.pack(fill="both",expand=True,pady=(0,16))
        tk.Label(lf,text="  📋  Log",font=("Segoe UI",11,"bold"),bg=CARD,fg=TEXT,anchor="w").pack(fill="x",padx=12,pady=(8,0))
        sb=tk.Scrollbar(lf); sb.pack(side="right",fill="y")
        self.log=tk.Text(lf,bg="#0a0c10",fg=GREEN,font=("Consolas",9),relief="flat",bd=0,padx=10,pady=8,yscrollcommand=sb.set,state="disabled",wrap="word")
        self.log.pack(fill="both",expand=True,padx=1,pady=(0,1)); sb.config(command=self.log.yview)
        bot=tk.Frame(body,bg=BG); bot.pack(fill="x")
        self.prog=ttk.Progressbar(bot,mode="indeterminate"); self.prog.pack(side="left",fill="x",expand=True,padx=(0,12))
        self.btn=tk.Button(bot,text="▶  Run Scan",font=("Segoe UI",11,"bold"),bg=ACCENT,fg="white",relief="flat",bd=0,padx=22,pady=10,cursor="hand2",command=self._start)
        self.btn.pack(side="right")
        sf=tk.Frame(self,bg=SURFACE,pady=5); sf.pack(fill="x")
        tk.Label(sf,textvariable=self.status_var,font=("Segoe UI",9),bg=SURFACE,fg=MUTED).pack()

    def _row(self,p,title,sub,var,cmd):
        c=tk.Frame(p,bg=CARD,highlightbackground=BORDER,highlightthickness=1,padx=16,pady=12); c.pack(fill="x",pady=(0,12))
        i=tk.Frame(c,bg=CARD); i.pack(side="left",fill="x",expand=True)
        tk.Label(i,text=title,font=("Segoe UI",11,"bold"),bg=CARD,fg=TEXT).pack(anchor="w")
        tk.Label(i,text=sub,font=("Segoe UI",9),bg=CARD,fg=MUTED).pack(anchor="w")
        tk.Label(i,textvariable=var,font=("Segoe UI",9),bg=CARD,fg=ACCENT,wraplength=450,justify="left").pack(anchor="w")
        tk.Button(c,text="Browse",font=("Segoe UI",9),bg=BORDER,fg=TEXT,relief="flat",bd=0,padx=14,pady=6,cursor="hand2",command=cmd).pack(side="right")

    def _pick_folder(self):
        f=filedialog.askdirectory()
        if f: self.folder_var.set(f);
        if f and not self.out_var.get(): self.out_var.set(os.path.join(f,"pointcloud.ply"))

    def _pick_out(self):
        p=filedialog.asksaveasfilename(defaultextension=".ply",filetypes=[("Point Cloud","*.ply")])
        if p: self.out_var.set(p)

    def _start(self):
        if self.running: return
        f,o=self.folder_var.get().strip(),self.out_var.get().strip()
        if not f or not os.path.isdir(f): messagebox.showerror("Error","Select a valid image folder!"); return
        if not o: messagebox.showerror("Error","Choose output path!"); return
        self.running=True; self.btn.config(state="disabled",bg=MUTED)
        self.prog.start(12); self.status_var.set("Processing…")
        self.log.config(state="normal"); self.log.delete("1.0","end"); self.log.config(state="disabled")
        threading.Thread(target=run_pipeline,args=(f,o,self._write,self._done,self._err),daemon=True).start()

    def _write(self,msg):
        self.after(0,lambda:(self.log.config(state="normal"),self.log.insert("end",msg+"\n"),self.log.see("end"),self.log.config(state="disabled")))

    def _done(self,p):
        self.after(0,lambda:(self.prog.stop(),setattr(self,'running',False),self.btn.config(state="normal",bg=ACCENT),self.status_var.set("✅ Done!"),messagebox.showinfo("Done 🎉",f"Saved:\n{p}\n\nOpen in MeshLab or Blender!")))

    def _err(self,msg):
        self.after(0,lambda:(self.prog.stop(),setattr(self,'running',False),self.btn.config(state="normal",bg=ACCENT),self.status_var.set("❌ Error"),self._write(f"\n❌ {msg}"),messagebox.showerror("Error",msg)))

    def _center(self):
        self.update_idletasks(); w,h=760,640
        self.geometry(f"{w}x{h}+{(self.winfo_screenwidth()-w)//2}+{(self.winfo_screenheight()-h)//2}")

if __name__ == "__main__":
    App().mainloop()
