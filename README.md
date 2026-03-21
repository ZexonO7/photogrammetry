# 🛸 DroneScanner

A photogrammetry tool that turns drone photos into 3D point clouds.

**No LiDAR needed — just photos.**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## How It Works

```
Drone takes photos → dump to PC → DroneScanner processes → 3D .ply file
```

Under the hood it runs **Structure from Motion (SfM)**:
1. Detects keypoints in every photo (SIFT)
2. Matches keypoints across overlapping image pairs
3. Estimates camera poses using Essential Matrix decomposition
4. Triangulates 3D points from matched pairs
5. Cleans outliers and exports a coloured point cloud

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/dronescanner.git
cd dronescanner
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run**
```bash
python scanner.py
```

---

## Usage

1. Click **Browse** next to *Image Folder* → select your folder of drone photos
2. Click **Browse** next to *Output File* → choose where to save the `.ply`
3. Hit **▶ Run Scan**
4. 3D viewer pops up when done — close it to finish

---

## Tips for Best Results

| Thing | Recommendation |
|---|---|
| Number of photos | 50–200 |
| Overlap between shots | >60% |
| Lighting | Consistent, avoid harsh shadows |
| Surfaces | Avoid glass/mirrors (reflective = bad) |
| Resolution | Higher = more detail |
| Flight pattern | Grid or orbit around subject |

---

## Viewing the Output

The output `.ply` file can be opened in:
- **[MeshLab](https://www.meshlab.net/)** — free, great for point clouds
- **[CloudCompare](https://cloudcompare.org/)** — free, very powerful
- **[Blender](https://www.blender.org/)** — free, import via File > Import > Stanford PLY

---

## Requirements

- Python 3.9+
- Windows / macOS / Linux
- GPU optional but recommended for large scans

---

## Roadmap

- [ ] Mesh generation (not just point cloud)
- [ ] Auto waypoint photo capture via MAVLink
- [ ] Web UI version
- [ ] GPU acceleration

---

## License

MIT — do whatever you want with it
