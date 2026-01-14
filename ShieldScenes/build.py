#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    SHIELD SLIDE → SCENE FORGE                                  ║
║              Transform PNGs into Explorable 3D Dioramas                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  USAGE:                                                                        ║
║    cd /Users/gaia/TMM/ShieldScenes                                             ║
║    python3 build.py                                                            ║
║                                                                                ║
║  REQUIREMENTS:                                                                 ║
║    pip install pillow numpy opencv-python                                      ║
║                                                                                ║
║  OUTPUT:                                                                       ║
║    dist/manifest.json           - Scene registry                               ║
║    dist/pages/XXXX.png          - Copied slides                                ║
║    dist/scenes/XXXX/scene.json  - 3D diorama specs                             ║
║    dist/scenes/XXXX/scene.mpd   - LDraw exports (optional)                     ║
║                                                                                ║
║  THEN VIEW:                                                                    ║
║    cd dist && python3 -m http.server 8000                                      ║
║    open http://localhost:8000                                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Try to import image processing libraries
try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL/numpy not installed. Using fallback cutout generation.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not installed. Using simplified cutout detection.")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
SOURCE_DIR = Path("/Users/gaia/TMM/Shield of Achilles Production Pipeline")
OUTPUT_DIR = Path("/Users/gaia/TMM/ShieldScenes/dist")
WORLD_SCALE = 0.1  # 1000px = 100 world units
MIN_CUTOUTS = 8
MAX_CUTOUTS = 20
MIN_AREA_RATIO = 0.02
MAX_AREA_RATIO = 0.60
Z_NEAR = 20
Z_FAR = 240

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def natural_sort_key(s: str) -> List:
    """Sort strings with embedded numbers naturally."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]

def find_pngs(directory: Path) -> List[Path]:
    """Find all PNG files recursively and sort naturally."""
    pngs = list(directory.glob("**/*.png")) + list(directory.glob("**/*.PNG"))
    return sorted(pngs, key=lambda p: natural_sort_key(p.name))

def iou(box1: Tuple, box2: Tuple) -> float:
    """Calculate intersection over union of two boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# CUTOUT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_cutouts_opencv(image_path: Path) -> List[Dict]:
    """Detect salient regions using OpenCV edge detection and contours."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    h, w = img.shape[:2]
    total_area = w * h
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate to connect nearby edges
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate saliency map (simple gradient magnitude)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    saliency = np.sqrt(sobelx**2 + sobely**2)
    saliency = (saliency / saliency.max() * 255).astype(np.uint8)
    
    # Convert to HSV for saturation analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    candidates = []
    
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        area_ratio = area / total_area
        
        # Filter by area
        if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            continue
        
        # Expand bounding box slightly
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        cw = min(w - x, cw + 2 * margin)
        ch = min(h - y, ch + 2 * margin)
        
        # Calculate scores
        region_edges = edges[y:y+ch, x:x+cw]
        edge_density = np.mean(region_edges) / 255
        
        region_saliency = saliency[y:y+ch, x:x+cw]
        saliency_score = np.mean(region_saliency) / 255
        
        region_sat = saturation[y:y+ch, x:x+cw]
        sat_score = np.mean(region_sat) / 255
        
        # Combined score
        score = edge_density * 0.4 + saliency_score * 0.4 + sat_score * 0.2
        
        candidates.append({
            'rect': (x, y, cw, ch),
            'area_ratio': area_ratio,
            'score': score
        })
    
    # Sort by score
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    # Non-maximum suppression
    selected = []
    for cand in candidates:
        overlaps = False
        for sel in selected:
            if iou(cand['rect'], sel['rect']) > 0.3:
                overlaps = True
                break
        if not overlaps:
            selected.append(cand)
        if len(selected) >= MAX_CUTOUTS:
            break
    
    # Ensure minimum cutouts by adding grid-based fallbacks
    if len(selected) < MIN_CUTOUTS:
        selected.extend(generate_grid_cutouts(w, h, MIN_CUTOUTS - len(selected), selected))
    
    return selected

def detect_cutouts_pil(image_path: Path) -> List[Dict]:
    """Fallback cutout detection using PIL only."""
    img = Image.open(image_path)
    w, h = img.size
    total_area = w * h
    
    # Convert to grayscale and numpy array
    gray = np.array(img.convert('L'))
    
    # Simple edge detection (gradient magnitude)
    gx = np.abs(np.diff(gray.astype(float), axis=1))
    gy = np.abs(np.diff(gray.astype(float), axis=0))
    
    # Pad to original size
    gx = np.pad(gx, ((0, 0), (0, 1)), mode='constant')
    gy = np.pad(gy, ((0, 1), (0, 0)), mode='constant')
    
    edges = np.sqrt(gx**2 + gy**2)
    
    # Find high-edge regions using sliding window
    window_sizes = [(h//4, w//4), (h//3, w//3), (h//5, w//5)]
    candidates = []
    
    for wh, ww in window_sizes:
        step_y, step_x = wh // 2, ww // 2
        for y in range(0, h - wh, step_y):
            for x in range(0, w - ww, step_x):
                region = edges[y:y+wh, x:x+ww]
                score = np.mean(region) / 255
                area_ratio = (ww * wh) / total_area
                
                if MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO:
                    candidates.append({
                        'rect': (x, y, ww, wh),
                        'area_ratio': area_ratio,
                        'score': score
                    })
    
    # Sort and filter
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    selected = []
    for cand in candidates:
        overlaps = False
        for sel in selected:
            if iou(cand['rect'], sel['rect']) > 0.3:
                overlaps = True
                break
        if not overlaps:
            selected.append(cand)
        if len(selected) >= MAX_CUTOUTS:
            break
    
    if len(selected) < MIN_CUTOUTS:
        selected.extend(generate_grid_cutouts(w, h, MIN_CUTOUTS - len(selected), selected))
    
    return selected

def generate_grid_cutouts(w: int, h: int, count: int, existing: List) -> List[Dict]:
    """Generate grid-based cutouts as fallback."""
    cutouts = []
    cols = 3
    rows = max(1, (count + cols - 1) // cols)
    
    cw = w // cols
    ch = h // rows
    
    for row in range(rows):
        for col in range(cols):
            if len(cutouts) >= count:
                break
            x = col * cw + cw // 4
            y = row * ch + ch // 4
            rect_w = cw // 2
            rect_h = ch // 2
            
            # Check overlap with existing
            overlaps = False
            for ex in existing:
                if iou((x, y, rect_w, rect_h), ex['rect']) > 0.3:
                    overlaps = True
                    break
            
            if not overlaps:
                cutouts.append({
                    'rect': (x, y, rect_w, rect_h),
                    'area_ratio': (rect_w * rect_h) / (w * h),
                    'score': 0.5
                })
    
    return cutouts

def detect_cutouts(image_path: Path) -> List[Dict]:
    """Main cutout detection dispatcher."""
    if HAS_CV2:
        return detect_cutouts_opencv(image_path)
    elif HAS_PIL:
        return detect_cutouts_pil(image_path)
    else:
        # Pure fallback: grid-based
        img = Image.open(image_path) if HAS_PIL else None
        w, h = (1920, 1080)
        if img:
            w, h = img.size
        return generate_grid_cutouts(w, h, MIN_CUTOUTS, [])

# ═══════════════════════════════════════════════════════════════════════════════
# SCENE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_scene_json(slide_id: str, title: str, image_path: Path, w: int, h: int, cutouts: List[Dict]) -> Dict:
    """Generate a complete scene.json for a slide."""
    
    scene_w = w * WORLD_SCALE
    scene_h = h * WORLD_SCALE
    
    # Camera positioned to see full slide
    cam_z = max(scene_w, scene_h) * 1.2
    
    scene = {
        "id": slide_id,
        "title": title,
        "sourceImage": f"../../pages/{slide_id}.png",
        "canvasSize": [w, h],
        "camera": {
            "type": "perspective",
            "fov": 45,
            "position": [0, 0, cam_z],
            "target": [0, 0, 0],
            "near": 1,
            "far": 5000
        },
        "environment": {
            "background": "#050505",
            "fog": {"enabled": True, "near": cam_z * 0.5, "far": cam_z * 3},
            "post": {"bloom": True, "vignette": True, "filmGrain": 0.08}
        },
        "lights": [
            {"type": "hemisphere", "intensity": 0.55},
            {"type": "directional", "position": [500, 900, 400], "intensity": 1.1, "shadows": True},
            {"type": "point", "position": [-350, 200, 500], "intensity": 0.45}
        ],
        "assets": {
            "textures": [{"id": "page", "src": f"../../pages/{slide_id}.png"}],
            "ldraw": []
        },
        "entities": [],
        "navigation": {
            "snapPoints": [
                {"name": "front", "cameraPos": [0, 0, cam_z], "target": [0, 0, 0]},
                {"name": "oblique", "cameraPos": [cam_z * 0.5, cam_z * 0.3, cam_z * 0.8], "target": [0, 0, 0]},
                {"name": "close", "cameraPos": [0, 0, cam_z * 0.5], "target": [0, 0, 0]}
            ],
            "tour": []
        },
        "motion": {
            "float": {"enabled": True, "amplitude": 1.6, "speed": 0.25},
            "drift": {"enabled": True, "strength": 0.35}
        }
    }
    
    # Background plate
    scene["entities"].append({
        "id": "bg",
        "type": "plane",
        "texture": "page",
        "size": [scene_w, scene_h],
        "position": [0, 0, 0],
        "rotation": [0, 0, 0],
        "material": {"transparent": False, "opacity": 1, "emissive": 0.04}
    })
    
    # Sort cutouts by area (larger = closer)
    cutouts_sorted = sorted(cutouts, key=lambda c: c['area_ratio'], reverse=True)
    
    # Generate cutout entities
    for i, cutout in enumerate(cutouts_sorted):
        x, y, cw, ch = cutout['rect']
        
        # Normalized rect
        rect_n = [x / w, y / h, cw / w, ch / h]
        
        # World position (centered at origin)
        world_x = (x + cw / 2 - w / 2) * WORLD_SCALE
        world_y = -(y + ch / 2 - h / 2) * WORLD_SCALE  # Flip Y
        
        # Z depth based on rank (larger areas closer)
        z_range = Z_FAR - Z_NEAR
        z = Z_NEAR + (i / max(1, len(cutouts_sorted) - 1)) * z_range
        
        # Emissive higher for top-ranking (likely text/important)
        emissive = 0.15 - (i * 0.01)
        
        entity = {
            "id": f"cut-{i+1:02d}",
            "type": "cutoutPlane",
            "fromTexture": "page",
            "rectN": rect_n,
            "size": [cw * WORLD_SCALE, ch * WORLD_SCALE],
            "position": [world_x, world_y, z],
            "rotation": [0, 0, 0],
            "material": {
                "transparent": True,
                "alphaMode": "mask",
                "opacity": 1,
                "emissive": max(0.05, emissive)
            }
        }
        scene["entities"].append(entity)
        
        # Add to tour (first 5 cutouts)
        if i < 5:
            scene["navigation"]["tour"].append({
                "focusEntity": f"cut-{i+1:02d}",
                "seconds": 2.5
            })
    
    return scene

def generate_scene_mpd(slide_id: str, scene: Dict) -> str:
    """Generate an LDraw MPD file for the scene."""
    w, h = scene["canvasSize"]
    scene_w = w * WORLD_SCALE
    scene_h = h * WORLD_SCALE
    
    # Scale to LDraw units (1 stud = 20 LDU, scene unit ~= 2 LDU)
    ldu_scale = 2
    plate_w = int(scene_w * ldu_scale / 20) * 20
    plate_h = int(scene_h * ldu_scale / 20) * 20
    
    mpd = f"""0 FILE {slide_id}.ldr
0 {scene["title"]}
0 Name: {slide_id}.ldr
0 Author: Shield Scene Forge
0 !MENTO SHOT Front CAMERA 0 -200 {plate_h} TARGET 0 0 0
0 !MENTO SHOT Oblique CAMERA {plate_w//2} -150 {plate_h//2} TARGET 0 0 0
0 !MENTO SHOT Overhead CAMERA 0 -{plate_h} 50 TARGET 0 0 0
0 !MENTO LIGHT Sun TYPE directional POSITION 500 -900 400 INTENSITY 1.1
0 !MENTO LIGHT Fill TYPE point POSITION -350 -200 500 INTENSITY 0.45

0 // Base plate
1 71 0 0 0 1 0 0 0 1 0 0 0 1 3811.dat

0 // Back wall (simplified as tall plate)
1 72 0 -{plate_h//2} -{plate_w//2} 0 0 1 0 1 0 -1 0 0 3811.dat

"""
    
    # Add frame elements for key cutouts
    for i, entity in enumerate(scene["entities"][:6]):
        if entity["type"] == "cutoutPlane":
            x, y, z = entity["position"]
            lx = int(x * ldu_scale)
            ly = int(-z * ldu_scale)  # Z becomes Y in LDraw
            lz = int(-y * ldu_scale)  # Y becomes Z
            
            mpd += f"0 // Frame for {entity['id']}\n"
            mpd += f"1 {14 + i} {lx} {ly} {lz} 1 0 0 0 1 0 0 0 1 3024.dat\n"
    
    mpd += "\n0 FILE main.mpd\n"
    mpd += f"1 16 0 0 0 1 0 0 0 1 0 0 0 1 {slide_id}.ldr\n"
    mpd += "0\n"
    
    return mpd

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def build():
    """Main build pipeline."""
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║              SHIELD SLIDE → SCENE FORGE                       ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    # Create output directories
    pages_dir = OUTPUT_DIR / "pages"
    scenes_dir = OUTPUT_DIR / "scenes"
    pages_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    
    # Find source PNGs
    pngs = find_pngs(SOURCE_DIR)
    print(f"\n→ Found {len(pngs)} PNG slides")
    
    if not pngs:
        print("✗ No PNG files found!")
        return
    
    manifest = {
        "title": "Shield of Achilles Production Pipeline",
        "generated": "2026-01-14",
        "slides": []
    }
    
    for i, png_path in enumerate(pngs):
        slide_id = f"{i+1:04d}"
        title = f"Slide {slide_id} - {png_path.stem}"
        
        print(f"\n[{slide_id}] Processing: {png_path.name}")
        
        # Copy PNG to pages
        dest_png = pages_dir / f"{slide_id}.png"
        shutil.copy2(png_path, dest_png)
        print(f"  ✓ Copied to pages/{slide_id}.png")
        
        # Get image dimensions
        if HAS_PIL:
            img = Image.open(png_path)
            w, h = img.size
        else:
            w, h = 1920, 1080  # Default assumption
        
        # Detect cutouts
        cutouts = detect_cutouts(png_path)
        print(f"  ✓ Detected {len(cutouts)} cutout regions")
        
        # Generate scene JSON
        scene = generate_scene_json(slide_id, title, png_path, w, h, cutouts)
        
        scene_dir = scenes_dir / slide_id
        scene_dir.mkdir(exist_ok=True)
        
        scene_json_path = scene_dir / "scene.json"
        with open(scene_json_path, 'w') as f:
            json.dump(scene, f, indent=2)
        print(f"  ✓ Generated scenes/{slide_id}/scene.json")
        
        # Generate MPD
        mpd_content = generate_scene_mpd(slide_id, scene)
        mpd_path = scene_dir / "scene.mpd"
        with open(mpd_path, 'w') as f:
            f.write(mpd_content)
        print(f"  ✓ Generated scenes/{slide_id}/scene.mpd")
        
        # Auto-tag based on content
        tags = []
        if len(cutouts) > 12:
            tags.append("diagram")
        if any(c['area_ratio'] > 0.15 for c in cutouts):
            tags.append("title-heavy")
        if len(cutouts) < 10:
            tags.append("minimal")
        
        manifest["slides"].append({
            "id": slide_id,
            "title": title,
            "source": str(png_path.relative_to(SOURCE_DIR.parent)),
            "page": f"pages/{slide_id}.png",
            "scene": f"scenes/{slide_id}/scene.json",
            "mpd": f"scenes/{slide_id}/scene.mpd",
            "dimensions": [w, h],
            "cutouts": len(cutouts),
            "tags": tags
        })
    
    # Write manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✓ Generated manifest.json ({len(manifest['slides'])} slides)")
    
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║                    BUILD COMPLETE                              ║")
    print("╠═══════════════════════════════════════════════════════════════╣")
    print("║  To view:                                                      ║")
    print("║    cd /Users/gaia/TMM/ShieldScenes/dist                        ║")
    print("║    python3 -m http.server 8000                                 ║")
    print("║    open http://localhost:8000                                  ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    build()
