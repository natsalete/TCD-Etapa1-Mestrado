"""
Extração de features tabulares do dataset RoCoLe (Robusta Coffee Leaf).

Para cada imagem:
  - Estatísticas de cor em RGB e HSV
  - Features de textura GLCM (em grayscale)
  - Proporção de pixels em faixas cromáticas típicas (verde saudável, marrom/laranja de ferrugem)
  - Métricas de qualidade (brilho médio, variância do Laplaciano como proxy de nitidez)
  - Dimensões originais

Saída: outputs/rocole_features.csv — usado na AED e nas etapas seguintes do TCD no KNIME.

Uso:
    cd /Users/natalia.salete/UFU-Mestrado
    source .venv/bin/activate
    python TCD-Etapa1/scripts/extract_features.py
"""

from pathlib import Path
from typing import Optional
import sys
import time

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "Rocole"
PHOTOS_DIR = DATASET_DIR / "Photos"
CLASSES_XLSX = DATASET_DIR / "Annotations" / "RoCoLe-classes.xlsx"
OUTPUT_CSV = ROOT / "TCD-Etapa1" / "outputs" / "rocole_features.csv"

# Redimensionamento para acelerar GLCM sem perder assinatura de textura relevante.
RESIZE_DIM = 256

# Faixas HSV (OpenCV: H em [0,179], S e V em [0,255]).
HSV_RANGES = {
    "healthy_green": ((25, 40, 40), (85, 255, 255)),
    "rust_brown_orange": ((5, 60, 40), (25, 255, 255)),
    "dark_necrotic": ((0, 0, 0), (180, 255, 60)),
}


def color_stats(img_rgb: np.ndarray, img_hsv: np.ndarray) -> dict:
    stats = {}
    for i, ch in enumerate(("r", "g", "b")):
        stats[f"{ch}_mean"] = float(img_rgb[..., i].mean())
        stats[f"{ch}_std"] = float(img_rgb[..., i].std())
    for i, ch in enumerate(("h", "s", "v")):
        stats[f"{ch}_mean"] = float(img_hsv[..., i].mean())
        stats[f"{ch}_std"] = float(img_hsv[..., i].std())
    return stats


def hsv_range_ratios(img_hsv: np.ndarray) -> dict:
    total = img_hsv.shape[0] * img_hsv.shape[1]
    out = {}
    for name, (lo, hi) in HSV_RANGES.items():
        mask = cv2.inRange(img_hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        out[f"ratio_{name}"] = float(mask.sum() / 255) / total
    return out


def glcm_features(gray: np.ndarray) -> dict:
    # Quantiza para 32 níveis para reduzir custo do GLCM.
    levels = 32
    q = (gray.astype(np.float32) * (levels - 1) / 255).astype(np.uint8)
    glcm = graycomatrix(
        q,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    out = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"):
        vals = graycoprops(glcm, prop)[0]
        out[f"glcm_{prop.lower()}"] = float(vals.mean())
    return out


def quality_metrics(gray: np.ndarray) -> dict:
    return {
        "brightness": float(gray.mean()),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def extract_one(img_path: Path) -> Optional[dict]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    original_dims = {"width": w, "height": h, "aspect_ratio": float(w / h)}

    # Redimensiona (mantendo proporção pelo menor lado) para acelerar.
    scale = RESIZE_DIM / min(h, w)
    img_small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    feats = {"file": img_path.name}
    feats.update(original_dims)
    feats.update(color_stats(img_rgb, img_hsv))
    feats.update(hsv_range_ratios(img_hsv))
    feats.update(glcm_features(gray))
    feats.update(quality_metrics(gray))
    return feats


def main() -> int:
    if not CLASSES_XLSX.exists():
        print(f"[erro] Não encontrei {CLASSES_XLSX}")
        return 1
    if not PHOTOS_DIR.exists():
        print(f"[erro] Não encontrei {PHOTOS_DIR}")
        return 1

    classes = pd.read_excel(CLASSES_XLSX)
    classes = classes.rename(columns={"File": "file", "Binary.Label": "binary_label", "Multiclass.Label": "multiclass_label"})
    print(f"[info] {len(classes)} imagens listadas em RoCoLe-classes.xlsx")

    rows = []
    t0 = time.time()
    for i, row in classes.iterrows():
        img_path = PHOTOS_DIR / row["file"]
        feats = extract_one(img_path)
        if feats is None:
            print(f"[warn] falha ao ler {img_path.name}")
            continue
        feats["binary_label"] = row["binary_label"]
        feats["multiclass_label"] = row["multiclass_label"]
        rows.append(feats)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(classes)}  ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    # Reordena: label no final, file no começo.
    cols = ["file"] + [c for c in df.columns if c not in ("file", "binary_label", "multiclass_label")] + ["binary_label", "multiclass_label"]
    df = df[cols]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[ok] salvei {len(df)} linhas em {OUTPUT_CSV}")
    print(f"[ok] {len(df.columns) - 3} features numéricas geradas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
