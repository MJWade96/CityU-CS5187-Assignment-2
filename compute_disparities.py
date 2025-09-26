import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def read_grayscale(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def create_sgbm(width: int) -> cv2.StereoSGBM:
    # Choose numDisparities as a multiple of 16, scaled to image width
    # Heuristic: roughly width/8, rounded up to nearest multiple of 16
    raw_num_disp = max(64, int(np.ceil(width / 8.0 / 16.0)) * 16)
    block_size = 5

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=raw_num_disp,
        blockSize=block_size,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return sgbm


def scale_to_gt_range(pred_disp_px: np.ndarray, gt_disp_path: str, num_disparities: int) -> np.ndarray:
    # Read GT disparity to infer display range
    try:
        gt_img = np.array(Image.open(gt_disp_path), dtype=float)
        gt_max = float(np.max(gt_img))
        if not np.isfinite(gt_max) or gt_max <= 0:
            gt_max = 255.0
    except Exception:
        gt_max = 255.0

    # Map predicted disparity in pixels (0..numDisparities) to 0..gt_max
    scale = gt_max / max(1.0, float(num_disparities))
    pred_scaled = np.clip(pred_disp_px * scale, 0.0, gt_max)
    return pred_scaled.astype(np.uint8)


def compute_and_save_disparity(scene_name: str) -> Tuple[str, int]:
    left_path = os.path.join(scene_name, "view1.png")
    right_path = os.path.join(scene_name, "view5.png")
    gt_disp_path = os.path.join(
        "PSNR_Assignment2",
        "PSNR_Python",
        "gt",
        scene_name,
        "disp1.png",
    )

    left = read_grayscale(left_path)
    right = read_grayscale(right_path)

    if left.shape != right.shape:
        raise ValueError(
            f"Stereo pair size mismatch for {scene_name}: {left.shape} vs {right.shape}"
        )

    sgbm = create_sgbm(width=left.shape[1])

    raw = sgbm.compute(left, right).astype(np.float32)
    disp = raw / 16.0
    disp[disp < 0] = 0.0

    num_disparities = sgbm.getNumDisparities()
    disp_uint8 = scale_to_gt_range(disp, gt_disp_path, num_disparities)

    out_dir = os.path.join(
        "PSNR_Assignment2",
        "PSNR_Python",
        "pred",
        scene_name,
    )
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "disp1.png")

    # Save using PIL to ensure uint8 PNG
    Image.fromarray(disp_uint8).save(out_path)
    return out_path, num_disparities


def main() -> None:
    scenes: List[str] = ["Art", "Dolls", "Reindeer"]
    results = []
    for scene in scenes:
        out_path, num_disp = compute_and_save_disparity(scene)
        results.append((scene, out_path, num_disp))

    for scene, out_path, num_disp in results:
        print(f"Saved {scene} disparity to: {out_path} (numDisparities={num_disp})")


if __name__ == "__main__":
    main()


