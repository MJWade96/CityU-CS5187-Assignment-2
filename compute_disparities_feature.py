import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def read_color_and_gray(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return color, gray


def detect_and_match_sift(left_gray: np.ndarray, right_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # SIFT (requires opencv-contrib-python)
    if hasattr(cv2, "SIFT_create"):
        sift = cv2.SIFT_create()
    else:
        # Fallback to xfeatures2d if available
        try:
            sift = cv2.xfeatures2d.SIFT_create()  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError("SIFT is unavailable. Please install opencv-contrib-python.") from exc

    kp1, des1 = sift.detectAndCompute(left_gray, None)
    kp2, des2 = sift.detectAndCompute(right_gray, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise RuntimeError("Insufficient SIFT features detected.")

    # BFMatcher with L2 for SIFT
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    # Keep top matches for robust F estimation
    keep = max(1000, int(0.25 * len(matches)))
    matches = matches[:keep]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def estimate_fundamental(pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)
    if F is None or F.size == 0:
        raise RuntimeError("Failed to estimate fundamental matrix.")
    mask = mask.reshape(-1).astype(bool)
    return F, mask


def uncalibrated_rectification(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = left_gray.shape[:2]
    # computeCorrespondEpilines (optional, mainly to use hinted API and can be helpful for debugging)
    _ = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    _ = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)

    retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, imgSize=(w, h))
    if not retval:
        raise RuntimeError("stereoRectifyUncalibrated failed to compute rectification.")

    left_rect = cv2.warpPerspective(left_gray, H1, (w, h))
    right_rect = cv2.warpPerspective(right_gray, H2, (w, h))
    return left_rect, right_rect, H1, H2


def create_sgbm(width: int) -> cv2.StereoSGBM:
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
    try:
        gt_img = np.array(Image.open(gt_disp_path), dtype=float)
        gt_max = float(np.max(gt_img))
        if not np.isfinite(gt_max) or gt_max <= 0:
            gt_max = 255.0
    except Exception:
        gt_max = 255.0
    scale = gt_max / max(1.0, float(num_disparities))
    pred_scaled = np.clip(pred_disp_px * scale, 0.0, gt_max)
    return pred_scaled.astype(np.uint8)


def process_scene(scene_name: str) -> Tuple[str, int]:
    left_color, left_gray = read_color_and_gray(os.path.join(scene_name, "view1.png"))
    right_color, right_gray = read_color_and_gray(os.path.join(scene_name, "view5.png"))

    if left_gray.shape != right_gray.shape:
        raise ValueError(
            f"Stereo pair size mismatch for {scene_name}: {left_gray.shape} vs {right_gray.shape}"
        )

    pts1, pts2 = detect_and_match_sift(left_gray, right_gray)
    F, inlier_mask = estimate_fundamental(pts1, pts2)
    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    left_rect, right_rect, _, _ = uncalibrated_rectification(left_gray, right_gray, pts1_in, pts2_in, F)

    sgbm = create_sgbm(width=left_rect.shape[1])
    raw = sgbm.compute(left_rect, right_rect).astype(np.float32)
    disp = raw / 16.0
    disp[disp < 0] = 0.0
    num_disparities = sgbm.getNumDisparities()

    gt_disp_path = os.path.join("PSNR_Assignment2", "PSNR_Python", "gt", scene_name, "disp1.png")
    disp_uint8 = scale_to_gt_range(disp, gt_disp_path, num_disparities)

    out_dir = os.path.join("PSNR_Assignment2", "PSNR_Python", "pred", scene_name)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "disp1.png")
    Image.fromarray(disp_uint8).save(out_path)

    return out_path, num_disparities


def main() -> None:
    scenes: List[str] = ["Art", "Dolls", "Reindeer"]
    results = []
    for scene in scenes:
        out_path, num_disp = process_scene(scene)
        results.append((scene, out_path, num_disp))
    for scene, out_path, num_disp in results:
        print(f"Saved {scene} (feature pipeline) to: {out_path} (numDisparities={num_disp})")


if __name__ == "__main__":
    main()


