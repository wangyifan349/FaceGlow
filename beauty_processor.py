"""安装依赖可直接执行 `pip install insightface onnxruntime opencv-python numpy`。FaceGlow 是一个用于离线照片美颜与批量处理的 Python 程序：InsightFace 负责检测人脸并提供 106 点与 5 点关键点，OpenCV 和 NumPy 负责精确肤色区域识别、五官保护、保边磨皮、纹理恢复、局部美白以及脸颊和下颌的局部瘦脸形变；程序支持命令行处理单张图片或整个文件夹，也可以作为 Python 模块直接调用，所有输入文件只读，处理结果始终写入新的输出路径，不覆盖、不移动、不删除原始照片。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# -----------------------------------------------------------------------------
# Configuration and public data structures
# -----------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class BeautyConfig:
    smooth_strength: float = 0.62              # 0..1, skin smoothing
    whiten_strength: float = 0.24              # 0..1, local skin whitening
    slim_strength: float = 0.16                # 0..1, cheek/jaw slimming
    detail_strength: float = 0.30              # 0..1, texture and feature restoration
    mask_feather_ratio: float = 0.035          # Feather radius relative to face width
    max_faces: int = 0                         # 0 means all detected faces
    detection_size: int = 640                  # InsightFace detection input size
    detection_threshold: float = 0.50          # Face detection threshold
    jpeg_quality: int = 95                     # Output JPEG quality
    png_compression: int = 3                   # Output PNG compression


@dataclass
class ProcessResult:
    output_path: Path | None = None
    face_count: int = 0
    success: bool = False
    message: str = ""


@dataclass
class BatchSummary:
    total_files: int = 0
    success_files: int = 0
    skipped_files: int = 0
    no_face_files: int = 0


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def ensure_odd_kernel_size(value: int, minimum: int = 3) -> int:
    kernel_size = max(minimum, int(value))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_path_inside(path: Path, directory: Path) -> bool:
    resolved_path = path.resolve()
    resolved_directory = directory.resolve()
    return resolved_path == resolved_directory or resolved_directory in resolved_path.parents


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def calculate_bounding_box_area(bounding_box: np.ndarray) -> float:
    left_x, top_y, right_x, bottom_y = [float(value) for value in bounding_box[:4]]
    box_width = max(0.0, right_x - left_x)
    box_height = max(0.0, bottom_y - top_y)
    return box_width * box_height


def clip_point(
    point_coordinates: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    x_coordinate = int(np.clip(round(float(point_coordinates[0])), 0, image_width - 1))
    y_coordinate = int(np.clip(round(float(point_coordinates[1])), 0, image_height - 1))
    return x_coordinate, y_coordinate


def expand_face_region(
    bounding_box: np.ndarray,
    image_width: int,
    image_height: int,
    margin: float = 0.18,
) -> tuple[int, int, int, int]:
    left_x, top_y, right_x, bottom_y = [float(value) for value in bounding_box[:4]]
    face_width = max(1.0, right_x - left_x)
    face_height = max(1.0, bottom_y - top_y)

    region_left = int(np.floor(left_x - face_width * margin))
    region_top = int(np.floor(top_y - face_height * margin))
    region_right = int(np.ceil(right_x + face_width * margin))
    region_bottom = int(np.ceil(bottom_y + face_height * margin))

    region_left = int(np.clip(region_left, 0, image_width - 1))
    region_top = int(np.clip(region_top, 0, image_height - 1))
    region_right = int(np.clip(region_right, region_left + 1, image_width))
    region_bottom = int(np.clip(region_bottom, region_top + 1, image_height))
    return region_left, region_top, region_right, region_bottom


class FaceAnalyzer:
    """Loads InsightFace once and reuses it across all images in a batch."""

    def __init__(self, config: BeautyConfig) -> None:
        self.analyzer = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "landmark_2d_106"],
            providers=["CPUExecutionProvider"],
        )
        self.analyzer.prepare(
            ctx_id=-1,
            det_thresh=float(config.detection_threshold),
            det_size=(int(config.detection_size), int(config.detection_size)),
        )

    def detect(self, image: np.ndarray) -> list[Any]:
        detected_faces = self.analyzer.get(image)
        return list(detected_faces)


def get_landmarks_106(face: Any) -> np.ndarray | None:
    landmark_data = getattr(face, "landmark_2d_106", None)
    if landmark_data is None:
        return None

    landmark_points = np.asarray(landmark_data, dtype=np.float32)
    if landmark_points.shape != (106, 2):
        return None
    return landmark_points


def get_five_keypoints(face: Any) -> np.ndarray | None:
    keypoint_data = getattr(face, "kps", None)
    if keypoint_data is None:
        return None

    keypoint_array = np.asarray(keypoint_data, dtype=np.float32)
    if keypoint_array.shape != (5, 2):
        return None
    return keypoint_array


# -----------------------------------------------------------------------------
# Precise face and skin masks
# -----------------------------------------------------------------------------


def create_geometry_face_mask(
    image_shape: tuple[int, int, int],
    landmarks: np.ndarray,
    bounding_box: np.ndarray,
) -> np.ndarray:
    """Creates a conservative face area from the 106-point hull and face ellipse."""

    image_height, image_width = image_shape[:2]
    face_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    landmark_points = np.rint(landmarks).astype(np.int32)
    landmark_points[:, 0] = np.clip(landmark_points[:, 0], 0, image_width - 1)
    landmark_points[:, 1] = np.clip(landmark_points[:, 1], 0, image_height - 1)
    landmark_hull = cv2.convexHull(landmark_points)
    cv2.fillConvexPoly(face_mask, landmark_hull, 255, lineType=cv2.LINE_AA)

    left_x, top_y, right_x, bottom_y = [float(value) for value in bounding_box[:4]]
    face_width = max(1.0, right_x - left_x)
    face_height = max(1.0, bottom_y - top_y)
    ellipse_center = (
        int(round((left_x + right_x) * 0.50)),
        int(round(top_y + face_height * 0.53)),
    )
    ellipse_axes = (
        max(1, int(round(face_width * 0.49))),
        max(1, int(round(face_height * 0.56))),
    )

    ellipse_mask = np.zeros_like(face_mask)
    cv2.ellipse(ellipse_mask, ellipse_center, ellipse_axes, 0.0, 0.0, 360.0, 255, -1, cv2.LINE_AA)
    return cv2.bitwise_and(face_mask, ellipse_mask)


def create_skin_color_mask(image: np.ndarray) -> np.ndarray:
    """Uses YCrCb, HSV and Lab voting to avoid relying on one rigid color rule."""

    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    luminance_channel = ycrcb_image[:, :, 0]
    saturation_channel = hsv_image[:, :, 1]
    value_channel = hsv_image[:, :, 2]
    lightness_channel = lab_image[:, :, 0]
    green_red_channel = lab_image[:, :, 1]
    blue_yellow_channel = lab_image[:, :, 2]

    ycrcb_skin_mask = cv2.inRange(
        ycrcb_image,
        np.array([20, 118, 68], dtype=np.uint8),
        np.array([255, 192, 155], dtype=np.uint8),
    )

    hsv_primary_skin_mask = cv2.inRange(
        hsv_image,
        np.array([0, 8, 25], dtype=np.uint8),
        np.array([40, 235, 255], dtype=np.uint8),
    )
    hsv_red_wrap_skin_mask = cv2.inRange(
        hsv_image,
        np.array([165, 8, 25], dtype=np.uint8),
        np.array([179, 235, 255], dtype=np.uint8),
    )
    hsv_skin_mask = cv2.bitwise_or(hsv_primary_skin_mask, hsv_red_wrap_skin_mask)

    lab_green_red_mask = cv2.inRange(green_red_channel, 118, 175)
    lab_blue_yellow_mask = cv2.inRange(blue_yellow_channel, 118, 195)
    lab_lightness_mask = cv2.inRange(lightness_channel, 18, 255)
    lab_skin_mask = cv2.bitwise_and(lab_green_red_mask, lab_blue_yellow_mask)
    lab_skin_mask = cv2.bitwise_and(lab_skin_mask, lab_lightness_mask)

    valid_luminance_mask = cv2.inRange(luminance_channel, 18, 255)
    valid_value_mask = cv2.inRange(value_channel, 22, 255)
    valid_saturation_mask = cv2.inRange(saturation_channel, 5, 245)
    valid_color_mask = cv2.bitwise_and(valid_luminance_mask, valid_value_mask)
    valid_color_mask = cv2.bitwise_and(valid_color_mask, valid_saturation_mask)

    ycrcb_vote = (ycrcb_skin_mask > 0).astype(np.uint8)
    hsv_vote = (hsv_skin_mask > 0).astype(np.uint8)
    lab_vote = (lab_skin_mask > 0).astype(np.uint8)
    skin_vote_count = ycrcb_vote + hsv_vote + lab_vote

    skin_color_mask = np.zeros_like(luminance_channel, dtype=np.uint8)
    skin_color_mask[skin_vote_count >= 2] = 255
    skin_color_mask = cv2.bitwise_and(skin_color_mask, valid_color_mask)

    morphology_kernel = np.ones((3, 3), dtype=np.uint8)
    skin_color_mask = cv2.morphologyEx(skin_color_mask, cv2.MORPH_OPEN, morphology_kernel, iterations=1)
    skin_color_mask = cv2.morphologyEx(skin_color_mask, cv2.MORPH_CLOSE, morphology_kernel, iterations=2)
    return skin_color_mask


def create_feature_protection_mask(
    image_shape: tuple[int, int, int],
    keypoints: np.ndarray | None,
    bounding_box: np.ndarray,
) -> np.ndarray:
    """Protects eyes, eyebrows, lips and central nose details from strong smoothing."""

    image_height, image_width = image_shape[:2]
    feature_protection_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if keypoints is None:
        return feature_protection_mask

    left_eye_point = keypoints[0]
    right_eye_point = keypoints[1]
    nose_point = keypoints[2]
    left_mouth_point = keypoints[3]
    right_mouth_point = keypoints[4]

    eye_distance = max(8.0, float(np.linalg.norm(right_eye_point - left_eye_point)))
    mouth_distance = max(8.0, float(np.linalg.norm(right_mouth_point - left_mouth_point)))
    face_height = max(8.0, float(bounding_box[3] - bounding_box[1]))

    eye_ellipse_axes = (
        max(5, int(round(eye_distance * 0.24))),
        max(4, int(round(eye_distance * 0.14))),
    )
    eyebrow_ellipse_axes = (
        max(6, int(round(eye_distance * 0.26))),
        max(4, int(round(eye_distance * 0.10))),
    )

    for eye_point in (left_eye_point, right_eye_point):
        eye_center = clip_point(eye_point, image_width, image_height)
        cv2.ellipse(
            feature_protection_mask,
            eye_center,
            eye_ellipse_axes,
            0.0,
            0.0,
            360.0,
            255,
            -1,
            cv2.LINE_AA,
        )

        eyebrow_point = eye_point.copy()
        eyebrow_point[1] -= eye_distance * 0.20
        eyebrow_center = clip_point(eyebrow_point, image_width, image_height)
        cv2.ellipse(
            feature_protection_mask,
            eyebrow_center,
            eyebrow_ellipse_axes,
            0.0,
            0.0,
            360.0,
            220,
            -1,
            cv2.LINE_AA,
        )

    mouth_center_point = (left_mouth_point + right_mouth_point) * 0.5
    mouth_ellipse_axes = (
        max(7, int(round(mouth_distance * 0.72))),
        max(4, int(round(mouth_distance * 0.40))),
    )
    cv2.ellipse(
        feature_protection_mask,
        clip_point(mouth_center_point, image_width, image_height),
        mouth_ellipse_axes,
        0.0,
        0.0,
        360.0,
        255,
        -1,
        cv2.LINE_AA,
    )

    nose_ellipse_axes = (
        max(3, int(round(eye_distance * 0.12))),
        max(4, int(round(face_height * 0.055))),
    )
    cv2.ellipse(
        feature_protection_mask,
        clip_point(nose_point, image_width, image_height),
        nose_ellipse_axes,
        0.0,
        0.0,
        360.0,
        170,
        -1,
        cv2.LINE_AA,
    )
    return feature_protection_mask


def create_precise_skin_mask(
    image: np.ndarray,
    landmarks: np.ndarray,
    keypoints: np.ndarray | None,
    bounding_box: np.ndarray,
    feather_ratio: float,
) -> np.ndarray:
    """Combines facial geometry, skin color and protected facial features."""

    geometry_mask = create_geometry_face_mask(image.shape, landmarks, bounding_box)
    skin_color_mask = create_skin_color_mask(image)
    feature_protection_mask = create_feature_protection_mask(image.shape, keypoints, bounding_box)

    precise_skin_mask = cv2.bitwise_and(geometry_mask, skin_color_mask)
    precise_skin_mask = cv2.subtract(precise_skin_mask, feature_protection_mask)

    face_width = max(1.0, float(bounding_box[2] - bounding_box[0]))
    feather_kernel_size = ensure_odd_kernel_size(int(round(face_width * max(0.01, feather_ratio))))
    precise_skin_mask = cv2.GaussianBlur(
        precise_skin_mask,
        (feather_kernel_size, feather_kernel_size),
        0,
    )
    return precise_skin_mask


def blend_with_mask(
    base_image: np.ndarray,
    effect_image: np.ndarray,
    mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Alpha blends an effect using a soft single-channel mask."""

    normalized_strength = clamp01(strength)
    if normalized_strength <= 0.0:
        return base_image.copy()

    alpha_mask = mask.astype(np.float32) / 255.0
    alpha_mask *= normalized_strength
    alpha_mask = alpha_mask[..., None]

    base_image_float = base_image.astype(np.float32)
    effect_image_float = effect_image.astype(np.float32)
    blended_image = base_image_float * (1.0 - alpha_mask) + effect_image_float * alpha_mask
    return np.clip(blended_image, 0.0, 255.0).astype(np.uint8)


def smooth_skin(
    image: np.ndarray,
    skin_mask: np.ndarray,
    strength: float,
    detail_strength: float = 0.30,
) -> np.ndarray:
    """Edge-preserving smoothing followed by controlled high-frequency restoration."""

    smoothing_strength = clamp01(strength)
    texture_strength = clamp01(detail_strength)
    if smoothing_strength <= 0.0:
        return image.copy()

    bilateral_sigma_color = 26.0 + 58.0 * smoothing_strength
    bilateral_sigma_space = 6.0 + 26.0 * smoothing_strength
    bilateral_filtered_image = cv2.bilateralFilter(
        image,
        d=0,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    detail_blur_sigma = 0.9 + 1.5 * smoothing_strength
    detail_blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=detail_blur_sigma)
    high_frequency_detail = image.astype(np.float32) - detail_blurred_image.astype(np.float32)

    detail_gain = 0.14 + 0.58 * texture_strength
    restored_image = bilateral_filtered_image.astype(np.float32) + high_frequency_detail * detail_gain
    restored_image = np.clip(restored_image, 0.0, 255.0).astype(np.uint8)
    return blend_with_mask(image, restored_image, skin_mask, smoothing_strength)


def whiten_skin(
    image: np.ndarray,
    skin_mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Local Lab tone lift + mild chroma neutralization + soft gamma correction."""

    whitening_strength = clamp01(strength)
    if whitening_strength <= 0.0:
        return image.copy()

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness_channel = lab_image[:, :, 0]
    green_red_channel = lab_image[:, :, 1]
    blue_yellow_channel = lab_image[:, :, 2]

    available_highlight_range = 255.0 - lightness_channel
    lightness_lift_ratio = 0.06 + 0.18 * whitening_strength
    lightness_channel += available_highlight_range * lightness_lift_ratio

    green_red_neutralization_ratio = 0.015 + 0.045 * whitening_strength
    blue_yellow_neutralization_ratio = 0.012 + 0.040 * whitening_strength
    green_red_channel += (128.0 - green_red_channel) * green_red_neutralization_ratio
    blue_yellow_channel += (128.0 - blue_yellow_channel) * blue_yellow_neutralization_ratio

    lab_image[:, :, 0] = np.clip(lightness_channel, 0.0, 255.0)
    lab_image[:, :, 1] = np.clip(green_red_channel, 0.0, 255.0)
    lab_image[:, :, 2] = np.clip(blue_yellow_channel, 0.0, 255.0)

    tone_lifted_image = cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2BGR)
    gamma_value = max(0.82, 1.0 - 0.14 * whitening_strength)
    normalized_image = tone_lifted_image.astype(np.float32) / 255.0
    gamma_corrected_image = np.power(normalized_image, gamma_value) * 255.0
    gamma_corrected_image = np.clip(gamma_corrected_image, 0.0, 255.0).astype(np.uint8)

    return blend_with_mask(image, gamma_corrected_image, skin_mask, whitening_strength)


def restore_feature_details(
    image: np.ndarray,
    geometry_mask: np.ndarray,
    skin_mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Sharpens protected facial details without globally sharpening the photograph."""

    detail_strength = clamp01(strength)
    if detail_strength <= 0.0:
        return image.copy()

    facial_detail_mask = cv2.subtract(geometry_mask, skin_mask)
    facial_detail_mask = cv2.GaussianBlur(facial_detail_mask, (7, 7), 0)

    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharpening_amount = 0.35 + 0.45 * detail_strength
    sharpened_image = cv2.addWeighted(
        image,
        1.0 + sharpening_amount,
        blurred_image,
        -sharpening_amount,
        0.0,
    )
    return blend_with_mask(image, sharpened_image, facial_detail_mask, 0.55 * detail_strength)


def choose_side_anchor(
    landmarks: np.ndarray,
    bounding_box: np.ndarray,
    side: str,
    minimum_y_ratio: float,
    maximum_y_ratio: float,
) -> np.ndarray:
    """Finds a stable outer-face anchor without hard-coding landmark indexes."""

    left_x, top_y, right_x, bottom_y = [float(value) for value in bounding_box[:4]]
    face_width = max(1.0, right_x - left_x)
    face_height = max(1.0, bottom_y - top_y)
    face_center_x = (left_x + right_x) * 0.5
    minimum_y = top_y + face_height * minimum_y_ratio
    maximum_y = top_y + face_height * maximum_y_ratio

    candidate_points: list[np.ndarray] = []
    for landmark_point in landmarks:
        point_x = float(landmark_point[0])
        point_y = float(landmark_point[1])
        if point_y < minimum_y or point_y > maximum_y:
            continue
        if side == "left" and point_x < face_center_x - face_width * 0.05:
            candidate_points.append(landmark_point)
        if side == "right" and point_x > face_center_x + face_width * 0.05:
            candidate_points.append(landmark_point)

    fallback_y = top_y + face_height * ((minimum_y_ratio + maximum_y_ratio) * 0.5)
    if len(candidate_points) == 0:
        fallback_x = left_x + face_width * 0.16
        if side == "right":
            fallback_x = right_x - face_width * 0.16
        return np.array([fallback_x, fallback_y], dtype=np.float32)

    candidate_array = np.asarray(candidate_points, dtype=np.float32)
    sorted_indices = np.argsort(candidate_array[:, 0])
    if side == "right":
        sorted_indices = np.argsort(-candidate_array[:, 0])

    selected_count = min(4, len(sorted_indices))
    selected_points = candidate_array[sorted_indices[:selected_count]]
    return np.mean(selected_points, axis=0)


def local_translation_warp(
    image: np.ndarray,
    source_point: np.ndarray,
    target_point: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Performs a smooth inverse-mapped local translation with radial falloff."""

    image_height, image_width = image.shape[:2]
    warp_radius = max(8.0, float(radius))
    translation_vector = target_point.astype(np.float32) - source_point.astype(np.float32)
    if float(np.linalg.norm(translation_vector)) < 0.5:
        return image.copy()

    warp_center = source_point.astype(np.float32)
    region_left = max(0, int(np.floor(warp_center[0] - warp_radius)))
    region_top = max(0, int(np.floor(warp_center[1] - warp_radius)))
    region_right = min(image_width, int(np.ceil(warp_center[0] + warp_radius + 1.0)))
    region_bottom = min(image_height, int(np.ceil(warp_center[1] + warp_radius + 1.0)))
    if region_right <= region_left or region_bottom <= region_top:
        return image.copy()

    source_region = image[region_top:region_bottom, region_left:region_right]
    grid_y, grid_x = np.indices(
        (region_bottom - region_top, region_right - region_left),
        dtype=np.float32,
    )
    absolute_x = grid_x + float(region_left)
    absolute_y = grid_y + float(region_top)

    x_distance = absolute_x - warp_center[0]
    y_distance = absolute_y - warp_center[1]
    distance_squared = x_distance * x_distance + y_distance * y_distance
    radius_squared = warp_radius * warp_radius

    inside_radius_mask = distance_squared < radius_squared
    radial_weight = np.zeros_like(distance_squared, dtype=np.float32)
    normalized_distance = np.zeros_like(distance_squared, dtype=np.float32)
    normalized_distance[inside_radius_mask] = distance_squared[inside_radius_mask] / radius_squared
    radial_weight[inside_radius_mask] = (1.0 - normalized_distance[inside_radius_mask]) ** 2

    map_x = absolute_x - translation_vector[0] * radial_weight - float(region_left)
    map_y = absolute_y - translation_vector[1] * radial_weight - float(region_top)

    warped_region = cv2.remap(
        source_region,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    output_image = image.copy()
    output_image[region_top:region_bottom, region_left:region_right] = warped_region
    return output_image


def slim_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    bounding_box: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Slims left/right cheek and jaw locally instead of scaling the whole face."""

    slimming_strength = clamp01(strength)
    if slimming_strength <= 0.0:
        return image.copy()

    left_x, _, right_x, _ = [float(value) for value in bounding_box[:4]]
    face_width = max(1.0, right_x - left_x)
    face_center_x = (left_x + right_x) * 0.5

    left_cheek_anchor = choose_side_anchor(landmarks, bounding_box, "left", 0.38, 0.62)
    right_cheek_anchor = choose_side_anchor(landmarks, bounding_box, "right", 0.38, 0.62)
    left_jaw_anchor = choose_side_anchor(landmarks, bounding_box, "left", 0.58, 0.82)
    right_jaw_anchor = choose_side_anchor(landmarks, bounding_box, "right", 0.58, 0.82)

    maximum_cheek_shift = face_width * 0.070 * slimming_strength
    maximum_jaw_shift = face_width * 0.050 * slimming_strength

    left_cheek_target = left_cheek_anchor.copy()
    right_cheek_target = right_cheek_anchor.copy()
    left_jaw_target = left_jaw_anchor.copy()
    right_jaw_target = right_jaw_anchor.copy()

    left_cheek_space = max(0.0, face_center_x - float(left_cheek_anchor[0]))
    right_cheek_space = max(0.0, float(right_cheek_anchor[0]) - face_center_x)
    left_jaw_space = max(0.0, face_center_x - float(left_jaw_anchor[0]))
    right_jaw_space = max(0.0, float(right_jaw_anchor[0]) - face_center_x)

    left_cheek_target[0] += min(maximum_cheek_shift, left_cheek_space * 0.22)
    right_cheek_target[0] -= min(maximum_cheek_shift, right_cheek_space * 0.22)
    left_jaw_target[0] += min(maximum_jaw_shift, left_jaw_space * 0.18)
    right_jaw_target[0] -= min(maximum_jaw_shift, right_jaw_space * 0.18)

    cheek_warp_radius = face_width * (0.22 + 0.03 * slimming_strength)
    jaw_warp_radius = face_width * (0.20 + 0.03 * slimming_strength)

    output_image = local_translation_warp(
        image,
        left_cheek_anchor,
        left_cheek_target,
        cheek_warp_radius,
    )
    output_image = local_translation_warp(
        output_image,
        right_cheek_anchor,
        right_cheek_target,
        cheek_warp_radius,
    )
    output_image = local_translation_warp(
        output_image,
        left_jaw_anchor,
        left_jaw_target,
        jaw_warp_radius,
    )
    output_image = local_translation_warp(
        output_image,
        right_jaw_anchor,
        right_jaw_target,
        jaw_warp_radius,
    )
    return output_image


class BeautyProcessor:
    """Reusable high-level processor. Create once, then process many images."""

    def __init__(self, config: BeautyConfig | None = None) -> None:
        self.config = config if config is not None else BeautyConfig()
        self.face_analyzer = FaceAnalyzer(self.config)

    def _face_area(self, face: Any) -> float:
        bounding_box = np.asarray(face.bbox, dtype=np.float32)
        return calculate_bounding_box_area(bounding_box)

    def _select_faces(self, image: np.ndarray) -> list[Any]:
        detected_faces = self.face_analyzer.detect(image)
        detected_faces.sort(key=self._face_area, reverse=True)

        selected_faces: list[Any] = []
        for detected_face in detected_faces:
            if get_landmarks_106(detected_face) is None:
                continue

            selected_faces.append(detected_face)
            if self.config.max_faces > 0 and len(selected_faces) >= self.config.max_faces:
                break

        return selected_faces

    def _apply_surface_effects(self, image: np.ndarray, face: Any) -> np.ndarray:
        landmarks = get_landmarks_106(face)
        if landmarks is None:
            return image.copy()

        keypoints = get_five_keypoints(face)
        bounding_box = np.asarray(face.bbox, dtype=np.float32)
        image_height, image_width = image.shape[:2]
        region_left, region_top, region_right, region_bottom = expand_face_region(
            bounding_box,
            image_width,
            image_height,
        )
        if region_right <= region_left or region_bottom <= region_top:
            return image.copy()

        full_geometry_mask = create_geometry_face_mask(image.shape, landmarks, bounding_box)
        full_skin_mask = create_precise_skin_mask(
            image,
            landmarks,
            keypoints,
            bounding_box,
            self.config.mask_feather_ratio,
        )

        output_image = image.copy()
        face_region = output_image[region_top:region_bottom, region_left:region_right].copy()
        geometry_mask = full_geometry_mask[region_top:region_bottom, region_left:region_right]
        skin_mask = full_skin_mask[region_top:region_bottom, region_left:region_right]

        face_region = smooth_skin(
            face_region,
            skin_mask,
            self.config.smooth_strength,
            self.config.detail_strength,
        )
        face_region = whiten_skin(
            face_region,
            skin_mask,
            self.config.whiten_strength,
        )
        face_region = restore_feature_details(
            face_region,
            geometry_mask,
            skin_mask,
            self.config.detail_strength,
        )

        output_image[region_top:region_bottom, region_left:region_right] = face_region
        return output_image

    def beautify_image(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        """Beautifies a BGR uint8 image and never modifies the input array in place."""

        if image is None or image.size == 0:
            raise ValueError("image must be a non-empty BGR image")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape H x W x 3")
        if image.dtype != np.uint8:
            raise ValueError("image dtype must be uint8")

        source_image = image.copy()
        selected_faces = self._select_faces(source_image)
        if len(selected_faces) == 0:
            return source_image, 0

        output_image = source_image.copy()

        # Pixel-level effects run first while the original landmark coordinates remain valid.
        for selected_face in selected_faces:
            output_image = self._apply_surface_effects(output_image, selected_face)

        # Geometry changes run last so the previously built masks do not become misaligned.
        for selected_face in selected_faces:
            landmarks = get_landmarks_106(selected_face)
            if landmarks is None:
                continue

            bounding_box = np.asarray(selected_face.bbox, dtype=np.float32)
            output_image = slim_face(
                output_image,
                landmarks,
                bounding_box,
                self.config.slim_strength,
            )

        return output_image, len(selected_faces)


def beautify_image(
    image: np.ndarray,
    processor: BeautyProcessor,
) -> tuple[np.ndarray, int]:
    """Functional wrapper for projects that prefer a def-style API."""

    return processor.beautify_image(image)


# -----------------------------------------------------------------------------
# Safe image file I/O
# -----------------------------------------------------------------------------


def read_image(image_path: Path) -> np.ndarray | None:
    """Reads without modifying the file and supports Unicode paths on Windows."""

    if not is_image_file(image_path):
        return None

    encoded_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if encoded_bytes.size == 0:
        return None
    return cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR)


def write_image(
    output_path: Path,
    image: np.ndarray,
    config: BeautyConfig,
) -> bool:
    """Writes only to the provided output path; callers always resolve a non-existing path first."""

    ensure_directory(output_path.parent)
    output_extension = output_path.suffix.lower()
    if output_extension not in SUPPORTED_IMAGE_EXTENSIONS:
        return False

    encoding_parameters: list[int] = []
    if output_extension in (".jpg", ".jpeg"):
        encoding_parameters = [cv2.IMWRITE_JPEG_QUALITY, int(config.jpeg_quality)]
    elif output_extension == ".png":
        encoding_parameters = [cv2.IMWRITE_PNG_COMPRESSION, int(config.png_compression)]
    elif output_extension == ".webp":
        encoding_parameters = [cv2.IMWRITE_WEBP_QUALITY, int(config.jpeg_quality)]

    encoding_success, encoded_image = cv2.imencode(
        output_extension,
        image,
        encoding_parameters,
    )
    if not encoding_success:
        return False

    encoded_image.tofile(str(output_path))
    return True


def resolve_single_output_path(
    input_path: Path,
    output: Path,
    output_suffix: str,
) -> Path:
    """Treats an image-suffixed output as a file; otherwise treats it as a directory."""

    if output.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        ensure_directory(output.parent)
        return unique_output_path(output)

    ensure_directory(output)
    candidate = output / f"{input_path.stem}{output_suffix}{input_path.suffix}"
    return unique_output_path(candidate)


def process_image_file(
    input_path: Path | str,
    output: Path | str,
    processor: BeautyProcessor,
    output_suffix: str = "_beauty",
) -> ProcessResult:
    """Processes one file and always writes a new file instead of overwriting the input."""

    source_path = Path(input_path)
    output_target = Path(output)

    source_image = read_image(source_path)
    if source_image is None:
        return ProcessResult(message=f"Cannot decode image: {source_path}")

    beautified_image, processed_face_count = processor.beautify_image(source_image)
    output_path = resolve_single_output_path(source_path, output_target, output_suffix)

    if output_path.resolve() == source_path.resolve():
        output_path = unique_output_path(output_path)

    write_success = write_image(output_path, beautified_image, processor.config)
    if not write_success:
        return ProcessResult(
            face_count=processed_face_count,
            message=f"Cannot write image: {output_path}",
        )

    return ProcessResult(
        output_path=output_path,
        face_count=processed_face_count,
        success=True,
        message="ok",
    )


# -----------------------------------------------------------------------------
# Batch discovery and processing
# -----------------------------------------------------------------------------


def discover_image_files(
    input_path: Path,
    recursive: bool,
    excluded_directory: Path | None = None,
) -> list[Path]:
    """Finds input images before processing starts, preventing output files from joining the same run."""

    if input_path.is_file():
        if is_image_file(input_path):
            return [input_path]
        return []

    file_iterator: Iterable[Path]
    if recursive:
        file_iterator = input_path.rglob("*")
    else:
        file_iterator = input_path.iterdir()

    image_paths: list[Path] = []
    for candidate_path in file_iterator:
        if not is_image_file(candidate_path):
            continue
        if excluded_directory is not None and is_path_inside(candidate_path, excluded_directory):
            continue
        image_paths.append(candidate_path)

    image_paths.sort()
    return image_paths


def build_batch_output_path(
    source_path: Path,
    input_root: Path,
    output_root: Path,
    output_suffix: str,
) -> Path:
    """Preserves source subfolders under a separate output root."""

    relative_source_path = source_path.relative_to(input_root)
    output_filename = (
        f"{relative_source_path.stem}{output_suffix}{relative_source_path.suffix}"
    )
    output_path = output_root / relative_source_path.parent / output_filename
    ensure_directory(output_path.parent)
    return unique_output_path(output_path)


def process_batch(
    input_directory: Path | str,
    output_directory: Path | str,
    processor: BeautyProcessor,
    recursive: bool = True,
    output_suffix: str = "_beauty",
    verbose: bool = True,
) -> BatchSummary:
    """Batch-processes a directory tree while preserving the original directory structure."""

    input_root = Path(input_directory)
    output_root = Path(output_directory)
    if input_root.resolve() == output_root.resolve():
        raise ValueError("Batch output directory must be different from the input directory")

    ensure_directory(output_root)

    excluded_output_directory: Path | None = None
    if is_path_inside(output_root, input_root):
        excluded_output_directory = output_root

    image_paths = discover_image_files(
        input_root,
        recursive,
        excluded_output_directory,
    )
    batch_summary = BatchSummary(total_files=len(image_paths))

    for file_index, image_path in enumerate(image_paths, start=1):
        if verbose:
            print(f"[{file_index}/{batch_summary.total_files}] {image_path}")

        source_image = read_image(image_path)
        if source_image is None:
            batch_summary.skipped_files += 1
            if verbose:
                print("  skipped: cannot decode")
            continue

        beautified_image, processed_face_count = processor.beautify_image(source_image)
        output_path = build_batch_output_path(
            image_path,
            input_root,
            output_root,
            output_suffix,
        )

        if output_path.resolve() == image_path.resolve():
            output_path = unique_output_path(output_path)

        write_success = write_image(output_path, beautified_image, processor.config)
        if not write_success:
            batch_summary.skipped_files += 1
            if verbose:
                print("  skipped: cannot write")
            continue

        batch_summary.success_files += 1
        if processed_face_count == 0:
            batch_summary.no_face_files += 1

        if verbose:
            print(f"  saved: {output_path}    faces={processed_face_count}")

    return batch_summary


def process_path(
    input_path: Path | str,
    output: Path | str,
    processor: BeautyProcessor,
    recursive: bool = True,
    output_suffix: str = "_beauty",
    verbose: bool = True,
) -> ProcessResult | BatchSummary:
    """Unified public entry: accepts either one image file or one image directory."""

    source_path = Path(input_path)
    if source_path.is_file():
        return process_image_file(source_path, output, processor, output_suffix)
    if source_path.is_dir():
        return process_batch(
            source_path,
            output,
            processor,
            recursive,
            output_suffix,
            verbose,
        )
    raise ValueError(f"Input path does not exist: {source_path}")


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InsightFace-assisted offline beauty processor for one image or a folder of images"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image file or folder")
    parser.add_argument("--output", type=Path, default=Path("beauty_output"), help="New output file or folder")
    parser.add_argument("--no-recursive", action="store_true", help="Do not scan subfolders")
    parser.add_argument("--output-suffix", default="_beauty", help="Suffix added before the extension")

    parser.add_argument("--smooth", type=float, default=0.62, help="Skin smoothing strength, 0..1")
    parser.add_argument("--whiten", type=float, default=0.24, help="Skin whitening strength, 0..1")
    parser.add_argument("--slim", type=float, default=0.16, help="Face slimming strength, 0..1")
    parser.add_argument("--detail", type=float, default=0.30, help="Detail restoration strength, 0..1")
    parser.add_argument("--max-faces", type=int, default=0, help="Maximum faces, 0 means all")
    parser.add_argument("--det-size", type=int, default=640, help="InsightFace detection size")
    parser.add_argument("--det-threshold", type=float, default=0.50, help="Face detection threshold")
    return parser.parse_args()


def build_config(arguments: argparse.Namespace) -> BeautyConfig:
    config = BeautyConfig()
    config.smooth_strength = clamp01(arguments.smooth)
    config.whiten_strength = clamp01(arguments.whiten)
    config.slim_strength = clamp01(arguments.slim)
    config.detail_strength = clamp01(arguments.detail)
    config.max_faces = max(0, int(arguments.max_faces))
    config.detection_size = max(320, int(arguments.det_size))
    config.detection_threshold = clamp01(arguments.det_threshold)
    return config


def print_batch_summary(summary: BatchSummary, output: Path) -> None:
    print("Batch finished")
    print(f"  total   : {summary.total_files}")
    print(f"  success : {summary.success_files}")
    print(f"  skipped : {summary.skipped_files}")
    print(f"  no face : {summary.no_face_files}")
    print(f"  output  : {output}")


def main() -> None:
    arguments = parse_arguments()
    if not arguments.input.exists():
        raise ValueError(f"Input path does not exist: {arguments.input}")

    config = build_config(arguments)
    processor = BeautyProcessor(config)
    processing_result = process_path(
        input_path=arguments.input,
        output=arguments.output,
        processor=processor,
        recursive=not arguments.no_recursive,
        output_suffix=arguments.output_suffix,
        verbose=True,
    )

    if isinstance(processing_result, ProcessResult):
        if not processing_result.success:
            raise RuntimeError(processing_result.message)
        print(f"Saved: {processing_result.output_path}")
        print(f"Faces processed: {processing_result.face_count}")
        return

    print_batch_summary(processing_result, arguments.output)


if __name__ == "__main__":
    main()
