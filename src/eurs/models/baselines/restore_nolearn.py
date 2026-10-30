"""학습 없는 복원 기법(복원 baseline 하한).

- ``identity``: 입력을 그대로 반환.
- ``grayworld_clahe``: LAB L채널 CLAHE(clip 2.0, tile 8x8) 후 gray-world 채널 게인.

입출력은 RGB. uint8(0~255) 또는 float(0~1) 배열을 받아 입력과 같은 dtype/스케일로
반환한다. 색공간 변환은 OpenCV(cv2)를 쓴다.
"""
import cv2
import numpy as np


def _to_uint8(image):
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr, False
    scaled = np.clip(arr.astype(np.float32), 0, 1) * 255
    return scaled.round().astype(np.uint8), True


def _restore_dtype(arr_uint8, was_float):
    if was_float:
        return arr_uint8.astype(np.float32) / 255
    return arr_uint8


def identity(image):
    """입력을 그대로 반환(복사)."""
    return np.array(image, copy=True)


def grayworld_clahe(image, clip_limit=2.0, tile=8):
    """LAB L채널 CLAHE 후 gray-world 채널 게인으로 색 편향을 보정한다."""
    rgb_u8, was_float = _to_uint8(image)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    means = balanced.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    gain = gray / np.clip(means, 1e-6, None)
    out = np.clip(balanced * gain[None, None, :], 0, 255).astype(np.uint8)
    return _restore_dtype(out, was_float)


METHODS = {"identity": identity, "grayworld_clahe": grayworld_clahe}
