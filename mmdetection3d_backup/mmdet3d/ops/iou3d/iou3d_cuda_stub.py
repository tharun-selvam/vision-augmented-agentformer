"""
Stub for iou3d_cuda when CUDA extensions are not compiled.
Provides minimal interface to avoid import errors.
"""
import warnings


def boxes_overlap_bev_gpu(*args, **kwargs):
    """Stub for boxes_overlap_bev_gpu."""
    raise NotImplementedError(
        "iou3d_cuda extensions not compiled. "
        "This operation requires CUDA compilation."
    )


def nms_gpu(*args, **kwargs):
    """Stub for nms_gpu."""
    raise NotImplementedError(
        "iou3d_cuda extensions not compiled. "
        "This operation requires CUDA compilation."
    )


def nms_normal_gpu(*args, **kwargs):
    """Stub for nms_normal_gpu."""
    raise NotImplementedError(
        "iou3d_cuda extensions not compiled. "
        "This operation requires CUDA compilation."
    )


warnings.warn(
    "Using iou3d_cuda stub - CUDA extensions not compiled. "
    "Some 3D operations will not be available.",
    UserWarning
)
