import warnings
import sys

# Try to import the compiled CUDA extension
try:
    from . import iou3d_cuda
except ImportError:
    warnings.warn("iou3d_cuda CUDA extension not available, using stub")
    # Import the stub module
    from . import iou3d_cuda_stub as iou3d_cuda
    sys.modules['mmdet3d.ops.iou3d.iou3d_cuda'] = iou3d_cuda

try:
    from .iou3d_utils import boxes_iou_bev, nms_gpu, nms_normal_gpu
except ImportError:
    warnings.warn("iou3d_utils not available, using stubs")
    boxes_iou_bev = None
    nms_gpu = None
    nms_normal_gpu = None

__all__ = ['boxes_iou_bev', 'nms_gpu', 'nms_normal_gpu', 'iou3d_cuda']
