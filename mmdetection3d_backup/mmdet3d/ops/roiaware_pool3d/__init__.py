import warnings
import sys

# Create stub for roiaware_pool3d_ext
class RoiawarePool3dExtStub:
    """Stub for roiaware_pool3d_ext when not compiled."""
    @staticmethod
    def forward(*args, **kwargs):
        raise NotImplementedError("roiaware_pool3d_ext not compiled")

    @staticmethod
    def points_in_boxes_gpu(*args, **kwargs):
        raise NotImplementedError("roiaware_pool3d_ext not compiled")

    @staticmethod
    def points_in_boxes_cpu(*args, **kwargs):
        raise NotImplementedError("roiaware_pool3d_ext not compiled")

# Try to import compiled extension, use stub if not available
try:
    from . import roiaware_pool3d_ext
except ImportError:
    warnings.warn("roiaware_pool3d_ext not available, using stub")
    roiaware_pool3d_ext = RoiawarePool3dExtStub()
    sys.modules['mmdet3d.ops.roiaware_pool3d.roiaware_pool3d_ext'] = roiaware_pool3d_ext

try:
    from .points_in_boxes import (points_in_boxes_batch, points_in_boxes_cpu,
                                  points_in_boxes_gpu)
except ImportError:
    warnings.warn("points_in_boxes not available, using stubs")
    points_in_boxes_batch = None
    points_in_boxes_cpu = None
    points_in_boxes_gpu = None

try:
    from .roiaware_pool3d import RoIAwarePool3d
except ImportError:
    warnings.warn("RoIAwarePool3d not available, using stub")
    RoIAwarePool3d = None

__all__ = [
    'RoIAwarePool3d', 'points_in_boxes_gpu', 'points_in_boxes_cpu',
    'points_in_boxes_batch'
]
