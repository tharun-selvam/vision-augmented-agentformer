# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.ops import (RoIAlign, SigmoidFocalLoss, get_compiler_version,
                      get_compiling_cuda_version, nms, roi_align,
                      sigmoid_focal_loss)

from .norm import NaiveSyncBatchNorm1d, NaiveSyncBatchNorm2d

# Try to import CUDA ops, provide stubs if not available
try:
    from .ball_query import ball_query
except ImportError:
    warnings.warn("ball_query CUDA extension not available, using stub")
    ball_query = None

try:
    from .furthest_point_sample import (Points_Sampler, furthest_point_sample,
                                        furthest_point_sample_with_dist)
except ImportError:
    warnings.warn("furthest_point_sample CUDA extension not available, using stub")
    Points_Sampler = None
    furthest_point_sample = None
    furthest_point_sample_with_dist = None

try:
    from .gather_points import gather_points
except ImportError:
    warnings.warn("gather_points CUDA extension not available, using stub")
    gather_points = None

try:
    from .group_points import (GroupAll, QueryAndGroup, group_points,
                               grouping_operation)
except ImportError:
    warnings.warn("group_points CUDA extension not available, using stub")
    GroupAll = None
    QueryAndGroup = None
    group_points = None
    grouping_operation = None

try:
    from .interpolate import three_interpolate, three_nn
except ImportError:
    warnings.warn("interpolate CUDA extension not available, using stub")
    three_interpolate = None
    three_nn = None

try:
    from .knn import knn
except ImportError:
    warnings.warn("knn CUDA extension not available, using stub")
    knn = None

try:
    from .paconv import PAConv, PAConvCUDA, assign_score_withk
except ImportError:
    warnings.warn("paconv CUDA extension not available, using stub")
    PAConv = None
    PAConvCUDA = None
    assign_score_withk = None

try:
    from .pointnet_modules import (PAConvCUDASAModule, PAConvCUDASAModuleMSG,
                                   PAConvSAModule, PAConvSAModuleMSG,
                                   PointFPModule, PointSAModule, PointSAModuleMSG,
                                   build_sa_module)
except ImportError:
    warnings.warn("pointnet_modules CUDA extension not available, using stub")
    PAConvCUDASAModule = None
    PAConvCUDASAModuleMSG = None
    PAConvSAModule = None
    PAConvSAModuleMSG = None
    PointFPModule = None
    PointSAModule = None
    PointSAModuleMSG = None
    build_sa_module = None

try:
    from .roiaware_pool3d import (RoIAwarePool3d, points_in_boxes_batch,
                                  points_in_boxes_cpu, points_in_boxes_gpu)
except ImportError:
    warnings.warn("roiaware_pool3d CUDA extension not available, using stub")
    RoIAwarePool3d = None
    points_in_boxes_batch = None
    points_in_boxes_cpu = None
    points_in_boxes_gpu = None

try:
    from .sparse_block import (SparseBasicBlock, SparseBottleneck,
                               make_sparse_convmodule)
except ImportError:
    warnings.warn("sparse_block CUDA extension not available, using stub")
    SparseBasicBlock = None
    SparseBottleneck = None
    make_sparse_convmodule = None

try:
    from .voxel import DynamicScatter, Voxelization, dynamic_scatter, voxelization
except ImportError:
    warnings.warn("voxel CUDA extension not available, using stub")
    DynamicScatter = None
    Voxelization = None
    dynamic_scatter = None
    voxelization = None

__all__ = [
    'nms', 'soft_nms', 'RoIAlign', 'roi_align', 'get_compiler_version',
    'get_compiling_cuda_version', 'NaiveSyncBatchNorm1d',
    'NaiveSyncBatchNorm2d', 'batched_nms', 'Voxelization', 'voxelization',
    'dynamic_scatter', 'DynamicScatter', 'sigmoid_focal_loss',
    'SigmoidFocalLoss', 'SparseBasicBlock', 'SparseBottleneck',
    'RoIAwarePool3d', 'points_in_boxes_gpu', 'points_in_boxes_cpu',
    'make_sparse_convmodule', 'ball_query', 'knn', 'furthest_point_sample',
    'furthest_point_sample_with_dist', 'three_interpolate', 'three_nn',
    'gather_points', 'grouping_operation', 'group_points', 'GroupAll',
    'QueryAndGroup', 'PointSAModule', 'PointSAModuleMSG', 'PointFPModule',
    'points_in_boxes_batch', 'get_compiler_version', 'assign_score_withk',
    'get_compiling_cuda_version', 'Points_Sampler', 'build_sa_module',
    'PAConv', 'PAConvCUDA', 'PAConvSAModuleMSG', 'PAConvSAModule',
    'PAConvCUDASAModule', 'PAConvCUDASAModuleMSG'
]
