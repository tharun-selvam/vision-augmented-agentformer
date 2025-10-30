# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmdet

# Make mmseg optional
try:
    import mmseg
    HAS_MMSEG = True
except ImportError:
    HAS_MMSEG = False
    import warnings
    warnings.warn("mmseg not installed, some features may not be available")

from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.3.8'
mmcv_maximum_version = '2.0.0'  # Extended to support mmcv-full 1.7.x
mmcv_version = digit_version(mmcv.__version__)


# Relaxed version check for compatibility
if not (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)):
    import warnings
    warnings.warn(f'MMCV=={mmcv.__version__} may be incompatible. '
                  f'Recommended: mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.')

mmdet_minimum_version = '2.14.0'
mmdet_maximum_version = '3.0.0'
mmdet_version = digit_version(mmdet.__version__)
# Relaxed version check
if not (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version <= digit_version(mmdet_maximum_version)):
    import warnings
    warnings.warn(f'MMDET=={mmdet.__version__} may be incompatible. '
                  f'Recommended: mmdet>={mmdet_minimum_version}, <={mmdet_maximum_version}.')

if HAS_MMSEG:
    mmseg_minimum_version = '0.14.1'
    mmseg_maximum_version = '1.0.0'
    mmseg_version = digit_version(mmseg.__version__)
    # Relaxed version check
    if not (mmseg_version >= digit_version(mmseg_minimum_version)
            and mmseg_version <= digit_version(mmseg_maximum_version)):
        import warnings
        warnings.warn(f'MMSEG=={mmseg.__version__} may be incompatible. '
                      f'Recommended: mmseg>={mmseg_minimum_version}, <={mmseg_maximum_version}.')

__all__ = ['__version__', 'short_version']
