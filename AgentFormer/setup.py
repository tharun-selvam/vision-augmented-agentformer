
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)

def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


setup(
    name='AgentFormer',
    version='0.1.0',
    author='Ye Yuan',
    author_email='ye.yuan@kit.edu',
    description='AgentFormer with BEVDepth integration',
    long_description='''This is an integration of BEVDepth's visual encoder into the AgentFormer model.''',
    long_description_content_type='text/markdown',
    url='https://github.com/Khrylx/AgentFormer',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[],
    ext_modules=[
        make_cuda_ext(
            name='voxel_pooling_train_ext',
            module='ops.voxel_pooling_train',
            sources=['src/voxel_pooling_train_forward.cpp'],
            sources_cuda=['src/voxel_pooling_train_forward_cuda.cu'],
        ),
        make_cuda_ext(
            name='voxel_pooling_inference_ext',
            module='ops.voxel_pooling_inference',
            sources=['src/voxel_pooling_inference_forward.cpp'],
            sources_cuda=['src/voxel_pooling_inference_forward_cuda.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
