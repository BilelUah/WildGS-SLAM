from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension('droid_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/lietorch/eigen')],
            sources=[
                'src/lib/droid.cpp',
                'src/lib/droid_kernels.cu',
                'src/lib/correlation_kernels.cu',
                'src/lib/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3',
                    '-gencode=arch=compute_120,code=sm_120',
                    '-gencode=arch=compute_120,code=compute_120',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
