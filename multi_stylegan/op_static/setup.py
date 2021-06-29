from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = []

nvcc_args = [
    '-gencode=arch=compute_60,code="sm_60,compute_60"', '-lineinfo',
]

setup(
    name='upfirdn2d_cuda',
    ext_modules=[
        CUDAExtension('upfirdn2d_cuda', [
            'upfirdn2d.cpp',
            'upfirdn2d_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='fused_act_cuda',
    ext_modules=[
        CUDAExtension('fused_act_cuda', [
            'fused_bias_act.cpp',
            'fused_bias_act_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
