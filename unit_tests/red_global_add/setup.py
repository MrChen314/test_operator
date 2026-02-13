"""Build script for red_global_add_cuda."""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_flags = [
    "-O3",
    "-std=c++20",
    "-DNDEBUG",
    "-D_USE_MATH_DEFINES",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ptxas-options=-v",
    "-lineinfo",
    "-gencode=arch=compute_100a,code=sm_100a",
    "--threads",
    os.getenv("NVCC_THREADS", "192"),
]

cxx_flags = [
    "-O3",
    "-std=c++20",
    "-DNDEBUG",
    "-Wno-deprecated-declarations",
]

ext_modules = [
    CUDAExtension(
        name="red_global_add_cuda",
        sources=["test_red_global_add.cu"],
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    )
]

setup(
    name="red_global_add_cuda",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
