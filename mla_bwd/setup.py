"""
Setup script for mla_bwd_cuda extension.
Compiles the CUDA kernel for testing mla_bwd on SM100 (Blackwell).
"""

import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get current directory and FlashMLA directory
this_dir = Path(__file__).parent.absolute()
flash_mla_dir = this_dir.parent.parent / "FlashMLA"

# Check FlashMLA directory exists
assert flash_mla_dir.exists(), f"FlashMLA directory not found: {flash_mla_dir}"

# NVCC compile flags
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
    # SM100 architecture
    "-gencode=arch=compute_100a,code=sm_100a",
    # Parallel compilation threads (max_jobs=192)
    "--threads", os.getenv("NVCC_THREADS", "192"),
]

# C++ compile flags
cxx_flags = [
    "-O3",
    "-std=c++20",
    "-DNDEBUG",
    "-Wno-deprecated-declarations",
]

# Include directories
include_dirs = [
    str(flash_mla_dir / "csrc"),
    str(flash_mla_dir / "csrc" / "kerutils" / "include"),
    str(flash_mla_dir / "csrc" / "cutlass" / "include"),
    str(flash_mla_dir / "csrc" / "cutlass" / "tools" / "util" / "include"),
]

# Verify include directories exist
for inc_dir in include_dirs:
    if not Path(inc_dir).exists():
        print(f"Warning: include directory not found: {inc_dir}")

ext_modules = [
    CUDAExtension(
        name="mla_bwd_cuda",
        sources=["mla_bwd.cu"],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        include_dirs=include_dirs,
    )
]

setup(
    name="mla_bwd_cuda",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
