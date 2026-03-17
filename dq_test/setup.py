"""Build script for dq_phase_cuda."""

import os
import shutil
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).parent.resolve()
flash_mla_dir = this_dir.parent.parent / "FlashMLA"
assert flash_mla_dir.exists(), f"FlashMLA directory not found: {flash_mla_dir}"

if "CUDA_HOME" not in os.environ:
    nvcc_path = shutil.which("nvcc")
    cuda_candidates = [
        Path(nvcc_path).resolve().parent.parent if nvcc_path else None,
        Path("/usr/local/cuda"),
        Path("/usr/local/cuda-12.8"),
        Path("/usr/local/cuda-12.6"),
        Path("/opt/cuda"),
    ]
    for candidate in cuda_candidates:
        if candidate and candidate.exists():
            os.environ["CUDA_HOME"] = str(candidate)
            break

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

include_dirs = [
    str(this_dir),
    str(flash_mla_dir / "csrc"),
    str(flash_mla_dir / "csrc" / "kerutils" / "include"),
    str(flash_mla_dir / "csrc" / "cutlass" / "include"),
    str(flash_mla_dir / "csrc" / "cutlass" / "tools" / "util" / "include"),
]

ext_modules = [
    CUDAExtension(
        name="dq_phase_cuda",
        sources=["dq_phase_test.cu"],
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        include_dirs=include_dirs,
    )
]

setup(
    name="dq_phase_cuda",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
