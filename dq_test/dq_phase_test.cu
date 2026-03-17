#include "dq_phase_test.cuh"

#include <ATen/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>
#include <torch/extension.h>

#include "params.h"
#include "dq_phase_capture.cuh"

namespace test_operator::dq_phase_test {

namespace {

using bf16 = cutlass::bfloat16_t;

int as_int(int64_t value, const char* name) {
    TORCH_CHECK(value >= std::numeric_limits<int>::min() && value <= std::numeric_limits<int>::max(),
                name, " is out of int32 range: ", value);
    return static_cast<int>(value);
}

}  // namespace

std::vector<torch::Tensor> run_dq_phase(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& o,
    const torch::Tensor& dO,
    const torch::Tensor& indices,
    const torch::Tensor& lse,
    float sm_scale,
    int q_start_index_s
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(kv.is_cuda(), "kv must be a CUDA tensor");
    TORCH_CHECK(o.is_cuda(), "o must be a CUDA tensor");
    TORCH_CHECK(dO.is_cuda(), "dO must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(lse.is_cuda(), "lse must be a CUDA tensor");

    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kv.scalar_type() == torch::kBFloat16, "kv must be bfloat16");
    TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "o must be bfloat16");
    TORCH_CHECK(dO.scalar_type() == torch::kBFloat16, "dO must be bfloat16");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be float32");

    TORCH_CHECK(q.dim() == 3, "q must have shape [s_q, h_q, d_qk]");
    TORCH_CHECK(kv.dim() == 3, "kv must have shape [s_kv, h_kv, d_qk]");
    TORCH_CHECK(o.dim() == 3, "o must have shape [s_q, h_q, d_v]");
    TORCH_CHECK(dO.dim() == 3, "dO must have shape [s_q, h_q, d_v]");
    TORCH_CHECK(indices.dim() == 3, "indices must have shape [s_q, h_kv, topk]");
    TORCH_CHECK(lse.dim() == 2, "lse must have shape [s_q, h_q]");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");

    const int s_q = as_int(q.size(0), "q.size(0)");
    const int h_q = as_int(q.size(1), "q.size(1)");
    const int d_qk = as_int(q.size(2), "q.size(2)");
    const int s_kv = as_int(kv.size(0), "kv.size(0)");
    const int h_kv = as_int(kv.size(1), "kv.size(1)");
    const int d_k = as_int(kv.size(2), "kv.size(2)");
    const int d_v = as_int(dO.size(2), "dO.size(2)");
    const int topk = as_int(indices.size(2), "indices.size(2)");

    TORCH_CHECK(h_q == 128, "Only h_q=128 is supported, got ", h_q);
    TORCH_CHECK(h_kv == 1, "Only h_kv=1 is supported, got ", h_kv);
    TORCH_CHECK(d_qk == 576, "Only d_qk=576 is supported, got ", d_qk);
    TORCH_CHECK(d_k == 576, "Only kv last dim 576 is supported, got ", d_k);
    TORCH_CHECK(d_v == 512, "Only d_v=512 is supported, got ", d_v);
    TORCH_CHECK(topk == sm100::bwd::head128_2kernels::dq::B_TOPK,
                "dq_test currently fixes topk to ", sm100::bwd::head128_2kernels::dq::B_TOPK,
                ", got ", topk);
    TORCH_CHECK(o.sizes() == dO.sizes(), "o and dO must have the same shape");
    TORCH_CHECK(o.size(0) == s_q && o.size(1) == h_q, "o shape must match q on [s_q, h_q]");
    TORCH_CHECK(indices.size(0) == s_q && indices.size(1) == h_kv, "indices shape mismatch");
    TORCH_CHECK(lse.size(0) == s_q && lse.size(1) == h_q, "lse shape mismatch");

    at::cuda::CUDAGuard device_guard{q.device()};
    auto opts_bf16 = q.options().dtype(torch::kBFloat16);
    auto opts_f32 = q.options().dtype(torch::kFloat32);

    auto dQ = torch::empty({s_q, h_q, d_qk}, opts_bf16);
    auto s_out = torch::empty({s_q, h_q, topk}, opts_bf16);
    auto ds_out = torch::empty({s_q, h_q, topk}, opts_bf16);
    auto delta = torch::empty({s_q, h_q}, opts_f32);

    SparseAttnBwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        q_start_index_s,
        sm_scale, sm_scale * 1.4426950408889634f,

        reinterpret_cast<bf16*>(q.data_ptr()),
        reinterpret_cast<bf16*>(kv.data_ptr()),
        reinterpret_cast<bf16*>(o.data_ptr()),
        reinterpret_cast<bf16*>(dO.data_ptr()),
        indices.data_ptr<int>(),
        lse.data_ptr<float>(),
        nullptr,

        as_int(q.stride(0), "q.stride(0)"),
        as_int(q.stride(1), "q.stride(1)"),
        as_int(kv.stride(0), "kv.stride(0)"),
        as_int(kv.stride(1), "kv.stride(1)"),
        as_int(o.stride(0), "o.stride(0)"),
        as_int(o.stride(1), "o.stride(1)"),
        as_int(dO.stride(0), "dO.stride(0)"),
        as_int(dO.stride(1), "dO.stride(1)"),
        as_int(indices.stride(0), "indices.stride(0)"),
        as_int(indices.stride(1), "indices.stride(1)"),

        reinterpret_cast<bf16*>(dQ.data_ptr()),
        nullptr,
        delta.data_ptr<float>(),
        as_int(dQ.stride(0), "dQ.stride(0)"),
        as_int(dQ.stride(1), "dQ.stride(1)"),
        0,
        0,
        as_int(delta.stride(0), "delta.stride(0)"),
        as_int(delta.stride(1), "delta.stride(1)"),

        0,
        at::cuda::getCurrentCUDAStream().stream(),
    };

    sm100::bwd::head128_2kernels::dq::run_bwd_dq_phase_capture_kernel<576>(
        params,
        reinterpret_cast<bf16*>(s_out.data_ptr()),
        reinterpret_cast<bf16*>(ds_out.data_ptr()),
        as_int(s_out.stride(0), "s_out.stride(0)"),
        as_int(s_out.stride(1), "s_out.stride(1)"),
        as_int(ds_out.stride(0), "ds_out.stride(0)"),
        as_int(ds_out.stride(1), "ds_out.stride(1)")
    );

    return {dQ, s_out, ds_out};
}

}  // namespace test_operator::dq_phase_test

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_dq_phase", &test_operator::dq_phase_test::run_dq_phase, "Run dq_phase and capture dq/s/ds");
}
