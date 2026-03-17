#pragma once

#include <torch/extension.h>

namespace test_operator::dq_phase_test {

std::vector<torch::Tensor> run_dq_phase(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& o,
    const torch::Tensor& dO,
    const torch::Tensor& indices,
    const torch::Tensor& lse,
    float sm_scale,
    int q_start_index_s = 0
);

}  // namespace test_operator::dq_phase_test
