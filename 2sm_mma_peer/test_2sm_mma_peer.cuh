#pragma once

#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>
#include <cutlass/arch/arch.h>

// Avoid using namespace cute to prevent conflicts with at::Layout
using bf16 = cutlass::bfloat16_t;

// Matrix dimension constants
static constexpr int M = 64;            // Q rows
static constexpr int N = 256;           // K columns (output columns), i.e., P columns
static constexpr int K_DIM = 128;       // Q columns (shared dimension for Q)
static constexpr int K_ROWS = 128;      // K rows (total, each CTA processes 64 rows)

// SMEM Layout definitions (from config.h)
// Q matrix SMEM layout: stores [M, K_DIM] = [64, 128]
template<int NUM_TILES>
using SmemLayoutQTiles = decltype(cute::coalesce(cute::tile_to_shape(
    cute::UMMA::Layout_K_SW128_Atom<bf16>{},
    cute::Shape<cute::Int<M>, cute::Int<64*NUM_TILES>>{},
    cute::Step<cute::_1, cute::_2>{}
), cute::Shape<cute::_1, cute::_1>{}));

using SmemLayoutQ = SmemLayoutQTiles<2>;  // NUM_TILES=2 for full Q: [64, 128] = [64, 64*2]

// K matrix SMEM layout: each CTA stores [N, K_ROWS/2] = [256, 64] (transposed for MMA)
// MMA performs Q @ K^T, so K stored as [256, 64] -> K^T = [64, 256]
// This allows [64, 64] @ [64, 256] = [64, 256]
template<int NUM_TILES>
using SmemLayoutKTiles = decltype(cute::coalesce(cute::tile_to_shape(
    cute::UMMA::Layout_K_SW128_Atom<bf16>{},
    cute::Shape<cute::Int<N>, cute::Int<64*NUM_TILES>>{},
    cute::Step<cute::_1, cute::_2>{}
), cute::Shape<cute::_1, cute::_1>{}));

// TMEM column configuration (from config.h)
struct tmem_cols {
    // P matrix stored in TMEM columns 256-511 (256 columns)
    // Output shape: [64, 256] float32, needs 256 columns
    static constexpr int p = 256;
    static_assert(p + 256 <= 512, "TMEM column overflow");
};

// TiledMMA_O definition (from config.h)
// Used to compute P = Q @ K^T, using WS_SS_NOELECT mode
using TiledMMA_O = decltype(cute::make_tiled_mma(
    cute::SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, M, N, cute::UMMA::Major::K, cute::UMMA::Major::K>{}
));

// Shared Memory structure
struct SharedMemory {
    alignas(128) bf16 q[M * K_DIM];         // Q: [64, 128]
    alignas(128) bf16 k[N * (K_ROWS/2)];    // K: [256, 64] per CTA (transposed storage)
    alignas(4) uint32_t tmem_start_addr;     // TMEM start address (allocated by Allocator2Sm)
};

// SMEM size
static constexpr size_t SMEM_SIZE = sizeof(SharedMemory);

// Thread count
static constexpr int NUM_THREADS = 128;  // 128 threads per CTA

// Kernel declaration (cluster dims specified at launch time)
__global__ void test_utcmma_ss_peer_kernel(
    const bf16* __restrict__ Q,      // [M, K_DIM] = [64, 128]
    const bf16* __restrict__ K,      // [K_ROWS, N] = [128, 256]
    float* __restrict__ P_out,       // [M, N] = [64, 256] (matrix multiplication result)
    bf16* __restrict__ Q_out,        // [M, K_DIM] = [64, 128] (output Q from SMEM)
    bf16* __restrict__ K_out,        // [K_ROWS, N] = [128, 256] (output K from SMEM, merged from cta0 and cta1)
    bf16* __restrict__ Q_first_half_out,  // [M, K_DIM/2] = [64, 64] (output Q first half from SMEM)
    bf16* __restrict__ Q_second_half_out  // [M, K_DIM/2] = [64, 64] (output Q second half from SMEM)
);

// C++ wrapper declaration
void launch_test_utcmma_ss_peer(
    const bf16* Q,
    const bf16* K,
    float* P_out,
    bf16* Q_out,
    bf16* K_out,
    bf16* Q_first_half_out,
    bf16* Q_second_half_out,
    cudaStream_t stream = nullptr
);
