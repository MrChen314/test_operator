#include "test_2sm_mma_peer.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <kerutils/kerutils.cuh>
#include <cutlass/arch/arch.h>

// Kernel implementation: test utcmma_ss matrix multiplication with peer SMEM access
__global__ void test_utcmma_ss_peer_kernel(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    float* __restrict__ P_out,
    bf16* __restrict__ Q_out,
    bf16* __restrict__ K_out,
    bf16* __restrict__ Q_first_half_out,
    bf16* __restrict__ Q_second_half_out
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);
    
    const int cta_idx = blockIdx.x % 2;  // 0 or 1
    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    const int lane_idx = tid % 32;
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 1: Kernel started\n", cta_idx);
    }
    
    // Construct SMEM Tensors
    auto sQ = cute::make_tensor(cute::make_smem_ptr(smem.q), SmemLayoutQ{});
    auto sK = cute::make_tensor(cute::make_smem_ptr(smem.k), SmemLayoutKTiles<1>{});
    
    // Load Q and K from Global Memory to SMEM
    // Q: [64, 128] - all CTAs load the same Q
    constexpr int Q_ELEMENTS = M * K_DIM;
    constexpr int Q_ELEMENTS_PER_THREAD = (Q_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;
    for (int i = 0; i < Q_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * Q_ELEMENTS_PER_THREAD + i;
        if (linear_idx < Q_ELEMENTS) {
            int row = linear_idx / K_DIM;
            int col = linear_idx % K_DIM;
            sQ(row, col) = Q[row * K_DIM + col];
        }
    }
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 2: Q loaded to SMEM\n", cta_idx);
    }
    
    // Load K: each CTA loads half with transpose
    // Global K layout: [K_ROWS, N] = [128, 256]
    // cta0: reads K[0:64, :] = [64, 256], stores as [256, 64] in SMEM (transposed)
    // cta1: reads K[64:128, :] = [64, 256], stores as [256, 64] in SMEM (transposed)
    const int k_row_offset = cta_idx * (K_ROWS / 2);
    constexpr int K_ELEMENTS = (K_ROWS / 2) * N;
    constexpr int K_ELEMENTS_PER_THREAD = (K_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;
    for (int i = 0; i < K_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * K_ELEMENTS_PER_THREAD + i;
        if (linear_idx < K_ELEMENTS) {
            // Read from global K[row, col], store transposed to sK[col, row]
            int row = linear_idx / N;
            int col = linear_idx % N;
            int global_row = k_row_offset + row;
            sK(col, row) = K[global_row * N + col];
        }
    }
    
    __syncthreads();
    cute::cluster_sync();
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 3: K loaded to SMEM\n", cta_idx);
    }
    
    // Allocate TMEM (warp 0 in each CTA, all threads in warp participate)
    if (warp_idx == 0) {
        cute::TMEM::Allocator2Sm().allocate(512, &smem.tmem_start_addr);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();  // Wait for TMEM allocation
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 4: TMEM allocated\n", cta_idx);
    }
    
    // Cluster sync before CTA0 accesses peer SMEM - all CTAs must participate
    __syncthreads();
    cute::cluster_sync();
    
    // Only CTA0 performs the two MMA operations and writes outputs
    bf16* peer_k_ptr = nullptr;
    auto sK_peer = cute::make_tensor(cute::make_smem_ptr<bf16>(nullptr), SmemLayoutKTiles<1>{});
    TiledMMA_O tiled_mma_O{};
    auto tP = cute::partition_fragment_C(tiled_mma_O, cute::Shape<cute::Int<M>, cute::Int<N>>{});
    
    if (cta_idx == 0) {
        // Create TiledMMA and fragments
        tiled_mma_O = TiledMMA_O{};
        
        // Create tP fragment for output: [M, N] = [64, 256]
        tP = cute::partition_fragment_C(tiled_mma_O, cute::Shape<cute::Int<M>, cute::Int<N>>{});
        // Set TMEM address: base + column offset
        tP.data().get() = tmem_cols::p;
        
        // Get peer CTA's smem.k address
        // Use get_peer_addr to get cta1's smem.k address
        peer_k_ptr = kerutils::get_peer_addr(smem.k);
        
        // Debug: Print addresses to verify peer address calculation
        // PEER_ADDR_MASK = 16777216 = 0x1000000 (bit 24)
        if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
            printf("[CTA %d] smem.k address: %p\n", cta_idx, smem.k);
            printf("[CTA %d] peer_k_ptr address: %p\n", cta_idx, peer_k_ptr);
            printf("[CTA %d] Address difference: %ld (expected: %d or %d)\n", 
                   cta_idx, (int64_t)peer_k_ptr - (int64_t)smem.k, 
                   16777216, -16777216);
        }
        
        // Create peer tensor - use make_smem_ptr to preserve the peer address
        // Note: make_smem_ptr should preserve the address value, but we need to ensure
        // the peer address bit (bit 24) is maintained
        sK_peer = cute::make_tensor(cute::make_smem_ptr(peer_k_ptr), SmemLayoutKTiles<1>{});

        if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
            printf("[CTA %d] finish 5: Peer K address obtained\n", cta_idx);
        }

        // Write Q, sK, sK_peer to global memory - only CTA0 executes
        // Write Q from SMEM to Q_out: [64, 128]
        constexpr int Q_ELEMENTS_OUT = M * K_DIM;
        constexpr int Q_ELEMENTS_PER_THREAD_OUT = (Q_ELEMENTS_OUT + NUM_THREADS - 1) / NUM_THREADS;
        for (int i = 0; i < Q_ELEMENTS_PER_THREAD_OUT; ++i) {
            int linear_idx = tid * Q_ELEMENTS_PER_THREAD_OUT + i;
            if (linear_idx < Q_ELEMENTS_OUT) {
                int row = linear_idx / K_DIM;
                int col = linear_idx % K_DIM;
                Q_out[row * K_DIM + col] = sQ(row, col);
            }
        }
        __syncthreads();
        
        if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
            printf("[CTA %d] finish 6: Q written to global memory\n", cta_idx);
        }
        
        // Write K from SMEM to K_out: merge sK (cta0) and sK_peer (cta1)
        // sK and sK_peer are stored as [256, 64] (transposed), need to write back as [128, 256]
        // sK (cta0): rows 0-63 of K, stored as [256, 64] -> write to K_out[0:64, :]
        // sK_peer (cta1): rows 64-127 of K, stored as [256, 64] -> write to K_out[64:128, :]
        constexpr int K_ELEMENTS_OUT = (K_ROWS / 2) * N;
        constexpr int K_ELEMENTS_PER_THREAD_OUT = (K_ELEMENTS_OUT + NUM_THREADS - 1) / NUM_THREADS;
        for (int i = 0; i < K_ELEMENTS_PER_THREAD_OUT; ++i) {
            int linear_idx = tid * K_ELEMENTS_PER_THREAD_OUT + i;
            if (linear_idx < K_ELEMENTS_OUT) {
                int row = linear_idx / N;
                int col = linear_idx % N;
                // Write cta0's K (first 64 rows)
                int global_row = row;
                K_out[global_row * N + col] = sK(col, row);
            }
        }
        __syncthreads();
        
        if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
            printf("[CTA %d] finish 7: K (cta0) written to global memory\n", cta_idx);
        }
        
        // Ensure peer SMEM is ready before accessing
        __threadfence();
        __syncthreads();
    }
    
    // Cluster sync before accessing peer SMEM - all CTAs must participate
    cute::cluster_sync();
    
    // Write cta1's K (last 64 rows) - only CTA0 executes
    if (cta_idx == 0) {
        if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
            printf("[CTA %d] finish 7.5: About to access peer SMEM\n", cta_idx);
        }
        
        // Test access to peer SMEM first element to ensure it's accessible
        if (tid == 0) {
            volatile bf16 test_val = sK_peer(0, 0);
            (void)test_val;  // Suppress unused variable warning
        }
        __syncthreads();
        
        // Write cta1's K (last 64 rows)
        constexpr int K_ELEMENTS_OUT = (K_ROWS / 2) * N;
        constexpr int K_ELEMENTS_PER_THREAD_OUT = (K_ELEMENTS_OUT + NUM_THREADS - 1) / NUM_THREADS;
        for (int i = 0; i < K_ELEMENTS_PER_THREAD_OUT; ++i) {
            int linear_idx = tid * K_ELEMENTS_PER_THREAD_OUT + i;
            if (linear_idx < K_ELEMENTS_OUT) {
                int row = linear_idx / N;
                int col = linear_idx % N;
                // Write cta1's K (last 64 rows)
                int global_row = (K_ROWS / 2) + row;
                K_out[global_row * N + col] = sK_peer(col, row);
            }
        }
        __syncthreads();
        
        if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
            printf("[CTA %d] finish 8: K (cta1) written to global memory\n", cta_idx);
        }
    }
        
        // Cluster sync after writing K - all CTAs must participate
        __syncthreads();
        cute::cluster_sync();
        
        // Split Q into two halves and execute MMA - only CTA0 executes
        if (cta_idx == 0) {
            // Split Q into two halves: [64, 64] each using flat_divide
            // First half: Q[:, 0:64]
            // Second half: Q[:, 64:128]
            auto sQ_first_half = cute::flat_divide(
                sQ,
                cute::Tile<cute::Int<M>, cute::Int<K_DIM/2>>{}
            )(cute::_, cute::_, cute::_0{}, cute::_0{});

            // Write sQ_first_half to global memory: [M, K_DIM/2] = [64, 64]
            constexpr int Q_FIRST_HALF_ELEMENTS = M * (K_DIM / 2);
            constexpr int Q_FIRST_HALF_ELEMENTS_PER_THREAD = (Q_FIRST_HALF_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;
            for (int i = 0; i < Q_FIRST_HALF_ELEMENTS_PER_THREAD; ++i) {
                int linear_idx = tid * Q_FIRST_HALF_ELEMENTS_PER_THREAD + i;
                if (linear_idx < Q_FIRST_HALF_ELEMENTS) {
                    int row = linear_idx / (K_DIM / 2);
                    int col = linear_idx % (K_DIM / 2);
                    Q_first_half_out[row * (K_DIM / 2) + col] = sQ_first_half(row, col);
                }
            }
            __syncthreads();
            
            // Execute first matrix multiplication: P = Q_first_half @ K_local^T
            // clear_accum = true (initialize accumulator)
            if (warp_idx == 0 && cute::elect_one_sync()) {
                ku::utcmma_ss(tiled_mma_O, sQ_first_half, sK, tP, true);
            }
            ku::tcgen05_after_thread_sync();
        }
        
        // Synchronize after first MMA - all CTAs must participate
        __syncthreads();
        cute::cluster_sync();
        
        if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
            printf("[CTA %d] finish 9: First MMA completed\n", cta_idx);
        }
        
        // Execute second matrix multiplication: P += Q_second_half @ K_peer^T
        // clear_accum = false (accumulate to existing result) - only CTA0 executes
        if (cta_idx == 0) {
            // Get sQ_second_half using flat_divide
            auto sQ_second_half = cute::flat_divide(
                sQ,
                cute::Tile<cute::Int<M>, cute::Int<K_DIM/2>>{}
            )(cute::_, cute::_, cute::_0{}, cute::_1{});

            // Write sQ_second_half to global memory: [M, K_DIM/2] = [64, 64]
            constexpr int Q_SECOND_HALF_ELEMENTS = M * (K_DIM / 2);
            constexpr int Q_SECOND_HALF_ELEMENTS_PER_THREAD = (Q_SECOND_HALF_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;
            for (int i = 0; i < Q_SECOND_HALF_ELEMENTS_PER_THREAD; ++i) {
                int linear_idx = tid * Q_SECOND_HALF_ELEMENTS_PER_THREAD + i;
                if (linear_idx < Q_SECOND_HALF_ELEMENTS) {
                    int row = linear_idx / (K_DIM / 2);
                    int col = linear_idx % (K_DIM / 2);
                    Q_second_half_out[row * (K_DIM / 2) + col] = sQ_second_half(row, col);
                }
            }
            __syncthreads();
            
            if (warp_idx == 0 && cute::elect_one_sync()) {
                ku::utcmma_ss(tiled_mma_O, sQ_second_half, sK_peer, tP, false);
            }
            ku::tcgen05_after_thread_sync();
        }
        
        // Synchronize after second MMA - all CTAs must participate
        __syncthreads();
        cute::cluster_sync();
        
        if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
            printf("[CTA %d] finish 10: Second MMA completed\n", cta_idx);
        }
        
        // Read results from TMEM and write to global memory - only CTA0 executes
        if (cta_idx == 0) {
            // P matrix shape: [M, N] = [64, 256]
            // TMEM layout: 128 lanes (rows) × 256 columns
            // Each logical row (256 floats) is stored as 128 float2 values in 128 TMEM columns
            
            constexpr int P_ROWS = M;  // 64 rows
            constexpr int P_COLS = N;  // 256 columns
            constexpr int FLOAT2_PER_ROW = P_COLS / 2 / 2;  // 64 float2 per row
            
            // Each thread reads one logical row of P
            int logical_row = tid % 64;  // Logical row index: 0-63
            int col_offset = tid / 64;
            
            // Construct TMEM address: base + (lane << 16) | column
            // lane = logical_row (TMEM lane index, 0-63)
            // column = tmem_cols::p (starting column offset, 256)
            uint32_t tmem_base = smem.tmem_start_addr;
            uint32_t tmem_lane = logical_row;  // TMEM lane = logical row
            uint32_t tmem_col = tmem_cols::p;  // Starting column offset
            
            // Full TMEM address: base + (lane << 16) | column
            uint32_t tmem_addr = tmem_base + (tmem_lane << 16) + tmem_col;
            
            // Load one logical row of P from TMEM (128 float2 = 256 floats)
            // This reads from TMEM lane = logical_row, columns = tmem_cols::p to tmem_cols::p + 127
            float2 p_row[FLOAT2_PER_ROW];
            ku::tmem_ld_32dp32bNx<P_COLS/2>(tmem_addr, p_row);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            
            // Write to global memory
            for (int col = 0; col < FLOAT2_PER_ROW; ++col) {
                P_out[logical_row * P_COLS + 128 * col_offset + col * 2] = p_row[col].x;
                P_out[logical_row * P_COLS + 128 * col_offset + col * 2 + 1] = p_row[col].y;
            }
            
            __syncthreads();
            
            if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
                printf("[CTA %d] finish 11: P read from TMEM and written to global memory\n", cta_idx);
            }
        }
        
    // Cluster sync after reading P - all CTAs must participate
    __syncthreads();
    cute::cluster_sync();
    
    // Free TMEM
    if (warp_idx == 0 && cute::elect_one_sync()) {
        cute::TMEM::Allocator2Sm().free(smem.tmem_start_addr, 512);
    }
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 12: TMEM freed, kernel completed\n", cta_idx);
    }

#endif
}

// C++ wrapper
void launch_test_utcmma_ss_peer(
    const bf16* Q,
    const bf16* K,
    float* P_out,
    bf16* Q_out,
    bf16* K_out,
    bf16* Q_first_half_out,
    bf16* Q_second_half_out,
    cudaStream_t stream
) {
    dim3 grid(2, 1, 1);  // 2 CTAs
    dim3 block(NUM_THREADS, 1, 1);
    
    // Set dynamic shared memory size
    cudaError_t attr_err = cudaFuncSetAttribute(
        test_utcmma_ss_peer_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        SMEM_SIZE
    );
    if (attr_err != cudaSuccess) {
        fprintf(stderr, "cudaFuncSetAttribute failed with error: %s\n", cudaGetErrorString(attr_err));
        return;
    }
    
    // Cluster configuration
    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    printf("finish 0: Launching kernel...\n");
    cudaError_t err = cudaLaunchKernelEx(&config, test_utcmma_ss_peer_kernel, Q, K, P_out, Q_out, K_out, Q_first_half_out, Q_second_half_out);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchKernelEx failed with error: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("finish 13: Kernel launch completed\n");
}

// Python binding: return P matrix (Q @ K^T), Q_out, K_out, Q_first_half_out, and Q_second_half_out
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> utcmma_ss_peer_forward(
    torch::Tensor Q, torch::Tensor K) {
    // Check inputs
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be bfloat16");
    TORCH_CHECK(Q.dim() == 2 && Q.size(0) == M && Q.size(1) == K_DIM, 
                "Q shape must be [64, 128]");
    TORCH_CHECK(K.dim() == 2 && K.size(0) == K_ROWS && K.size(1) == N,
                "K shape must be [128, 256]");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    
    // Create output tensors
    auto options_f32 = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(Q.device());
    auto options_bf16 = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(Q.device());
    
    torch::Tensor P_out = torch::empty({M, N}, options_f32);
    torch::Tensor Q_out = torch::empty({M, K_DIM}, options_bf16);
    torch::Tensor K_out = torch::empty({K_ROWS, N}, options_bf16);
    torch::Tensor Q_first_half_out = torch::empty({M, K_DIM / 2}, options_bf16);
    torch::Tensor Q_second_half_out = torch::empty({M, K_DIM / 2}, options_bf16);
    
    // Get data pointers
    const bf16* Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>());
    const bf16* K_ptr = reinterpret_cast<const bf16*>(K.data_ptr<at::BFloat16>());
    float* P_out_ptr = P_out.data_ptr<float>();
    bf16* Q_out_ptr = reinterpret_cast<bf16*>(Q_out.data_ptr<at::BFloat16>());
    bf16* K_out_ptr = reinterpret_cast<bf16*>(K_out.data_ptr<at::BFloat16>());
    bf16* Q_first_half_out_ptr = reinterpret_cast<bf16*>(Q_first_half_out.data_ptr<at::BFloat16>());
    bf16* Q_second_half_out_ptr = reinterpret_cast<bf16*>(Q_second_half_out.data_ptr<at::BFloat16>());
    
    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call kernel
    printf("finish -1: Starting kernel launch from Python binding\n");
    launch_test_utcmma_ss_peer(Q_ptr, K_ptr, P_out_ptr, Q_out_ptr, K_out_ptr, Q_first_half_out_ptr, Q_second_half_out_ptr, stream);
    
    // Synchronize and wait for kernel completion
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    printf("finish 14: Kernel execution completed and synchronized\n");
    
    return std::make_tuple(P_out, Q_out, K_out, Q_first_half_out, Q_second_half_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("utcmma_ss_peer", &utcmma_ss_peer_forward, 
          "Test utcmma_ss matrix multiplication with peer SMEM access (CUDA). Returns (P_out, Q_out, K_out, Q_first_half_out, Q_second_half_out)");
}
