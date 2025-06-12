#include <torch/extension.h>
#include <pybind11/pybind11.h>
// #include <iostream>

// optimize compilation for  L4 , P100 , T4
//os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0 7.5 8.9"

__global__ void rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ cos,
    float* __restrict__ sin ,
    int batch_size,
    int num_attention_heads_q ,
    int num_attention_heads_k ,
    int seq_len, 
    int head_dim
) {

    // Determine the unique indices for this thread's work item
    int attenhead_q_idx = blockIdx.x;         // attention head idx q
    int seq_idx = blockIdx.y;         // Sequence index
    int batch_idx = blockIdx.z ;           // Batch index
    int pair_idx = threadIdx.x;     // Index for the pair within head_dim (0 to head_dim/2 - 1)

    // Boundary checks: ensure thread is within valid work dimensions
    /*
    if (batch_idx >= batch_size || seq_idx >= seq_len || pair_idx >= (head_dim / 2)) {
        return;
    }
    */


    // shared mem size know at compile time 
    //__shared__ float cos_s[64] ;  since eack of thread block will read the value once : using SMEM is redundant
    //__shared__ float sin_s[64] ;  same as above 

    // each of q , k is used two times each : if loaded per thread it will have strided acess :: so use SMEM 
    __shared__ float q_s[128] ;
    __shared__ float k_s[128] ;


    // load the data from gmem to smem 

    // reinterpret_cast allows loading two floats at once with a single instruction
    // Q shape: [batch_size, seq_len, num_attention_heads_q, head_dim]
    int q_base_offset = (batch_idx * seq_len * num_attention_heads_q * head_dim) + 
                        (seq_idx * num_attention_heads_q * head_dim) + (attenhead_q_idx * head_dim) ;

    // mem coalesced access
    q_s[pair_idx ] = q[q_base_offset + pair_idx] ;
    q_s[head_dim/2 + pair_idx ]  = q[q_base_offset + head_dim/2 + pair_idx] ;

    int k_base_offset = -1 ;
    // calc for k   in first 8 attenhead_q_idx 
    // changed logic to eleminate % and / which is very expensive 
    if(attenhead_q_idx < num_attention_heads_k )
    {
        k_base_offset = (batch_idx * seq_len * num_attention_heads_k * head_dim) + 
                            (seq_idx * num_attention_heads_k * head_dim) + (attenhead_q_idx * head_dim)  ; 

        k_s[pair_idx ] = k[k_base_offset + pair_idx] ;
        k_s[head_dim/2 + pair_idx]  = k[k_base_offset + head_dim/2 + pair_idx ] ;

    }

    // sync all threads in block 
    __syncthreads();

    // access cos and sin first half value 
    int cos_sin_offset = (batch_idx * seq_len * head_dim) + (seq_idx * head_dim) + pair_idx ; 

    // load from GMEM 
    float cos_val =  cos[cos_sin_offset] ;
    float sin_val =  sin[cos_sin_offset] ;

    // load from SMEM to local reg
    float q_val_1 = q_s[pair_idx] ;
    float q_val_2 = q_s[head_dim/2 + pair_idx] ;

    // store the q pairs to GMEM
    q[q_base_offset + pair_idx] = q_val_1 * cos_val - q_val_2 * sin_val ;
    q[q_base_offset + head_dim/2 + pair_idx] = q_val_2 * cos_val + q_val_1 * sin_val ;




    //  K [batch_size, seq_len, num_attention_heads_k, head_dim]
    // calc for k   in first 8 attenhead_q_idx 
    // changed logic to eleminate % and / which is very expensive 
    if(attenhead_q_idx < num_attention_heads_k )
    {
        // for attenhead_q_idx < num_attention_heads_k :: attenhead_q_idx = attenhead_k_idx
        // already calcualted k_base_offset

        // load from SMEM to local reg
        float k_val_1 = k_s[pair_idx] ;
        float k_val_2 = k_s[head_dim/2 + pair_idx] ;


        // load the k pairs
        k[k_base_offset + pair_idx] = k_val_1 * cos_val - k_val_2 * sin_val ;
        k[k_base_offset + head_dim/2 + pair_idx] = k_val_2 * cos_val + k_val_1 * sin_val ;
    }

}

void run(
    uintptr_t q_ptr,
    uintptr_t k_ptr,
    uintptr_t cos_ptr,
    uintptr_t sin_ptr,
    int batch_size,
    int num_attention_heads_q ,
    int num_attention_heads_k ,
    int seq_len, 
    int head_dim
) {
    auto* q = reinterpret_cast<float*>(q_ptr)   ;    // q [batch_size, seq_len, num_attention_heads_q, head_dim]
    auto* k = reinterpret_cast<float*>(k_ptr)    ;   // k [batch_size, seq_len, num_attention_heads_k, head_dim]
    auto* cos = reinterpret_cast<float*>(cos_ptr) ;  // cos [batch_size, seq_len, head_dim]
    auto* sin = reinterpret_cast<float*>(sin_ptr)  ; // sin [batch_size, seq_len, head_dim]

    int block_threads = head_dim/2 ; // head_dim = 128 : block_threads = 64

    dim3 block(block_threads) ;  // num_attention_heads
    dim3 grid(num_attention_heads_q , seq_len, batch_size) ;    // x , y, z
    
    // shared cos and sin for all num_attention_heads 
    rope_kernel<<< grid, block >>>(
        q, k, cos, sin, batch_size, 
        num_attention_heads_q ,num_attention_heads_k ,
        seq_len, head_dim
    );

}



PYBIND11_MODULE(rope_cuda_kernel, m) {
  m.def("rope_cuda_kernel", &run, "Cuda Rope kernel");
}

/*

    PYTHON: q shape: torch.Size([batch_size, seq_len, 24, 128])
    PYTHON: k shape: torch.Size([batch_size, seq_len, 8, 128])
    PYTHON: cos shape: torch.Size([batch_size, seq_len, 128]])
    PYTHON: sin shape: torch.Size([batch_size, seq_len, 128])
*/