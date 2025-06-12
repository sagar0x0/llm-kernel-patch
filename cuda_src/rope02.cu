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
    int batch_idx = blockIdx.x;         // Batch index
    int head_idx_q = blockIdx.y;       // Q Head index
    int seq_idx = blockIdx.z;         // Sequence index
    int pair_idx = threadIdx.x;     // Index for the pair within head_dim (0 to head_dim/2 - 1)

    // Boundary checks: ensure thread is within valid work dimensions
    if (batch_idx >= batch_size || head_idx_q >= num_attention_heads_q || seq_idx >= seq_len || pair_idx >= (head_dim / 2)) {
        return;
    }

    // Calculate base offset for Q tensor for the current batch, head, and sequence position
    // Q shape: [batch_size, num_attention_heads_q, seq_len, head_dim]
    int q_base_offset = (batch_idx * num_attention_heads_q * seq_len * head_dim) + 
                        (head_idx_q * seq_len * head_dim ) + (seq_idx * head_dim) ;
    
    
    // access q_val pair (0 , head_dim/2)
    float* q_val1 = q + q_base_offset + pair_idx ;
    float* q_val2 = q + q_base_offset + pair_idx + head_dim/2 ;

    // access cos and sin first half value 
    int cos_sin_offset = (batch_idx * seq_len * head_dim) + (seq_idx * head_dim) ;
    
    float* cos_val = cos + cos_sin_offset + pair_idx ;
    float* sin_val = sin + cos_sin_offset + pair_idx ;

    // Access the current elements and their corresponding rotated counterparts
    float q_rot1 = - q_val2[0] ;
    float q_rot2 =   q_val1[0] ;

    // now perform resultant cos sin mul
    q_val1[0] = q_val1[0] * cos_val[0] + q_rot1 * sin_val[0] ;
    q_val2[0] = q_val2[0] * cos_val[0] + q_rot2 * sin_val[0] ;


    // now calculate k [batch_size, num_attention_heads_k, seq_len, head_dim]
    int num_kv_groups = num_attention_heads_q / num_attention_heads_k ;

    int k_base_offset;
    
    if (head_idx_q % num_kv_groups == 0) {
        int head_idx_k = head_idx_q / num_kv_groups; 
        k_base_offset = (batch_idx * num_attention_heads_k * seq_len * head_dim) + 
                    (head_idx_k * seq_len * head_dim ) + (seq_idx * head_dim) ; 
    } 
    else {
        k_base_offset = -1 ;
    }

    if(k_base_offset != -1){
        // access k_val pair (0 , head_dim/2)
        float* k_val1 = k + k_base_offset + pair_idx ;
        float* k_val2 = k + k_base_offset + pair_idx + head_dim/2 ;

        float k_rot1 = - k_val2[0];
        float k_rot2 =   k_val1[0];

        // now perform resultant cos sin mul
        k_val1[0] = k_val1[0] * cos_val[0] + k_rot1 * sin_val[0] ;
        k_val2[0] = k_val2[0] * cos_val[0] + k_rot2 * sin_val[0] ;
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
    auto* q = reinterpret_cast<float*>(q_ptr)   ;    // q [batch_size, num_attention_heads_q, seq_len, head_dim]
    auto* k = reinterpret_cast<float*>(k_ptr)    ;   // k [batch_size, num_attention_heads_k, seq_len, head_dim]
    auto* cos = reinterpret_cast<float*>(cos_ptr) ;  // cos [batch_size, seq_len, head_dim]
    auto* sin = reinterpret_cast<float*>(sin_ptr)  ; // sin [batch_size, seq_len, head_dim]

    int block_threads = head_dim/2 ;

    dim3 block(block_threads) ;  // num_attention_heads
    dim3 grid(batch_size , num_attention_heads_q , seq_len) ;
    
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

    PYTHON: q shape: torch.Size([1, 24, 8, 128])
    PYTHON: k shape: torch.Size([1, 8, 8, 128])
    PYTHON: cos shape: torch.Size([1, 8, 128])
    PYTHON: sin shape: torch.Size([1, 8, 128])
*/