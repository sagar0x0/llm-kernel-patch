#include <torch/extension.h>
#include <pybind11/pybind11.h>
// #include <iostream>

//os.environ["TORCH_CUDA_ARCH_LIST"] = 

__global__ void rope_kernel(
    float* q,
    float* k,
    float* cos,
    float* sin ,
    int batch_size,
    int num_attention_heads_q ,
    int num_attention_heads_k ,
    int seq_len, 
    int head_dim
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x; 

    int total_threads = batch_size * num_attention_heads_q;
    if (tid >= total_threads) return;

    int batch_idx = tid / num_attention_heads_q ;  // num_atten_head = 24 
    int q_atten_head_idx = tid % num_attention_heads_q ;

    float* cos_tid = cos + batch_idx * (seq_len * head_dim) ;
    float* sin_tid = sin + batch_idx * (seq_len * head_dim) ;

    int num_kv_groups = num_attention_heads_q / num_attention_heads_k ;
    
    float* k_tid;
    
    if (q_atten_head_idx % num_kv_groups == 0) {
        int kv_head_idx_in_batch = q_atten_head_idx / num_kv_groups; 
        k_tid = k + (batch_idx * num_attention_heads_k + kv_head_idx_in_batch) * (seq_len * head_dim);
    } else {
        k_tid = nullptr;
    }

    float* q_tid = q + tid * (seq_len * head_dim) ;


    // q [seq_len, head_dim] || cos [seq_len, head_dim]

    // rotate_q


    for(int i = 0 ; i < seq_len ; i++){

        for(int j = 0 ; j < head_dim/2 ; j++){

            // Get the current cosine and sine values for this pair 
            // rope is applied in pair { [j] , [head_dim/2 +j] }   
            // only head_dim/2 of cos and sin is used    ||  other half of head_dim (head_dim/2 :) is 
            // either garbage or duplicate  
            float cos_val = cos_tid[i * head_dim + j];
            float sin_val = sin_tid[i * head_dim + j];

            // Access the current elements and their corresponding rotated counterparts
            float q_j = q_tid[i * head_dim + j];
            float q_j_half = q_tid[i * head_dim + head_dim/2 + j];

            // elemnetiwise q*cos + rot_q * sin
            // apply rotation to q
            q_tid[i * head_dim + j] = q_j * cos_val - q_j_half * sin_val;
            q_tid[i * head_dim + head_dim/2 + j] = q_j_half * cos_val + q_j * sin_val;

            if(k_tid != nullptr){
                float k_j = k_tid[i * head_dim + j];
                float k_j_half = k_tid[i * head_dim + head_dim/2 + j];

                k_tid[i * head_dim + j] = k_j * cos_val - k_j_half * sin_val;
                k_tid[i * head_dim + head_dim/2 + j] = k_j_half * cos_val + k_j * sin_val;
            }
        }
    
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
    auto* q = reinterpret_cast<float*>(q_ptr)   ;    // q [batch_size, num_attention_heads, seq_len, head_dim]
    auto* k = reinterpret_cast<float*>(k_ptr)    ;   // k [batch_size, num_attention_heads, seq_len, head_dim]
    auto* cos = reinterpret_cast<float*>(cos_ptr) ;  // cos [batch_size, seq_len, head_dim]
    auto* sin = reinterpret_cast<float*>(sin_ptr)  ; // sin [batch_size, seq_len, head_dim]


    dim3 block(num_attention_heads_q) ;  // num_attention_heads
    dim3 grid(batch_size) ;
    
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