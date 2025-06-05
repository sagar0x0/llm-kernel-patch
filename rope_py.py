import os
import torch
from torch.utils.cpp_extension import load_inline

os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0 7.5 8.9"

# optimize compilation for  L4 , P100 , T4
# os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0 7.5 8.9"

kernel_cpp = r"""
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
    int seq_idx = blockIdx.y;         // Sequence index
    int pair_idx = threadIdx.x;     // Index for the pair within head_dim (0 to head_dim/2 - 1)

    // Boundary checks: ensure thread is within valid work dimensions
    if (batch_idx >= batch_size || seq_idx >= seq_len || pair_idx >= (head_dim / 2)) {
        return;
    }

    // Calculate base offset for Q tensor for the current batch, head, and sequence position
    // Q shape: [batch_size, seq_len, head_dim , num_attention_heads_q]
    int q_base_offset = (batch_idx * num_attention_heads_q * seq_len * head_dim) + 
                        (seq_idx * head_dim * num_attention_heads_q) ;
    
    
    // access q_val pair (0 , head_dim/2)
    float* q_val1 = q + q_base_offset + pair_idx * num_attention_heads_q ;
    float* q_val2 = q + q_base_offset + (pair_idx + head_dim/2) * num_attention_heads_q  ;


    // access cos and sin first half value 
    int cos_sin_offset = (batch_idx * seq_len * head_dim) + (seq_idx * head_dim) ;
    
    float* cos_val_ptr = cos + cos_sin_offset + pair_idx ;
    float* sin_val_ptr = sin + cos_sin_offset + pair_idx ;

    float cos_val =  cos_val_ptr[0] ;
    float sin_val =  sin_val_ptr[0] ;

    // vectorized load reinterpret cast 
    float4* q_val1_vecld = reinterpret_cast<float4*>(q_val1);
    float4* q_val2_vecld = reinterpret_cast<float4*>(q_val2);

    for(int i = 0 ; i < num_attention_heads_q / 4 ; i++ )
    {

        // now perform resultant cos sin mul  
        // mem coelasced gemem access vectorized load
        float4 q_val1_tmp = q_val1_vecld[i] ; 
        float4 q_val2_tmp = q_val2_vecld[i] ;


        // ops and store in float4 q_val1_tmp
        q_val1_tmp.x =  q_val1_tmp.x * cos_val + (- q_val2_tmp.x) * sin_val ;
        q_val1_tmp.y =  q_val1_tmp.y * cos_val + (- q_val2_tmp.y) * sin_val ;
        q_val1_tmp.z =  q_val1_tmp.z * cos_val + (- q_val2_tmp.z) * sin_val ;
        q_val1_tmp.w =  q_val1_tmp.w * cos_val + (- q_val2_tmp.w) * sin_val ;

        q_val2_tmp.x =  q_val2_tmp.x * cos_val + q_val1_tmp.x * sin_val ;
        q_val2_tmp.y =  q_val2_tmp.y * cos_val + q_val1_tmp.y * sin_val ;
        q_val2_tmp.z =  q_val2_tmp.z * cos_val + q_val1_tmp.z * sin_val ;
        q_val2_tmp.w =  q_val2_tmp.w * cos_val + q_val1_tmp.w * sin_val ;

        // store back to gmem in vec store 
        q_val1_vecld[i] = q_val1_tmp ;
        q_val2_vecld[i] = q_val2_tmp ;
    }

    // calc for k 
    int k_base_offset = (batch_idx * seq_len * head_dim * num_attention_heads_k ) + 
                        (seq_idx * head_dim * num_attention_heads_k)  ; 

    float* k_val1 = k + k_base_offset + pair_idx * num_attention_heads_k ;
    float* k_val2 = k + k_base_offset + (head_dim/2 + pair_idx) * num_attention_heads_k ;
    
    // vectorized load reinterpret cast 
    float4* k_val1_vecld = reinterpret_cast<float4*>(k_val1);
    float4* k_val2_vecld = reinterpret_cast<float4*>(k_val2);

    for(int i = 0 ; i < num_attention_heads_k / 4 ; i++ )
    {
        // now perform resultant cos sin mul  
        // mem coelasced gemem access vectorized load
        float4 k_val1_tmp = k_val1_vecld[i] ; 
        float4 k_val2_tmp = k_val2_vecld[i] ;


        // ops and store in float4 q_val1_tmp
        k_val1_tmp.x =  k_val1_tmp.x * cos_val + (- k_val2_tmp.x) * sin_val ;
        k_val1_tmp.y =  k_val1_tmp.y * cos_val + (- k_val2_tmp.y) * sin_val ;
        k_val1_tmp.z =  k_val1_tmp.z * cos_val + (- k_val2_tmp.z) * sin_val ;
        k_val1_tmp.w =  k_val1_tmp.w * cos_val + (- k_val2_tmp.w) * sin_val ;

        k_val2_tmp.x =  k_val2_tmp.x * cos_val + k_val1_tmp.x * sin_val ;
        k_val2_tmp.y =  k_val2_tmp.y * cos_val + k_val1_tmp.y * sin_val ;
        k_val2_tmp.z =  k_val2_tmp.z * cos_val + k_val1_tmp.z * sin_val ;
        k_val2_tmp.w =  k_val2_tmp.w * cos_val + k_val1_tmp.w * sin_val ;



        // store back to gmem in vec store 
        k_val1_vecld[i] = k_val1_tmp ;
        k_val2_vecld[i] = k_val2_tmp ;
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
    auto* k = reinterpret_cast<float*>(k_ptr)    ;   // k [batch_size, seq_len, num_attention_heads_q, head_dim]
    auto* cos = reinterpret_cast<float*>(cos_ptr) ;  // cos [batch_size, seq_len, head_dim]
    auto* sin = reinterpret_cast<float*>(sin_ptr)  ; // sin [batch_size, seq_len, head_dim]

    int block_threads = head_dim/2 ;

    dim3 block(block_threads) ;  // num_attention_heads
    dim3 grid(batch_size , seq_len) ;
    
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

"""

rope_module = load_inline(
    name="rope_cuda_kernel",
    cpp_sources="",
    cuda_sources=kernel_cpp ,
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-std=c++20"]
)

def apply_custom_rope(q, k, cos, sin):
    seq_len = q.shape[2]
    batch_size = q.shape[0]
    num_attention_heads_q = q.shape[1]
    head_dim = q.shape[-1]

    num_attention_heads_k = k.shape[1]

    # print(f"cos shape: {cos.shape} \n")
    # PYTHON: q shape: torch.Size([1, 24, 8, 128])
    # PYTHON: k shape: torch.Size([1, 8, 8, 128])
    # PYTHON: cos shape: torch.Size([1, 8, 128])
    # PYTHON: sin shape: torch.Size([1, 8, 128])
    
    # Q/K are NOT contiguous
    # Cos/Sin are NOT contiguous
    # assert q.dtype == k.dtype == torch.float32, "Q/K must be float32"
    # assert cos.dtype == sin.dtype == torch.float32, "Cos/Sin must be float32"
    # dtype are NOT float32'''
    
    
    # print(f"q datatype : {q.dtype} \n")
    # print(f"cos datatype : {cos.dtype} \n")    

    # q [batch_size, num_attention_heads_q, seq_len, head_dim]
    # k [batch_size, num_attention_heads_k, seq_len, head_dim]
    # cos [batch_size, seq_len, head_dim]
    # sin [batch_size, seq_len, head_dim]
    
    # rope02 cpp config chnage  [batch_size, num_attention_heads_q, seq_len, head_dim] to [batch_size, seq_len, head_dim, num_attention_heads_q]
    q = q.permute(0, 2, 3, 1)
    k = k.permute(0, 2, 3, 1)

    # Convert inputs to float32 before kernel call 
    q_f32 = q.to(torch.float32).contiguous() 
    k_f32 = k.to(torch.float32).contiguous() 
    cos_f32 = cos.to(torch.float32).contiguous() 
    sin_f32 = sin.to(torch.float32).contiguous() 

    rope_module.rope_cuda_kernel(
        q_f32.data_ptr(),
        k_f32.data_ptr(),
        cos_f32.data_ptr(), 
        sin_f32.data_ptr(),
        batch_size ,
        num_attention_heads_q ,
        num_attention_heads_k ,
        seq_len,
        head_dim
    )

    q_f32 = q_f32.permute(0, 3, 1, 2)
    k_f32 = k_f32.permute(0, 3, 1, 2)

    # Convert back to input dtype (bfloat16 or other)
    return q_f32.to(q.dtype), k_f32.to(k.dtype)