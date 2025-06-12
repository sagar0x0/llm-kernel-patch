# llm-kernel-patch  
Patch the transformer library LLM part with a specific LLM CUDA kernel  

I have implemented a custom RoPE kernel for LLMs.  

I have tried implementing a few RoPE kernel versions:  
Located in the `cuda_src` folder  

You can also run the LLM model by monkey patching my kernel code into the Hugging Face `transformers` library:  
[custom-RoPE-kernel-patch on Kaggle](https://www.kaggle.com/code/sagar4u/custom-rope-kernel-patch)  

---

## LLM Kernel Patch  
### Custom CUDA RoPE Kernel  

This repository provides a custom CUDA implementation of the RoPE (Rotary Positional Embedding) kernel for use in LLMs (Large Language Models). It enables you to patch the Hugging Face `transformers` library with optimized CUDA kernels to accelerate RoPE computations.

---

## Features  
- Multiple RoPE kernel versions implemented in CUDA (see `cuda_src/`)  
- Easy monkey patching for integration with Hugging Face models  

---

## Structure  

- `cuda_src/`: Contains various versions of custom RoPE kernels written in CUDA  
- `custom_arch.py`: Patched `LlamaAttentionWithCustomRope` class in `transformers`  
- `rope_py.py`: Loads pybinded CUDA code into Python  

