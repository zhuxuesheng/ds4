# DeepSeek V4 Flash - Intel Xeon Dual-Socket Optimization Plan

## 1. Background & Motivation
The `ds4` inference engine currently targets macOS (Metal) and Linux (CUDA). The CPU path is currently an unoptimized reference implementation. However, for a dual-socket server like the Intel Xeon Gold 5318Y (Ice Lake-SP) with 48 physical cores, 512GB RAM, and AVX-512 VNNI support, the CPU is capable of acting as a high-performance inference engine for DeepSeek V4 Flash.

The initial plan considered a conservative INT8 (W4A8) approach. However, DeepSeek V4 heavily relies on MLA and MoE architectures, which produce extreme activation outliers. A strict INT8 activation path risks severe precision degradation (model "hallucination"). 

To achieve **1000+ Token/s Prefill** without sacrificing the model's reasoning capabilities, we are pivoting to a **High-Performance VNNI Static Graph Architecture** utilizing a **W4A16 / W2A16 Integer** path.

## 2. Inference Performance Analysis (Bottlenecks & Theoretical Limits)

### Hardware Capabilities (Dual Intel Xeon Gold 5318Y)
*   **Compute (VPDPWSSD):** The Ice Lake architecture supports `_mm512_dpwssd_epi32` (INT16 VNNI). This instruction allows processing 16-bit integer activations against 16-bit integer weights, yielding a theoretical peak of **~32 TFLOPS** equivalent.
*   **Memory Bandwidth:** Dual-socket 8-channel DDR4-2933 provides an effective sustained bandwidth of **~260 - 300 GB/s**.
*   **NUMA Penalty:** Cross-socket memory access over the UPI bus incurs massive latency and bandwidth penalties.

### DeepSeek V4 Flash GGUF Format Compatibility
Analysis of the `antirez/deepseek-v4-gguf` repository reveals two highly compatible formats for our strategy:
1.  **Q4KExperts (165GB):** 4-bit routed experts.
2.  **IQ2XXS (86.7GB):** 2-bit routed experts.

Both formats keep highly sensitive control layers (MLA compressor, router) in F16 and attention projections in Q8_0.

### Estimated Limits (Prefill)
*   **W4A16 (Q4_K):** Demands ~16 GFLOPs/token. With 32 TFLOPS peak, achievable target is **~500 - 800 tokens/s**.
*   **W2A16 (IQ2_XXS):** Halves memory bandwidth pressure. Target prefill speed is **~1000 - 1500 tokens/s**.

## 3. Scheme Design: High-Performance VNNI Graph

We will replace the dynamic memory allocation (`ds4_cpu_decode_scratch`) with a static, NUMA-aware execution graph (`ds4_cpu_graph`).

### Phase 1: Static Graph & NUMA Architecture
- **`ds4_cpu_graph`:** Pre-allocate all intermediate activation tensors for the maximum batch size (e.g., 1024 or 2048) as `int16_t` buffers. Zero `malloc`/`free` calls during the inference loop.
- **NUMA-Aware Expert Parallelism:** For the 256 MoE experts, the graph will statically partition the workload across the 96 threads. Threads bound to Socket 0 will strictly process experts resident in Socket 0's local memory, completely eliminating cross-socket UPI traffic during the heavy MoE routing phase.

### Phase 2: VPDPWSSD (INT16 VNNI) Compute Kernels
Instead of floating-point math, the core data path will be pure integer:
- **Activations (A16):** Intermediate states will be held as `int16_t`.
- **Weights (W4/W2):** Left in `Q4_K` or `IQ2_XXS` format in memory.
- **Kernel (`matmul_w4a16_vnni` / `matmul_w2a16_vnni`):**
  1. Use AVX-512 shuffle/shift instructions to dequantize 4-bit/2-bit blocks to 16-bit integers on the fly directly in registers.
  2. Execute `_mm512_dpwssd_epi32` to perform the dot product and accumulate into 32-bit registers.
  3. Rescale the 32-bit accumulators back to 16-bit for the next layer.

### Phase 3: Hybrid Precision Routing
To support the existing Hugging Face GGUF layout seamlessly:
- **Routed Experts:** Use the new W4A16 / W2A16 VNNI kernels.
- **Attention Projections (`Q8_0`):** Temporarily scale the A16 activations to A8 and use `VPDPBUSD` (INT8 VNNI) for double the throughput.
- **Control Layers (`F16`):** Dequantize A16 to FP32 and use standard AVX-512 FMA to preserve strict precision for MLA and routing logic.

## 4. Development Roadmap

To ensure the original engine's reference implementation remains completely untouched while maximizing server performance, we will implement this via a dedicated `DS4_BACKEND_XEON`. Development will proceed in the following test-driven phases:

### Step 1: Backend Isolation & Early Performance Verification (The "Micro-Benchmark")
**Goal:** Verify the theoretical INT16 VNNI performance bottleneck before rewriting the entire engine.
1. **Define the Backend:** Add `DS4_BACKEND_XEON` to `ds4_backend` enum. Create an isolation layer (e.g., `ds4_xeon.h` / `ds4_xeon.c`). 
2. **Build VNNI Micro-kernel:** Write the pure AVX-512 `_mm512_dpwssd_epi32` inline assembly/intrinsics for a single dense matrix multiplication (`matmul_w4a16_vnni` and `matmul_w2a16_vnni`).
3. **Micro-Benchmark Test:** Create a standalone test `tests/ds4_xeon_matmul_bench.c`. 
   - *Test criteria:* Feed it dummy Q4_K/IQ2_XXS data. Assert that it achieves >20 TFLOPS on the dual-socket machine. This proves our core assumption.

### Step 2: Unpacking & Quantization Scaling Logic
**Goal:** Implement the mathematical glue to convert the model's 4-bit/2-bit blocks to 16-bit, and scale the 32-bit accumulators back to 16-bit safely.
1. **Unpack Kernels:** Write AVX-512 shuffle/shift logic to expand `Q4_K` and `IQ2_XXS` blocks to `int16_t` registers.
2. **Scaling Logic:** Implement the dynamic scaling calculation (`Scale_Weight` * `Scale_Activation`) to prevent INT32 overflow and preserve the dynamic range.
3. **Validation Test:** Update `tests/ds4_test.c` or create `tests/ds4_xeon_math_test.c`.
   - *Test criteria:* Compare the output of the new VNNI dot product against the original `ds4.c` scalar `dot_q8_0_row` or `dot_f16_row` (converted to equivalent types). Assert that the mathematical output is within an acceptable epsilon, ensuring no severe precision loss.

### Step 3: The Static Graph & NUMA Topology (`ds4_xeon_graph`)
**Goal:** Pre-allocate all memory and map MoE experts to specific CPU sockets.
1. **Graph Definition:** Define the `ds4_xeon_graph` struct containing pre-allocated `int16_t` activation buffers for the maximum batch size.
2. **NUMA Binding:** Modify thread initialization for the `-xeon` backend to use `pthread_setaffinity_np`. Pin threads logically (e.g., Threads 0-47 to Socket 0, Threads 48-95 to Socket 1). 
3. **Expert Partitioning:** Write the scheduling logic that guarantees an expert residing in Socket 0's memory is only ever processed by Socket 0's threads.

### Step 4: Engine Integration & End-to-End Validation
**Goal:** Wire the Xeon backend into the main `ds4_engine` execution flow (Prefill & Decode).
1. **Hook into Prefill/Decode:** Implement `prefill_xeon_graph` and `decode_xeon_graph`. Route the inference call to these functions when `DS4_BACKEND_XEON` is selected.
2. **End-to-End Integration Test:** Run a full prompt through the model using `tests/ds4_test.c`.
   - *Test criteria:* The `-xeon` backend must produce the exact same token stream and logits as the `-cpu` reference backend for a complex reasoning prompt (e.g., `tests/test-vectors/prompts/short_reasoning_plain.txt`).

## 5. Rollback Strategy
Because all optimizations are strictly confined to the `DS4_BACKEND_XEON` execution path and conditionally compiled, any instability or precision degradation will not affect the project's core logic. Users can simply omit the `-xeon` flag to fall back to the safe, standard CPU reference implementation.