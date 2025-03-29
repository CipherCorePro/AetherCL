#define _CRT_SECURE_NO_WARNINGS // For Visual Studio (if sprintf/sprintf_s is used)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h> // For FLT_MAX, HUGE_VALF

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// --- Platform Specific Defines ---
#define M_PI 3.14159265358979323846
#define M_1_SQRT2PI 0.39894228040143267794f // Used in GELU backward

#define KERNEL_FP_TYPE float
#define KERNEL_FP_TYPE_STR "float"

#ifndef __linux__
// Windows specific includes/defines might go here if needed
#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_SHARED 1
#define MAP_FAILED ((void *) -1)
// Dummy implementations for non-Linux systems if needed (e.g., for PCI config reading)
void* mmap(void* addr, size_t length, int prot, int flags, int fd, long offset) { return MAP_FAILED; }
int munmap(void* addr, size_t length) { return -1; }
unsigned int read_pci_config(int gpu_index, int offset) { return 0; }
#define DLLEXPORT __declspec(dllexport)
#else
// Linux specific includes
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
unsigned int read_pci_config(int gpu_index, int offset) { return 0; } // Placeholder
#define DLLEXPORT __attribute__((visibility("default")))
#endif

// --- Global Data Type ---
#define FP_TYPE KERNEL_FP_TYPE

// --- OpenCL Globals ---
cl_context context = NULL;
cl_command_queue queue = NULL;
cl_device_id device_id = NULL;
cl_platform_id platform_id = NULL;
int has_fp64_support = 0;     // Flag indicating FP64 support on the device
int has_atomics_support = 0;  // Flag indicating sufficient atomics support (global int cmpxchg)

// --- Kernel/Program Variables ---
cl_program matmul_program = NULL;                 cl_kernel matmul_kernel = NULL;
cl_program softmax_program = NULL;                cl_kernel softmax_kernel = NULL;
cl_program gelu_program = NULL;                   cl_kernel gelu_kernel = NULL;
cl_program add_program = NULL;                    cl_kernel add_kernel = NULL;
cl_program mul_program = NULL;                    cl_kernel mul_kernel = NULL;
cl_program layernorm_program = NULL;              cl_kernel layernorm_kernel = NULL;
cl_program transpose_program = NULL;              cl_kernel transpose_kernel = NULL; // Basic 2D
cl_program gelu_backward_program = NULL;          cl_kernel gelu_backward_kernel = NULL;
cl_program matmul_backward_da_program = NULL;     cl_kernel matmul_backward_da_kernel = NULL;
cl_program matmul_backward_db_program = NULL;     cl_kernel matmul_backward_db_kernel = NULL;
cl_program layernorm_backward_program = NULL;     cl_kernel layernorm_backward_kernel = NULL;
cl_program adam_program = NULL;                   cl_kernel adam_kernel = NULL;
cl_program softmax_backward_program = NULL;       cl_kernel softmax_backward_kernel = NULL;
cl_program mul_backward_program = NULL;           cl_kernel mul_backward_kernel = NULL;
cl_program transpose_backward_program = NULL;     cl_kernel transpose_backward_kernel = NULL; // Basic 2D backward
cl_program embedding_lookup_program = NULL;       cl_kernel embedding_lookup_kernel = NULL;
cl_program embedding_backward_program = NULL;     cl_kernel embedding_backward_kernel = NULL;
cl_program reduce_sum_program = NULL;             cl_kernel reduce_sum_kernel = NULL; // For Bias grad
cl_program broadcast_add_program = NULL;          cl_kernel broadcast_add_kernel = NULL; // General Bias add
cl_program transpose_batched_program = NULL;      cl_kernel transpose_batched_kernel = NULL; // Transpose last two dims
cl_program transpose_12_batched_program = NULL;   cl_kernel transpose_12_batched_kernel = NULL; // Transpose (1,2) for 4D
cl_program matmul_batched_program = NULL;         cl_kernel matmul_batched_kernel = NULL;
cl_program matmul_batched_backward_da_program = NULL; cl_kernel matmul_batched_backward_da_kernel = NULL;
cl_program matmul_batched_backward_db_program = NULL; cl_kernel matmul_batched_backward_db_kernel = NULL;
// --- NEU: Cross Entropy Kernels ---
cl_program log_softmax_program = NULL;            cl_kernel log_softmax_kernel = NULL;
cl_program cross_entropy_program = NULL;          cl_kernel cross_entropy_kernel = NULL;
// --- NEU: Positional Encoding Add Kernel ---
cl_program add_broadcast_pe_program = NULL;       cl_kernel add_broadcast_pe_kernel = NULL;

// --- GPU Command Enumeration ---
typedef enum {
    COMMAND_MATRIX_MULTIPLY = 1,
    COMMAND_SOFTMAX_ROWWISE = 2,
    COMMAND_GELU_ELEMENTWISE = 3,
    COMMAND_ADD_ELEMENTWISE = 4,
    COMMAND_MUL_ELEMENTWISE = 5,
    COMMAND_LAYER_NORM = 6,
    COMMAND_CLONE = 7,
    COMMAND_TRANSPOSE = 8, // Basic 2D transpose
    COMMAND_GELU_BACKWARD_ELEMENTWISE = 9,
    COMMAND_MATMUL_BACKWARD_DA = 10,
    COMMAND_MATMUL_BACKWARD_DB = 11,
    COMMAND_LAYER_NORM_BACKWARD = 12,
    COMMAND_ADAM_UPDATE = 13,
    COMMAND_SOFTMAX_BACKWARD = 14,
    COMMAND_MUL_BACKWARD = 15,
    COMMAND_TRANSPOSE_BACKWARD = 16, // Basic 2D transpose backward
    COMMAND_EMBEDDING_LOOKUP = 17,
    COMMAND_EMBEDDING_BACKWARD = 18,
    COMMAND_REDUCE_SUM_AXIS01 = 19, // For Bias gradient
    COMMAND_BROADCAST_ADD_BIAS = 20, // For general bias addition
    COMMAND_TRANSPOSE_BATCHED = 21, // Transpose last two dims (-2, -1)
    COMMAND_MATRIX_MULTIPLY_BATCHED = 22,
    COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DA = 23,
    COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DB = 24,
    COMMAND_TRANSPOSE_12_BATCHED = 25, // Transpose dims (1, 2) for 4D
    // --- NEU: Commands für Cross Entropy & PE Add ---
    COMMAND_LOG_SOFTMAX_STABLE = 26,
    COMMAND_CROSS_ENTROPY_LOSS_GRAD = 27,
    COMMAND_ADD_BROADCAST_PE = 28
    // ---------------------------------------------
} GPUCommand;

// --- Forward Declarations for Exported Functions ---
DLLEXPORT int initialize_gpu(int gpu_index);
DLLEXPORT void *allocate_gpu_memory(int gpu_index, size_t size);
DLLEXPORT void free_gpu_memory(int gpu_index, void* buffer_handle);
DLLEXPORT int write_host_to_gpu_blocking(int gpu_index, void* gpu_buffer_handle, size_t offset, size_t size, const void* host_source_ptr);
DLLEXPORT int read_gpu_to_host_blocking(int gpu_index, void* gpu_buffer_handle, size_t offset, size_t size, void* host_destination_ptr);
DLLEXPORT int execute_matmul_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int B, int M, int N, int K);
DLLEXPORT int execute_softmax_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_rows, int row_size);
DLLEXPORT int execute_gelu_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_elements);
DLLEXPORT int execute_add_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int num_elements);
DLLEXPORT int execute_mul_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int num_elements);
DLLEXPORT int execute_layernorm_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_rows, int row_size, float eps);
DLLEXPORT int execute_clone_on_gpu(int gpu_index, void* src_buffer, void* dst_buffer, size_t size);
DLLEXPORT int execute_transpose_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int rows, int cols); // Basic 2D
DLLEXPORT int execute_gelu_backward_on_gpu(int gpu_index, void* buffer_input, void* buffer_grad_output, void* buffer_grad_input, int num_elements);
DLLEXPORT int execute_matmul_backward_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_dc, void* buffer_da, void* buffer_db, int B, int M, int N, int K);
DLLEXPORT int execute_layernorm_backward_on_gpu(int gpu_index, void* buffer_dy, void* buffer_x, void* buffer_dx, int num_rows, int row_size, float eps);
DLLEXPORT int execute_adam_update_on_gpu(int gpu_index, void* param_buffer, void* grad_buffer, void* m_buffer, void* v_buffer, int num_elements, int t, float lr, float beta1, float beta2, float eps, float weight_decay);
DLLEXPORT int execute_softmax_backward_on_gpu(int gpu_index, void* buffer_dy, void* buffer_y, void* buffer_dx, int num_rows, int row_size);
DLLEXPORT int execute_mul_backward_on_gpu(int gpu_index, void* buffer_dC, void* buffer_A, void* buffer_B, void* buffer_dA, void* buffer_dB, int num_elements);
DLLEXPORT int execute_transpose_backward_on_gpu(int gpu_index, void* buffer_dC, void* buffer_dA, int rows_A, int cols_A); // Basic 2D backward
DLLEXPORT int execute_embedding_lookup_gpu(int gpu_index, void* idx, void* w, void* o, int b, int s, int d, int v);
DLLEXPORT int execute_embedding_backward_gpu(int gpu_index, void* d_o, void* idx, void* d_w, int b, int s, int d, int v);
DLLEXPORT int execute_reduce_sum_gpu(int gpu_index, void* in, void* out, int B, int M, int N); // For Bias gradient
DLLEXPORT int execute_broadcast_add_gpu(int gpu_index, void* a, void* b, void* c, int B, int M, int N); // General bias add
DLLEXPORT int execute_transpose_batched_gpu(int gpu_index, void* in, void* out, int B_flat, int d1, int d2); // Transpose last two dims
DLLEXPORT int execute_transpose_12_batched_gpu(int gpu_index, void* buffer_in, void* buffer_out, int B, int D1, int D2, int D3);
DLLEXPORT int execute_matmul_batched_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int B, int M, int N, int K);
DLLEXPORT int execute_matmul_batched_backward_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_dc, void* buffer_da, void* buffer_db, int B, int M, int N, int K);
// --- NEU: Cross Entropy & PE Add Exports ---
DLLEXPORT int execute_log_softmax_stable_gpu(int gpu_index, void* input_logits, void* output_log_probs, int B, int S, int V);
DLLEXPORT int execute_cross_entropy_loss_grad_gpu(int gpu_index, void* log_probs, void* target_indices, void* grad_input, void* loss_per_sample, int B, int S, int V);
DLLEXPORT int execute_add_broadcast_pe_gpu(int gpu_index, void* input, void* pe_slice, void* output, int B, int S, int E);
// -------------------------------------------
// Simulation Layer (Dummy implementations for compatibility/testing without GPU)
DLLEXPORT unsigned long long simulated_kernel_allocate(int gpu_index, size_t size);
DLLEXPORT void simulated_kernel_free(int gpu_index, unsigned long long address, size_t size);
DLLEXPORT void simulated_kernel_write(int gpu_index, unsigned long long address, size_t size, const void *source);
DLLEXPORT unsigned int simulated_get_compute_unit_count(int gpu_index);

// --- Internal Helper Functions ---
cl_int compile_opencl_kernel(const char* kernel_source, const char* kernel_name, cl_program* program_out, cl_kernel* kernel_out);
const char* clGetErrorString(cl_int error);
int submit_kernel_command(int gpu_index, GPUCommand command, void *data);
int finish_queue_and_check(int gpu_index, const char* func_name);
void shutdown_driver();
unsigned int get_compute_unit_count(int gpu_index); // Get actual CU count

// --- Kernel Source Code Strings ---

// Matmul (Standard, Handles 3D @ 2D)
const char *matmul_kernel_src = R"CLC(
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
__kernel void matrix_multiply(__global const FP_TYPE *a,       // Input A (B, M, K) or (M, K)
                            __global const FP_TYPE *b,       // Input B (K, N)
                            __global FP_TYPE *c,       // Output C (B, M, N) or (M, N)
                            const int B, const int M, const int N, const int K) {
    int col = get_global_id(0); // N dimension
    int row = get_global_id(1); // M dimension
    int batch_idx = get_global_id(2); // B dimension

    // Check bounds for the output element C[batch_idx, row, col]
    if (batch_idx < B && row < M && col < N) {
        float sum = 0.0f;
        // Calculate offset for A based on batch index. If B=1, this offset is 0.
        size_t a_batch_offset = (size_t)batch_idx * M * K;
        // Calculate offset for C based on batch index.
        size_t c_batch_offset = (size_t)batch_idx * M * N;

        // Perform dot product: sum over k (A[batch, row, k] * B[k, col])
        for (int k = 0; k < K; ++k) {
             // Access A using batch offset + row/k indices
             // Access B using standard k/col indices (implicitly broadcasted over B)
             sum += (float)a[a_batch_offset + row * K + k] * (float)b[(size_t)k * N + col];
        }
        // Write result to output C
        c[c_batch_offset + row * N + col] = (FP_TYPE)sum;
    }
})CLC";

// Matmul Backward dA (Standard)
const char *matmul_backward_dA_kernel_src = R"CLC(
// dA[b,m,k] = sum_n dC[b,m,n] * B[k,n] (equivalent to dC @ B^T)
__kernel void matmul_backward_da(__global const FP_TYPE *dC, // Gradient dC (B, M, N)
                               __global const FP_TYPE *B,  // Original Input B (K, N)
                               __global FP_TYPE *dA, // Output Gradient dA (B, M, K)
                               const int B_dim, const int M_dim, const int N_dim, const int K_dim) {
    int k = get_global_id(0); // K dimension
    int m = get_global_id(1); // M dimension
    int b = get_global_id(2); // B dimension

    // Bounds check for dA element dA[b, m, k]
    if (b < B_dim && m < M_dim && k < K_dim) {
        float gradient_sum = 0.0f;
        size_t dc_batch_offset = (size_t)b * M_dim * N_dim;
        size_t da_batch_offset = (size_t)b * M_dim * K_dim;

        // Sum over N dimension
        for (int n = 0; n < N_dim; ++n) {
            // dC[b, m, n] * B[k, n]
            gradient_sum += (float)dC[dc_batch_offset + m * N_dim + n] * (float)B[(size_t)k * N_dim + n];
        }
        dA[da_batch_offset + m * K_dim + k] = (FP_TYPE)gradient_sum;
    }
})CLC";

// Matmul Backward dB (Standard)
const char *matmul_backward_dB_kernel_src = R"CLC(
// dB[k,n] = sum_b sum_m A[b,m,k] * dC[b,m,n] (equivalent to A^T @ dC, summed over B)
__kernel void matmul_backward_db(__global const FP_TYPE *A,  // Original Input A (B, M, K)
                               __global const FP_TYPE *dC, // Gradient dC (B, M, N)
                               __global FP_TYPE *dB, // Output Gradient dB (K, N)
                               const int B_dim, const int M_dim, const int N_dim, const int K_dim) {
    int n = get_global_id(0); // N dimension
    int k = get_global_id(1); // K dimension

    // Bounds check for dB element dB[k, n]
    if (k < K_dim && n < N_dim) {
        float gradient_sum = 0.0f;
        // Sum over Batch dimension B
        for (int b = 0; b < B_dim; ++b) {
            size_t a_batch_offset = (size_t)b * M_dim * K_dim;
            size_t dc_batch_offset = (size_t)b * M_dim * N_dim;
            // Sum over M dimension
            for (int m = 0; m < M_dim; ++m) {
                // A[b, m, k] * dC[b, m, n]
                gradient_sum += (float)A[a_batch_offset + m * K_dim + k] * (float)dC[dc_batch_offset + m * N_dim + n];
            }
        }
        // Write the final summed gradient to dB
        dB[(size_t)k * N_dim + n] = (FP_TYPE)gradient_sum;
    }
})CLC";

// Softmax (Row-wise, Numerically Stable)
const char *softmax_kernel_src = R"CLC(
#ifndef HUGE_VALF // Standard C float constant for infinity
#define HUGE_VALF (__builtin_huge_valf()) // Use compiler built-in if available
#endif
#ifndef native_exp // Use standard exp if native_exp is not defined/available
#define native_exp exp
#endif

__kernel void softmax_rowwise(__global const FP_TYPE *input, // Input tensor (num_rows, row_size) flattened
                            __global FP_TYPE *output,      // Output tensor (num_rows, row_size) flattened
                            const int num_rows, const int row_size) {
    int row = get_global_id(0); // Index for the row (0 to num_rows-1)

    if (row < num_rows) {
        size_t offset = (size_t)row * row_size; // Base offset for this row
        __global const FP_TYPE* in_row = input + offset;
        __global FP_TYPE* out_row = output + offset;

        // 1. Find max value in the row for numerical stability
        float max_val = -HUGE_VALF;
        for (int i = 0; i < row_size; ++i) {
            if ((float)in_row[i] > max_val) {
                max_val = (float)in_row[i];
            }
        }

        // 2. Calculate sum of exponentials (shifted by max_val)
        float sum_exp = 0.0f;
        for (int i = 0; i < row_size; ++i) {
            sum_exp += native_exp((float)in_row[i] - max_val);
        }

        // 3. Calculate inverse sum (with epsilon for stability if sum_exp is close to zero)
        float inv_sum = 1.0f / (sum_exp + 1e-9f);

        // 4. Calculate softmax probabilities: exp(x_i - max_val) / sum(exp(x_j - max_val))
        for (int i = 0; i < row_size; ++i) {
            out_row[i] = (FP_TYPE)(native_exp((float)in_row[i] - max_val) * inv_sum);
        }
    }
})CLC";

// LogSoftmax (Row-wise, Numerically Stable)
const char *log_softmax_stable_kernel_src = R"CLC(
#ifndef HUGE_VALF
#define HUGE_VALF (__builtin_huge_valf())
#endif
#ifndef native_exp
#define native_exp exp
#endif
#ifndef native_log // Use standard log if native_log is not defined/available
#define native_log log
#endif

__kernel void log_softmax_stable_rowwise(
                    __global const FP_TYPE *input_logits, // Input (B * S, V) flattened
                    __global FP_TYPE *output_log_probs,   // Output (B * S, V) flattened
                    const int num_rows,  // B * S
                    const int row_size   // V (Vocabulary size)
                    ) {
    int row = get_global_id(0); // Index from 0 to B*S - 1

    if (row < num_rows) {
        size_t offset = (size_t)row * row_size;
        __global const FP_TYPE* in_row = input_logits + offset;
        __global FP_TYPE* out_row = output_log_probs + offset;

        // 1. Find max value in the row for numerical stability
        float max_val = -HUGE_VALF;
        for (int i = 0; i < row_size; ++i) {
            if ((float)in_row[i] > max_val) {
                max_val = (float)in_row[i];
            }
        }

        // 2. Calculate sum of exponentials (shifted by max_val)
        float sum_exp = 0.0f;
        for (int i = 0; i < row_size; ++i) {
            sum_exp += native_exp((float)in_row[i] - max_val);
        }

        // 3. Calculate log of the sum of exponentials (LogSumExp trick part 2)
        float log_sum_exp = native_log(sum_exp + 1e-9f); // Add epsilon for stability

        // 4. Calculate log probabilities: log_prob = x - max - log(sum(exp(x - max)))
        for (int i = 0; i < row_size; ++i) {
            out_row[i] = (FP_TYPE)(((float)in_row[i] - max_val) - log_sum_exp);
        }
    }
})CLC";

// Cross Entropy Loss + Gradient w.r.t Logits
const char *cross_entropy_loss_grad_kernel_src = R"CLC(
#ifndef native_exp
#define native_exp exp
#endif

// Calculates loss and gradient for cross-entropy.
// Assumes log_probs input is from a log_softmax operation.
// Target indices are integer class IDs.
__kernel void cross_entropy_loss_grad(
                __global const FP_TYPE* log_probs,      // Input: Log probabilities (B, S, V) flattened (B*S, V)
                __global const int* target_indices,   // Input: Target class indices (B, S) flattened (B*S,)
                __global FP_TYPE* grad_input,         // Output: Gradient w.r.t logits (B, S, V) flattened (B*S, V)
                __global FP_TYPE* loss_per_sample,    // Output: Loss per sample/token (B, S) flattened (B*S,)
                // B and S dimensions are combined into num_rows for simpler kernel launch
                const int num_rows, // B * S
                const int V // Vocabulary size (row_size)
                ) {

    // Global ID maps to the row (token/sample) index
    int row = get_global_id(0); // Index from 0 to num_rows-1

    if (row < num_rows) {
        size_t base_offset = (size_t)row * V; // Offset for log_probs and grad_input row
        __global const FP_TYPE* log_probs_row = log_probs + base_offset;
        __global FP_TYPE* grad_input_row = grad_input + base_offset;

        // Get the target index for this row (sample/token)
        int target_idx = target_indices[row];

        // --- Calculate Gradient: grad = probs - one_hot ---
        // This requires calculating probs = exp(log_probs)
        for (int v = 0; v < V; ++v) {
            float current_log_prob = (float)log_probs_row[v];
            float current_prob = native_exp(current_log_prob);
            float grad_val = current_prob; // Initialize gradient with probability

            // Subtract 1.0f if this is the target class index
            if (v == target_idx) {
                grad_val -= 1.0f;
            }
            grad_input_row[v] = (FP_TYPE)grad_val;
        }

        // --- Calculate Loss: loss = -log_prob[target_idx] ---
        // Ensure target_idx is valid before accessing log_probs
        if (target_idx >= 0 && target_idx < V) {
            float target_log_prob = (float)log_probs_row[target_idx];
            loss_per_sample[row] = (FP_TYPE)(-target_log_prob); // Negative log likelihood
        } else {
            // Handle invalid target index (e.g., padding index like 0)
            // Assign 0 loss for invalid/padding targets.
            loss_per_sample[row] = (FP_TYPE)(0.0f);
        }
    }
}
)CLC";

// Softmax Backward
const char *softmax_backward_kernel_src = R"CLC(
#ifdef CL_HAS_FP64 // Use double for accumulation if supported
    typedef double ACCUM_TYPE;
    #define ACCUM_CONST(x) (double)(x)
#else // Fallback to float accumulation
    typedef float ACCUM_TYPE;
    #define ACCUM_CONST(x) (float)(x)
#endif

// Computes dL/dx = (dL/dy - sum(dL/dy * y)) * y
__kernel void softmax_backward(__global const FP_TYPE *dy_in, // Gradient dL/dy (num_rows, row_size)
                               __global const FP_TYPE *y,    // Output of forward softmax y (num_rows, row_size)
                               __global FP_TYPE *dx,   // Output Gradient dL/dx (num_rows, row_size)
                               const int num_rows, const int row_size) {
    int row = get_global_id(0); // Row index

    if (row < num_rows) {
        size_t offset = (size_t)row * row_size;
        __global const FP_TYPE* dy_row = dy_in + offset;
        __global const FP_TYPE* y_row = y + offset;
        __global FP_TYPE* dx_row = dx + offset;

        // 1. Calculate dot product: sum(dL/dy * y) for this row
        ACCUM_TYPE dot_product = ACCUM_CONST(0.0);
        for (int i = 0; i < row_size; ++i) {
            dot_product += (ACCUM_TYPE)dy_row[i] * (ACCUM_TYPE)y_row[i];
        }

        // 2. Calculate gradient dL/dx for each element in the row
        for (int i = 0; i < row_size; ++i) {
            ACCUM_TYPE dy_val = (ACCUM_TYPE)dy_row[i];
            ACCUM_TYPE y_val = (ACCUM_TYPE)y_row[i];
            // dx_i = (dy_i - dot_product) * y_i
            ACCUM_TYPE dx_val = (dy_val - dot_product) * y_val;
            dx_row[i] = (FP_TYPE)dx_val; // Cast back to original FP_TYPE
        }
    }
})CLC";

// GELU Activation (Elementwise)
const char *gelu_kernel_src = R"CLC(
// Define constants used by GELU
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#ifndef M_SQRT1_2 // 1/sqrt(2)
#define M_SQRT1_2 0.70710678118654752440f
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable // Enable FP64 if available for erf calculation
#ifndef native_erf // Use standard erf if native version is not available/defined
#define native_erf erf
#endif

// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
__kernel void gelu_elementwise(__global const FP_TYPE *input, // Input tensor
                               __global FP_TYPE *output,      // Output tensor
                               const int num_elements) {     // Total number of elements
    int idx = get_global_id(0); // Global element index

    if (idx < num_elements) {
        float x = (float)input[idx]; // Read input as float
        // Calculate GELU using native erf if possible
        float gelu_val = 0.5f * x * (1.0f + native_erf(x * M_SQRT1_2));
        output[idx] = (FP_TYPE)gelu_val; // Write result, cast back to FP_TYPE
    }
})CLC";

// GELU Backward (Elementwise)
const char *gelu_backward_kernel_src = R"CLC(
// Define constants used by GELU backward
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#ifndef M_SQRT1_2 // 1/sqrt(2)
#define M_SQRT1_2 0.70710678118654752440f
#endif
#ifndef M_1_SQRT2PI // 1/sqrt(2*pi) - Used in PDF
#define M_1_SQRT2PI 0.39894228040143267794f
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable // Enable FP64 for erf/exp if available
#ifndef native_erf // Use standard erf if native is not defined
#define native_erf erf
#endif
#ifndef native_exp // Use standard exp if native is not defined
#define native_exp exp
#endif

// dGELU/dx = 0.5 * (1 + erf(x / sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-0.5 * x^2)
//           = CDF(x) + x * PDF(x)
// dL/dx = dL/dy * dGELU/dx
__kernel void gelu_backward_elementwise(__global const FP_TYPE *input,       // Original input x to GELU forward
                                       __global const FP_TYPE *grad_output, // Gradient dL/dy from subsequent layer
                                       __global FP_TYPE *grad_input,  // Output gradient dL/dx
                                       const int num_elements) {      // Total number of elements
    int idx = get_global_id(0); // Global element index

    if (idx < num_elements) {
        float x = (float)input[idx];       // Original input value
        float dy = (float)grad_output[idx]; // Incoming gradient

        // Calculate CDF term: 0.5 * (1 + erf(x / sqrt(2)))
        float cdf_term = 0.5f * (1.0f + native_erf(x * M_SQRT1_2));
        // Calculate PDF term: (1/sqrt(2*pi)) * exp(-0.5 * x^2)
        float pdf_term = M_1_SQRT2PI * native_exp(-0.5f * x * x);
        // Calculate dGELU/dx = CDF(x) + x * PDF(x)
        float dgelu_dx = cdf_term + x * pdf_term;

        // Calculate final gradient: dL/dx = dL/dy * dGELU/dx
        grad_input[idx] = (FP_TYPE)(dy * dgelu_dx); // Write result, cast back to FP_TYPE
    }
})CLC";

// Add (Elementwise)
const char *add_kernel_src = R"CLC(
// c[i] = a[i] + b[i]
__kernel void add_elementwise(__global const FP_TYPE *a, // Input tensor A
                             __global const FP_TYPE *b, // Input tensor B
                             __global FP_TYPE *c, // Output tensor C
                             const int num_elements) { // Total number of elements
    int idx = get_global_id(0); // Global element index

    if (idx < num_elements) {
        c[idx] = (FP_TYPE)((float)a[idx] + (float)b[idx]); // Perform addition and cast back
    }
})CLC";

// Multiply (Elementwise)
const char *mul_kernel_src = R"CLC(
// c[i] = a[i] * b[i]
__kernel void mul_elementwise(__global const FP_TYPE *a, // Input tensor A
                             __global const FP_TYPE *b, // Input tensor B
                             __global FP_TYPE *c, // Output tensor C
                             const int num_elements) { // Total number of elements
    int idx = get_global_id(0); // Global element index

    if (idx < num_elements) {
        c[idx] = (FP_TYPE)((float)a[idx] * (float)b[idx]); // Perform multiplication and cast back
    }
})CLC";

// Multiply Backward (Elementwise)
const char *mul_backward_kernel_src = R"CLC(
// Computes gradients for elementwise multiplication C = A * B
// dA = dC * B
// dB = dC * A
__kernel void mul_backward(__global const FP_TYPE *dC, // Gradient dL/dC from subsequent layer
                         __global const FP_TYPE *A,  // Original Input A from forward pass
                         __global const FP_TYPE *B,  // Original Input B from forward pass
                         __global FP_TYPE *dA, // Output gradient dL/dA (can be NULL conceptually, but kernel expects a buffer if arg is set)
                         __global FP_TYPE *dB, // Output gradient dL/dB (can be NULL conceptually, but kernel expects a buffer if arg is set)
                         const int num_elements) { // Total number of elements
    int idx = get_global_id(0); // Global element index

    if (idx < num_elements) {
        float dC_val = (float)dC[idx]; // Incoming gradient
        float A_val = (float)A[idx];   // Original input A
        float B_val = (float)B[idx];   // Original input B

        // Calculate gradient w.r.t. A: dA = dC * B
        // The check whether to calculate this happens *before* calling the kernel
        // by checking if the dA buffer pointer is provided by the host.
        // If dA is not 0 (meaning a valid buffer was passed), write the result.
        // WARNING: Direct check like dA != 0 is NOT reliable for buffer validity inside kernel!
        //          The HOST code MUST ensure only valid buffers are passed if grads are needed.
        //          We write unconditionally assuming the host passed valid buffers if required.
        // if (dA != 0) { // <-- REMOVE THIS CHECK
            dA[idx] = (FP_TYPE)(dC_val * B_val);
        // }

        // Calculate gradient w.r.t. B: dB = dC * A
        // Same assumption: Host ensures dB is valid if the gradient is needed.
        // if (dB != 0) { // <-- REMOVE THIS CHECK
            dB[idx] = (FP_TYPE)(dC_val * A_val);
        // }
    }
})CLC";

// Layer Normalization (Row-wise)
const char *layernorm_kernel_src = R"CLC(
// Define accumulation type based on FP64 support
#ifdef CL_HAS_FP64
    typedef double ACCUM_TYPE;
    #define ACCUM_CONST(x) (double)(x)
#else
    typedef float ACCUM_TYPE;
    #define ACCUM_CONST(x) (float)(x)
#endif

#ifndef native_rsqrt // Use standard rsqrt if native version is not available
#define native_rsqrt rsqrt
#endif

// Performs Layer Normalization along the last dimension.
__kernel void layer_norm(__global const FP_TYPE *input, // Input tensor (num_rows, row_size) flattened
                         __global FP_TYPE *output,      // Output tensor (num_rows, row_size) flattened
                         const int num_rows, const int row_size, const float cl_eps) { // Epsilon added in C host code
    int row = get_global_id(0); // Row index

    if (row < num_rows) {
        size_t offset = (size_t)row * row_size; // Base offset for this row
        __global const FP_TYPE* in_row = input + offset;
        __global FP_TYPE* out_row = output + offset;

        // 1. Calculate mean of the row
        ACCUM_TYPE mean = ACCUM_CONST(0.0);
        for (int i = 0; i < row_size; ++i) {
            mean += (ACCUM_TYPE)in_row[i];
        }
        mean /= ACCUM_CONST(row_size);

        // 2. Calculate variance of the row
        ACCUM_TYPE variance = ACCUM_CONST(0.0);
        for (int i = 0; i < row_size; ++i) {
            ACCUM_TYPE diff = (ACCUM_TYPE)in_row[i] - mean;
            variance += diff * diff;
        }
        variance /= ACCUM_CONST(row_size);

        // 3. Calculate inverse standard deviation (with epsilon)
        // Use native_rsqrt for potential performance improvement
        ACCUM_TYPE eps_accum = (ACCUM_TYPE)cl_eps;
        ACCUM_TYPE inv_stddev = native_rsqrt(variance + eps_accum);

        // 4. Normalize the row: output = (input - mean) * inv_stddev
        for (int i = 0; i < row_size; ++i) {
            out_row[i] = (FP_TYPE)(((ACCUM_TYPE)in_row[i] - mean) * inv_stddev);
        }
    }
})CLC";

// Layer Normalization Backward
const char *layernorm_backward_kernel_src = R"CLC(
#ifdef CL_HAS_FP64
    typedef double ACCUM_TYPE;
    #define ACCUM_CONST(x) (double)(x)
#else
    typedef float ACCUM_TYPE;
    #define ACCUM_CONST(x) (float)(x)
#endif

#ifndef native_rsqrt
#define native_rsqrt rsqrt
#endif

// Calculates gradient for Layer Normalization (without affine parameters gamma/beta).
__kernel void layer_norm_backward(__global const FP_TYPE *dy, // Gradient dL/dy from subsequent layer
                                __global const FP_TYPE *x,  // Original input x to forward LayerNorm
                                __global FP_TYPE *dx, // Output gradient dL/dx
                                const int num_rows, const int row_size, const float cl_eps) {
    int row = get_global_id(0); // Row index

    if (row < num_rows) {
        size_t offset = (size_t)row * row_size;
        __global const FP_TYPE* dy_row = dy + offset;
        __global const FP_TYPE* x_row = x + offset;
        __global FP_TYPE* dx_row = dx + offset;

        // --- Recompute mean and variance (needed for backward) ---
        ACCUM_TYPE mean = ACCUM_CONST(0.0);
        for (int i = 0; i < row_size; ++i) { mean += (ACCUM_TYPE)x_row[i]; }
        mean /= ACCUM_CONST(row_size);

        ACCUM_TYPE variance = ACCUM_CONST(0.0);
        for (int i = 0; i < row_size; ++i) { ACCUM_TYPE diff = (ACCUM_TYPE)x_row[i] - mean; variance += diff * diff; }
        variance /= ACCUM_CONST(row_size);

        ACCUM_TYPE eps_accum = (ACCUM_TYPE)cl_eps;
        ACCUM_TYPE inv_stddev = native_rsqrt(variance + eps_accum); // 1 / sqrt(var + eps)
        ACCUM_TYPE N_accum = ACCUM_CONST(row_size);

        // --- Calculate intermediate sums needed for the gradient ---
        ACCUM_TYPE sum_dy = ACCUM_CONST(0.0);           // sum(dy)
        ACCUM_TYPE sum_dy_xhat = ACCUM_CONST(0.0);    // sum(dy * x_hat)
        // Calculate x_hat = (x - mean) * inv_stddev on the fly
        for (int i = 0; i < row_size; i++) {
            ACCUM_TYPE x_hat = ((ACCUM_TYPE)x_row[i] - mean) * inv_stddev;
            ACCUM_TYPE dy_accum = (ACCUM_TYPE)dy_row[i];
            sum_dy += dy_accum;
            sum_dy_xhat += dy_accum * x_hat;
        }

        // --- Calculate gradient dL/dx for each element ---
        // Formula (simplified, without affine params):
        // dx = (1/N) * inv_stddev * [ N*dy - sum(dy) - x_hat * sum(dy * x_hat) ]
        for (int i = 0; i < row_size; i++) {
            ACCUM_TYPE x_hat = ((ACCUM_TYPE)x_row[i] - mean) * inv_stddev; // Recompute x_hat
            ACCUM_TYPE dy_accum = (ACCUM_TYPE)dy_row[i];

            ACCUM_TYPE term1 = N_accum * dy_accum; // N * dy_i
            ACCUM_TYPE term2 = sum_dy;             // sum(dy)
            ACCUM_TYPE term3 = x_hat * sum_dy_xhat; // x_hat_i * sum(dy * x_hat)

            // Combine terms and scale
            ACCUM_TYPE dx_accum = (ACCUM_CONST(1.0) / N_accum) * inv_stddev * (term1 - term2 - term3);

            dx_row[i] = (FP_TYPE)dx_accum; // Write final gradient
        }
    }
})CLC";

// Transpose (Basic 2D)
const char *transpose_kernel_src = R"CLC(
// Transposes a 2D matrix. Output[col, row] = Input[row, col]
__kernel void transpose(__global const FP_TYPE *input, // Input matrix (rows, cols)
                        __global FP_TYPE *output,      // Output matrix (cols, rows)
                        const int rows, const int cols) {
    // Use 2D global IDs corresponding to the OUTPUT matrix dimensions
    int out_row = get_global_id(0); // Corresponds to input cols (0 to cols-1)
    int out_col = get_global_id(1); // Corresponds to input rows (0 to rows-1)

    // Bounds check for output indices
    if (out_row < cols && out_col < rows) {
        // Calculate linear index for output[out_row, out_col] (stride is rows)
        size_t output_idx = (size_t)out_row * rows + out_col;
        // Calculate linear index for input[out_col, out_row] (stride is cols)
        size_t input_idx = (size_t)out_col * cols + out_row;

        output[output_idx] = input[input_idx];
    }
})CLC";

// Transpose Backward (Basic 2D)
const char *transpose_backward_kernel_src = R"CLC(
// Backward of transpose Y=X^T is dX = (dY)^T
// This kernel effectively performs another transpose on the incoming gradient dY.
__kernel void transpose_backward(__global const FP_TYPE *dC, // Gradient dL/dC (dims: cols_A x rows_A)
                               __global FP_TYPE *dA,       // Output gradient dL/dA (dims: rows_A x cols_A)
                               const int rows_A, const int cols_A) {
    // Use 2D global IDs corresponding to the OUTPUT gradient dA dimensions
    int dA_row = get_global_id(0); // 0 to rows_A-1
    int dA_col = get_global_id(1); // 0 to cols_A-1

    // Bounds check for dA indices
    if (dA_row < rows_A && dA_col < cols_A) {
        // Calculate linear index for dA[dA_row, dA_col] (stride is cols_A)
        size_t dA_idx = (size_t)dA_row * cols_A + dA_col;
        // Calculate corresponding linear index in dC (transposed access)
        // dC has dimensions (cols_A, rows_A), so dC[dA_col, dA_row]
        size_t dC_idx = (size_t)dA_col * rows_A + dA_row; // Stride is rows_A

        dA[dA_idx] = dC[dC_idx]; // Perform the transpose copy
    }
})CLC";

// Adam Optimizer Update
const char *adam_kernel_src = R"CLC(
// Use standard sqrt if native version is not available
#ifndef native_sqrt
#define native_sqrt sqrt
#endif

// Performs Adam weight update step.
// Note: m and v states are expected to be float, regardless of KERNEL_FP_TYPE.
__kernel void adam_update(__global FP_TYPE *param,       // Parameter tensor (to be updated)
                         __global const FP_TYPE *grad,       // Gradient tensor dL/dparam
                         __global float *m,           // Adam state m (1st moment, float)
                         __global float *v,           // Adam state v (2nd moment, float)
                         const int num_elements,   // Total number of elements
                         const float lr,             // Learning rate
                         const float beta1,          // Adam beta1 hyperparameter
                         const float beta2,          // Adam beta2 hyperparameter
                         const float epsilon,        // Adam epsilon hyperparameter
                         const float weight_decay,   // Weight decay factor (L2 regularization)
                         const float beta1_t,        // Precomputed beta1^t
                         const float beta2_t) {      // Precomputed beta2^t
    int idx = get_global_id(0); // Global element index

    if (idx < num_elements) {
        // Read values, using float for internal Adam calculations for stability/consistency
        float p = (float)param[idx];
        float g = (float)grad[idx];
        float m_curr = m[idx]; // Read current m state
        float v_curr = v[idx]; // Read current v state

        // Apply weight decay (L2 regularization) if enabled
        if (weight_decay > 0.0f) {
            g += weight_decay * p; // Add weight decay term to the gradient
        }

        // Update biased first moment estimate (m)
        float m_new = beta1 * m_curr + (1.0f - beta1) * g;
        // Update biased second raw moment estimate (v)
        float v_new = beta2 * v_curr + (1.0f - beta2) * (g * g);

        // Compute bias-corrected first moment estimate (m_hat)
        // Add small epsilon to denominator for numerical stability, although 1-beta1_t is usually far from 0 early on.
        float m_hat = m_new / (1.0f - beta1_t + 1e-9f);
        // Compute bias-corrected second raw moment estimate (v_hat)
        float v_hat = v_new / (1.0f - beta2_t + 1e-9f);

        // Compute the parameter update step
        // update = lr * m_hat / (sqrt(v_hat) + epsilon)
        float update = lr * m_hat / (native_sqrt(v_hat) + epsilon);

        // Apply the update to the parameter
        float p_new = p - update;

        // Write back updated parameter and Adam states
        param[idx] = (FP_TYPE)p_new; // Cast back to original parameter type
        m[idx] = m_new;             // Write updated m state (float)
        v[idx] = v_new;             // Write updated v state (float)
    }
})CLC";

// Embedding Lookup (GPU Version)
const char *embedding_lookup_kernel_src = R"CLC(
// Performs embedding lookup: output[b, s, :] = weights[indices[b, s], :]
__kernel void embedding_lookup(
             __global const int* indices,     // Input: Indices tensor (B, S) flattened (B*S,)
             __global const FP_TYPE* weights, // Input: Weight matrix (V, D)
             __global FP_TYPE* output,        // Output: Output tensor (B, S, D) flattened (B*S, D)
             const int seq_len,     // S
             const int embed_dim,   // D
             const int vocab_size   // V
             // B is implicit via global size dim 1
             ) {
    // Use 2D global IDs mapping to (s, b)
    int s = get_global_id(0); // Sequence dimension index (0 to S-1)
    int b = get_global_id(1); // Batch dimension index (0 to B-1)

    // Calculate linear index for the input indices array (B*S,)
    size_t indices_idx = (size_t)b * seq_len + s;

    // Read the vocabulary index for this (b, s) position
    int vocab_idx = indices[indices_idx];

    // Calculate the base offset for the output tensor row (B*S, D)
    size_t output_offset = ((size_t)b * seq_len + s) * embed_dim;

    // --- Bounds Check for Vocabulary Index ---
    // Check if the vocabulary index is valid (within [0, vocab_size-1])
    if (vocab_idx < 0 || vocab_idx >= vocab_size) {
        // Handle out-of-bounds index (e.g., padding or error) -> Output zeros
        for(int d = 0; d < embed_dim; ++d) {
            output[output_offset + d] = (FP_TYPE)0.0;
        }
        return; // Exit early for this work-item
    }
    // -----------------------------------------

    // Calculate the base offset for the corresponding row in the weight matrix (V, D)
    size_t weight_offset = (size_t)vocab_idx * embed_dim;

    // Copy the embedding vector from weights to output for the full embedding dimension D
    for (int d = 0; d < embed_dim; ++d) {
        output[output_offset + d] = weights[weight_offset + d];
    }
})CLC";

// Embedding Backward (GPU Version with Atomics)
// Embedding Backward (GPU Version with Atomics)
const char *embedding_backward_kernel_src = R"CLC(
// Enable required extensions for atomics
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable // Needed if FP_TYPE is double for 64-bit atomics

// Define ATOMIC_ADD_FP based on compiler flags set by host and FP_TYPE
#ifdef CL_HAS_ATOMICS // This macro is defined by the host C code if atomics are supported
    #if defined(FP_TYPE) && FP_TYPE == float
        // --- FP32 Atomic Add Implementation using atomic_cmpxchg ---
        inline void atomic_add_float_impl(__global float *addr, float val) {
            union { float f; int i; } old_val, new_val; // Union for type punning
            do {
                old_val.f = *addr; // Read current value
                new_val.f = old_val.f + val; // Calculate desired new value
            } while (atomic_cmpxchg((volatile __global int*)addr, old_val.i, new_val.i) != old_val.i);
        }
        #define ATOMIC_ADD_FP(ptr, val) atomic_add_float_impl(ptr, val) // Macro for FP32

    #elif defined(FP_TYPE) && FP_TYPE == double && defined(cl_khr_int64_base_atomics)
        // --- FP64 Atomic Add Implementation (Requires 64-bit integer atomics support) ---
        #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
        inline void atomic_add_double_impl(__global double *addr, double val) {
            union { double d; long l; } old_val, new_val; // Union for 64-bit types
            do {
                old_val.d = *addr;
                new_val.d = old_val.d + val;
            } while (atom_cmpxchg((volatile __global long*)addr, old_val.l, new_val.l) != old_val.l);
        }
        #define ATOMIC_ADD_FP(ptr, val) atomic_add_double_impl(ptr, val) // Macro for FP64

    #else // Unsupported FP_TYPE for atomics or missing 64-bit support
        // If atomics are indicated but type is wrong, use unsafe fallback.
        // The runtime check in Python is the primary safeguard.
        // #warning "Unsupported FP_TYPE for atomic_add or missing 64-bit atomics. Using non-atomic add -> RACE CONDITION LIKELY!" // REMOVED
        #define ATOMIC_ADD_FP(ptr, val) (*(ptr) += (val)) // UNSAFE FALLBACK
    #endif
#else // CL_HAS_ATOMICS not defined by host (atomics support disabled or not detected)
    // Provide a non-atomic fallback. The runtime check is the main safeguard.
    // #warning "CL_HAS_ATOMICS not defined by host. Using non-atomic add for embedding backward! RACE CONDITION HIGHLY LIKELY!" // REMOVED
    #define ATOMIC_ADD_FP(ptr, val) (*(ptr) += (val)) // UNSAFE FALLBACK
#endif

// Performs scatter-add for embedding backward pass.
// grad_weights[indices[b, s], :] += grad_output[b, s, :]
__kernel void embedding_backward_scatter_add(
                 __global const FP_TYPE* grad_output, // Input: Gradient dL/dOutput (B, S, D) flattened (B*S, D)
                 __global const int* indices,         // Input: Indices used in forward (B, S) flattened (B*S,)
                 __global FP_TYPE* grad_weights,      // Output: Gradient dL/dWeights (V, D), requires atomic updates
                 const int seq_len,     // S
                 const int embed_dim,   // D
                 const int vocab_size   // V
                 ) {
    int s = get_global_id(0); int b = get_global_id(1);
    size_t indices_idx = (size_t)b * seq_len + s;
    int vocab_idx = indices[indices_idx];
    if (vocab_idx < 0 || vocab_idx >= vocab_size) { return; }
    size_t grad_output_offset = ((size_t)b * seq_len + s) * embed_dim;
    size_t grad_weight_offset = (size_t)vocab_idx * embed_dim;
    for (int d = 0; d < embed_dim; ++d) {
        FP_TYPE grad_val = grad_output[grad_output_offset + d];
        __global FP_TYPE* target_addr = grad_weights + grad_weight_offset + d;
        ATOMIC_ADD_FP(target_addr, grad_val); // Uses atomic or unsafe fallback based on CL_HAS_ATOMICS define
    }
}
)CLC";

// Reduce Sum (Axis 0 and 1 for Bias Gradient)
const char *reduce_sum_kernel_src = R"CLC(
// Enable extensions if needed for local memory atomics (though not used here)
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable // For ACCUM_TYPE if needed

// Define work-group size for reduction (can be tuned)
#ifndef WORK_GROUP_SIZE_REDUCE
#define WORK_GROUP_SIZE_REDUCE 256
#endif

// Performs reduction sum over axes 0 (B) and 1 (M) of a 3D tensor (B, M, N).
// Output is a 1D tensor of size N.
// Uses local memory for efficient work-group reduction.
__kernel void reduce_sum_axis01(
                __global const FP_TYPE* input, // Input tensor (B, M, N)
                __global FP_TYPE* output,      // Output tensor (N)
                const int B, const int M, const int N,
                __local FP_TYPE* local_sums    // Local memory buffer, size = WORK_GROUP_SIZE_REDUCE
                ) {
    // --- Work-item / Work-group IDs ---
    int n_out_idx = get_group_id(0); // Index for the output element N this group calculates (0 to N-1)
    int tid = get_local_id(0);       // Local thread ID within the work-group (0 to WGS-1)
    int wg_size = get_local_size(0); // Work-group size (WORK_GROUP_SIZE_REDUCE)

    // Total number of elements to sum over per output element n_out_idx (B * M)
    size_t items_to_reduce = (size_t)B * M;
    // Accumulator for this thread's partial sum
    FP_TYPE thread_sum = (FP_TYPE)0.0;

    // --- Grid-Stride Loop for Initial Summation ---
    // Each thread sums a portion of the items along the B and M dimensions for a fixed N index.
    if (n_out_idx < N) { // Ensure the group works on a valid output index
        for (size_t i = tid; i < items_to_reduce; i += wg_size) {
            // Decompose linear index 'i' (ranging 0 to B*M-1) into batch 'b' and dimension 'm' indices
            int b = i / M;
            int m = i % M;
            // Calculate the linear index into the 3D input tensor (B, M, N)
            size_t input_idx = (size_t)b * M * N + (size_t)m * N + n_out_idx;
            // Accumulate the sum for this thread
            thread_sum += input[input_idx];
        }
    }
    // Store the thread's partial sum into local memory
    local_sums[tid] = thread_sum;

    // --- Work-Group Reduction using Local Memory ---
    // Synchronize threads within the work-group to ensure all writes to local_sums are complete.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform parallel reduction within the work-group (e.g., tree reduction)
    for (int offset = wg_size / 2; offset > 0; offset /= 2) {
        if (tid < offset) { // Only threads in the first half of the current range add
            local_sums[tid] += local_sums[tid + offset];
        }
        // Synchronize after each reduction step to ensure additions are complete
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- Write Final Result ---
    // The first thread (tid == 0) of each work-group now holds the final sum for its assigned n_out_idx.
    if (tid == 0 && n_out_idx < N) { // Check group index validity again before writing
        output[n_out_idx] = local_sums[0];
    }
}
)CLC";

// Broadcast Add (Bias Vector)
const char *broadcast_add_kernel_src = R"CLC(
// Performs broadcast addition: C[b, m, n] = A[b, m, n] + B_bias[n]
__kernel void broadcast_add_bias(
                __global const FP_TYPE* a,     // Input tensor A (B, M, N)
                __global const FP_TYPE* b_bias,// Input bias vector B (N)
                __global FP_TYPE* c,           // Output tensor C (B, M, N)
                const int M, const int N      // Dimensions M and N (B is implicit from GWS dim 2)
                ) {
    // Use 3D global IDs mapping to (n, m, b) -> aligns with typical memory access patterns
    int n = get_global_id(0); // Index along dimension N (0 to N-1)
    int m = get_global_id(1); // Index along dimension M (0 to M-1)
    int b = get_global_id(2); // Index along dimension B (0 to B-1)

    // Bounds check (n and m). 'b' is implicitly checked by the global work size dim 2.
    if (n < N && m < M) {
       // Calculate the linear index for tensors A and C (shape B, M, N)
       size_t idx_a_c = (size_t)b * M * N + (size_t)m * N + n;
       // Index for the bias tensor B (shape N) is simply 'n'
       int idx_b = n;

       // Perform the broadcast addition
       c[idx_a_c] = a[idx_a_c] + b_bias[idx_b];
    }
}
)CLC";

// Transpose Last Two Dimensions (Batched)
const char *transpose_batched_kernel_src = R"CLC(
// Transposes the last two dimensions of a tensor: (..., D1, D2) -> (..., D2, D1)
__kernel void transpose_batched_last_two(
                __global const FP_TYPE* input, // Input tensor (..., D1, D2)
                __global FP_TYPE* output,      // Output tensor (..., D2, D1)
                const int Dim1,           // Size of the dimension originally at -2
                const int Dim2            // Size of the dimension originally at -1
                // Leading dimensions (...) are flattened into GWS dim 2 (b_linear)
                ) {
    // Use 3D global IDs mapping to output dimensions (d2_out, d1_out, b_linear)
    int d2_out = get_global_id(0); // Index along the new Dim2 (output dim -1, size Dim1)
    int d1_out = get_global_id(1); // Index along the new Dim1 (output dim -2, size Dim2)
    int b_linear = get_global_id(2); // Linearized index for the leading batch dimensions

    // Calculate corresponding indices in the INPUT tensor
    int d1_in = d2_out; // Input dim1 index maps from output dim2 index
    int d2_in = d1_out; // Input dim2 index maps from output dim1 index

    // --- Bounds Check ---
    // Check if the output indices d1_out, d2_out are within their respective dimension sizes (Dim2, Dim1)
    if (d1_out < Dim2 && d2_out < Dim1) {
        // Calculate the size (stride) of one batch slice being transposed (D1 * D2)
        size_t slice_stride = (size_t)Dim1 * Dim2;
        // Calculate the base offset for the current batch slice in both input and output
        size_t batch_offset = (size_t)b_linear * slice_stride;

        // Calculate linear index within the input slice (layout ..., D1, D2)
        // Access input[b_linear][d1_in][d2_in]
        size_t input_idx  = batch_offset + (size_t)d1_in * Dim2 + d2_in;

        // Calculate linear index within the output slice (layout ..., D2, D1) - Note stride change!
        // Access output[b_linear][d2_out][d1_out]
        size_t output_idx = batch_offset + (size_t)d2_out * Dim1 + d1_out;

        // Perform the copy
        output[output_idx] = input[input_idx];
    }
}
)CLC";

// Transpose Dimensions 1 and 2 (Batched, 4D)
const char *transpose_12_batched_kernel_src = R"CLC(
// Transposes dimensions 1 and 2 of a 4D tensor: (B, D1, D2, D3) -> (B, D2, D1, D3)
__kernel void transpose_12_batched(
                __global const FP_TYPE* input,  // Input tensor (B, D1, D2, D3)
                __global FP_TYPE* output, // Output tensor (B, D2, D1, D3)
                const int B, const int D1, const int D2, const int D3
                ) {
    // Use 3D global IDs mapping: GWS = (D3, D1, D2*B) -> matches layout better
    int d3 = get_global_id(0);             // Index along the innermost dimension D3 (0 to D3-1)
    int d1_out = get_global_id(1);         // Index along output dim D1' (size D1)
    int d2_b_linear = get_global_id(2);    // Linearized index for (output D2', batch B)

    // De-linearize d2_b_linear to get batch 'b' and output D2' index 'd2_out'
    int d2_out = d2_b_linear % D2; // Index along output dim D2' (size D2)
    int b      = d2_b_linear / D2; // Batch index (0 to B-1)

    // --- Map output indices to input indices ---
    int d1_in = d1_out; // Input D1 index maps from output D1' index
    int d2_in = d2_out; // Input D2 index maps from output D2' index

    // --- Bounds Check ---
    // Check if the calculated indices are within the valid ranges for B, D1, D2, D3
    if (b < B && d1_out < D1 && d2_out < D2 && d3 < D3) {

        // Calculate linear index for the input tensor (Layout: B, D1, D2, D3)
        size_t input_idx = (size_t)b * D1 * D2 * D3 + // Offset for batch b
                           (size_t)d1_in * D2 * D3 +  // Offset for input dim d1_in
                           (size_t)d2_in * D3 +       // Offset for input dim d2_in
                           d3;                        // Offset for input dim d3

        // Calculate linear index for the output tensor (Layout: B, D2, D1, D3)
        size_t output_idx = (size_t)b * D2 * D1 * D3 + // Offset for batch b
                            (size_t)d2_out * D1 * D3 + // Offset for output dim d2_out (size D2)
                            (size_t)d1_out * D3 +    // Offset for output dim d1_out (size D1)
                            d3;                        // Offset for output dim d3

        // Perform the transpose copy
        output[output_idx] = input[input_idx];
    }
}
)CLC";

// Matmul (Batched, 3D @ 3D)
const char *matmul_batched_kernel_src = R"CLC(
// Performs batched matrix multiplication: C[b,:,:] = A[b,:,:] @ B[b,:,:]
__kernel void matmul_batched(__global const FP_TYPE *a, // Input A (B, M, K)
                           __global const FP_TYPE *b, // Input B (B, K, N)
                           __global FP_TYPE *c, // Output C (B, M, N)
                           const int B, const int M, const int N, const int K) {
    // Use 3D global IDs mapping to output dimensions (col_N, row_M, batch_B)
    int col = get_global_id(0); // Index along N dimension (0 to N-1)
    int row = get_global_id(1); // Index along M dimension (0 to M-1)
    int batch_idx = get_global_id(2); // Index along B dimension (0 to B-1)

    // Bounds check for output element C[batch_idx, row, col]
    if (batch_idx < B && row < M && col < N) {
        float sum = 0.0f;
        // Calculate base offsets for the current batch in A, B, and C
        size_t a_batch_offset = (size_t)batch_idx * M * K;
        size_t b_batch_offset = (size_t)batch_idx * K * N;
        size_t c_batch_offset = (size_t)batch_idx * M * N;

        // Perform dot product over dimension K
        for (int k = 0; k < K; ++k) {
             // A[batch, row, k] * B[batch, k, col]
             sum += (float)a[a_batch_offset + row * K + k] * (float)b[b_batch_offset + k * N + col];
        }
        // Write the result to output tensor C
        c[c_batch_offset + row * N + col] = (FP_TYPE)sum;
    }
})CLC";

// Matmul Backward dA (Batched)
const char *matmul_batched_backward_dA_kernel_src = R"CLC(
// dA[b,m,k] = sum_n dC[b,m,n] * B[b,k,n] (equivalent to dC @ B^T, batched)
__kernel void matmul_batched_backward_da(__global const FP_TYPE *dC, // Gradient dC (B, M, N)
                                       __global const FP_TYPE *B,  // Original Input B (B, K, N)
                                       __global FP_TYPE *dA, // Output Gradient dA (B, M, K)
                                       const int B_dim, const int M_dim, const int N_dim, const int K_dim) {
    // Use 3D global IDs mapping to output dA dimensions (k_K, m_M, b_B)
    int k = get_global_id(0); // Index along K dimension (0 to K_dim-1)
    int m = get_global_id(1); // Index along M dimension (0 to M_dim-1)
    int b = get_global_id(2); // Index along B dimension (0 to B_dim-1)

    // Bounds check for dA element dA[b, m, k]
    if (b < B_dim && m < M_dim && k < K_dim) {
        float gradient_sum = 0.0f;
        // Calculate base offsets for the current batch
        size_t dc_batch_offset = (size_t)b * M_dim * N_dim;
        size_t b_batch_offset  = (size_t)b * K_dim * N_dim;
        size_t da_batch_offset = (size_t)b * M_dim * K_dim;

        // Perform sum over dimension N
        for (int n = 0; n < N_dim; ++n) {
            // dC[b, m, n] * B[b, k, n]
            gradient_sum += (float)dC[dc_batch_offset + m * N_dim + n] * (float)B[b_batch_offset + k * N_dim + n];
        }
        // Write the result to output tensor dA
        dA[da_batch_offset + m * K_dim + k] = (FP_TYPE)gradient_sum;
    }
})CLC";

// Matmul Backward dB (Batched)
const char *matmul_batched_backward_dB_kernel_src = R"CLC(
// dB[b,k,n] = sum_m A[b,m,k] * dC[b,m,n] (equivalent to A^T @ dC, batched)
__kernel void matmul_batched_backward_db(__global const FP_TYPE *A,  // Original Input A (B, M, K)
                                       __global const FP_TYPE *dC, // Gradient dC (B, M, N)
                                       __global FP_TYPE *dB, // Output Gradient dB (B, K, N)
                                       const int B_dim, const int M_dim, const int N_dim, const int K_dim) {
    // Use 3D global IDs mapping to output dB dimensions (n_N, k_K, b_B)
    int n = get_global_id(0); // Index along N dimension (0 to N_dim-1)
    int k = get_global_id(1); // Index along K dimension (0 to K_dim-1)
    int b = get_global_id(2); // Index along B dimension (0 to B_dim-1)

    // Bounds check for dB element dB[b, k, n]
    if (b < B_dim && k < K_dim && n < N_dim) {
        float gradient_sum = 0.0f;
        // Calculate base offsets for the current batch
        size_t a_batch_offset  = (size_t)b * M_dim * K_dim;
        size_t dc_batch_offset = (size_t)b * M_dim * N_dim;
        size_t db_batch_offset = (size_t)b * K_dim * N_dim;

        // Perform sum over dimension M
        for (int m = 0; m < M_dim; ++m) {
            // A[b, m, k] * dC[b, m, n]
            gradient_sum += (float)A[a_batch_offset + m * K_dim + k] * (float)dC[dc_batch_offset + m * N_dim + n];
        }
        // Write the result to output tensor dB
        dB[db_batch_offset + k * N_dim + n] = (FP_TYPE)gradient_sum;
    }
})CLC";

// Broadcast Add for Positional Encoding
const char *add_broadcast_pe_kernel_src = R"CLC(
// Performs broadcast addition: Output[b, s, e] = Input[b, s, e] + PE[s, e]
__kernel void add_broadcast_pe(
                __global const FP_TYPE* input,  // Input tensor (B, S, E)
                __global const FP_TYPE* pe,     // Positional Encoding tensor (S, E) - Slice matching S
                __global FP_TYPE* output, // Output tensor (B, S, E)
                const int S, const int E        // Dimensions S and E (B is implicit from GWS dim 2)
                ) {
    // Use 3D global IDs mapping to (e, s, b) for potentially better memory access coalescence on 'e'
    int e = get_global_id(0); // Index along dimension E (0 to E-1)
    int s = get_global_id(1); // Index along dimension S (0 to S-1)
    int b = get_global_id(2); // Index along dimension B (0 to B-1)

    // Bounds check for s and e. 'b' is implicitly checked by GWS dim 2.
    if (s < S && e < E) {
       // Calculate linear index for input/output tensors (shape B, S, E)
       size_t idx_bse = (size_t)b * S * E + (size_t)s * E + e;
       // Calculate linear index for the PE tensor (shape S, E)
       // PE is indexed only by s and e, broadcast across b
       size_t idx_pe = (size_t)s * E + e;

       // Perform the broadcast addition
       output[idx_bse] = input[idx_bse] + pe[idx_pe];
    }
}
)CLC";
// ----------------------------------------------------------------------------------

// --- Helper Function Implementations ---

// Function to get OpenCL error string
const char* clGetErrorString(cl_int error) {
    // Static map of error codes to strings (standard OpenCL errors)
    static const char *errStr[] = {
        "CL_SUCCESS", "CL_DEVICE_NOT_FOUND", "CL_DEVICE_NOT_AVAILABLE", "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE", "CL_OUT_OF_RESOURCES", "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE", "CL_MEM_COPY_OVERLAP", "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED", "CL_BUILD_PROGRAM_FAILURE", "CL_MAP_FAILURE",
        // Placeholder for codes -13 to -30 (inclusive)
        "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
        "CL_INVALID_VALUE", "CL_INVALID_DEVICE_TYPE", "CL_INVALID_PLATFORM", "CL_INVALID_DEVICE", "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES", "CL_INVALID_COMMAND_QUEUE", "CL_INVALID_HOST_PTR", "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", "CL_INVALID_IMAGE_SIZE", "CL_INVALID_SAMPLER", "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS", "CL_INVALID_PROGRAM", "CL_INVALID_PROGRAM_EXECUTABLE", "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION", "CL_INVALID_KERNEL", "CL_INVALID_ARG_INDEX", "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE", "CL_INVALID_KERNEL_ARGS", "CL_INVALID_WORK_DIMENSION", "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE", "CL_INVALID_GLOBAL_OFFSET", "CL_INVALID_EVENT_WAIT_LIST", "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION", "CL_INVALID_GL_OBJECT", "CL_INVALID_BUFFER_SIZE", "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
        // Placeholder for codes -64 to -68 (inclusive)
         "", "", "", "", "",
        "CL_INVALID_PROPERTY", // -69
        "CL_INVALID_IMAGE_DESCRIPTOR", // -70
        "CL_INVALID_COMPILER_OPTIONS", // -71
        "CL_INVALID_LINKER_OPTIONS", // -72
        "CL_INVALID_DEVICE_PARTITION_COUNT" // -73
        // Add more specific error codes if needed for newer OpenCL versions
    };
    const int errCount = sizeof(errStr) / sizeof(errStr[0]);
    const int index = -error; // Error codes are negative integers

    // Check if the index is within the bounds of our static map
    if (index >= 0 && index < errCount) {
        const char* err = errStr[index];
        // Return the string if it's valid and not empty
        if (err && err[0] != '\0') {
             return err;
        }
    }
    // If the error code is unknown or the string is empty, return a generic message
    static char unknown_error[64];
    #ifdef _WIN32
        sprintf_s(unknown_error, sizeof(unknown_error), "Unknown OpenCL error %d", error);
    #else
        snprintf(unknown_error, sizeof(unknown_error), "Unknown OpenCL error %d", error);
    #endif
    return unknown_error;
}

// Function to compile an OpenCL kernel from source string
cl_int compile_opencl_kernel(const char* kernel_source, const char* kernel_name,
                             cl_program* program_out, cl_kernel* kernel_out) {
    cl_int err;
    size_t source_size = strlen(kernel_source);
    if (!context || !device_id) {
        fprintf(stderr, "[C] compile_opencl_kernel: Error - No context or device available.\n");
        return CL_INVALID_CONTEXT; // Or another appropriate error code
    }

    // Create OpenCL program object from source
    *program_out = clCreateProgramWithSource(context, 1, &kernel_source, &source_size, &err);
    if (!*program_out || err != CL_SUCCESS) {
        fprintf(stderr, "[C] compile_opencl_kernel: clCreateProgramWithSource failed for '%s': %s (%d)\n",
                kernel_name, clGetErrorString(err), err);
        return err;
    }

    // --- Build Options ---
    // Define preprocessor macros for the kernel compilation
    char build_options[512];
    snprintf(build_options, sizeof(build_options),
             "-cl-std=CL1.2 -Werror -D FP_TYPE=%s %s %s -DFP_TYPE_SIZE=%zu", // <-- OHNE -Wno-#warnings
             KERNEL_FP_TYPE_STR,
             has_fp64_support ? "-D CL_HAS_FP64" : "",
             has_atomics_support ? "-D CL_HAS_ATOMICS" : "",
             sizeof(FP_TYPE)
             );
    // --- End Build Options ---

    // Build (compile and link) the program for the selected device
    err = clBuildProgram(*program_out, 1, &device_id, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[C] compile_opencl_kernel: clBuildProgram failed for '%s' with options '%s': %s (%d)\n",
                kernel_name, build_options, clGetErrorString(err), err);
        // Get and print the build log if compilation fails
        size_t log_size = 0;
        clGetProgramBuildInfo(*program_out, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (log_size > 1) {
            char *log = (char *)malloc(log_size);
            if (log) {
                clGetProgramBuildInfo(*program_out, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                fprintf(stderr, "--- OpenCL Build Log (%s) ---\n%s\n------\n", kernel_name, log);
                free(log);
            } else { fprintf(stderr, "[C] compile_opencl_kernel: Failed to allocate memory for build log.\n"); }
        } else { fprintf(stderr, "[C] compile_opencl_kernel: Build log size is 0 or 1.\n");}
        // Clean up the failed program object
        clReleaseProgram(*program_out); *program_out = NULL;
        return err; // Return the build error code
    }

    // Create the kernel object from the successfully built program
    *kernel_out = clCreateKernel(*program_out, kernel_name, &err);
    if (!*kernel_out || err != CL_SUCCESS) {
        fprintf(stderr, "[C] compile_opencl_kernel: clCreateKernel failed for '%s': %s (%d)\n",
                kernel_name, clGetErrorString(err), err);
        clReleaseProgram(*program_out); *program_out = NULL; // Clean up program object
        return err;
    }

    printf("[C] compile_opencl_kernel: Successfully compiled kernel '%s'.\n", kernel_name);
    return CL_SUCCESS; // Kernel compiled successfully
}

// Function to release all OpenCL resources
void shutdown_driver() {
    printf("[C] shutdown_driver: Starting Cleanup...\n");

    // Release Kernels
    #define RELEASE_KERNEL(k) if (k) { clReleaseKernel(k); k = NULL; }
    RELEASE_KERNEL(matmul_kernel);
    RELEASE_KERNEL(softmax_kernel);
    RELEASE_KERNEL(gelu_kernel);
    RELEASE_KERNEL(add_kernel);
    RELEASE_KERNEL(mul_kernel);
    RELEASE_KERNEL(layernorm_kernel);
    RELEASE_KERNEL(transpose_kernel);
    RELEASE_KERNEL(gelu_backward_kernel);
    RELEASE_KERNEL(matmul_backward_da_kernel);
    RELEASE_KERNEL(matmul_backward_db_kernel);
    RELEASE_KERNEL(layernorm_backward_kernel);
    RELEASE_KERNEL(adam_kernel);
    RELEASE_KERNEL(softmax_backward_kernel);
    RELEASE_KERNEL(mul_backward_kernel);
    RELEASE_KERNEL(transpose_backward_kernel);
    RELEASE_KERNEL(embedding_lookup_kernel);
    RELEASE_KERNEL(embedding_backward_kernel);
    RELEASE_KERNEL(reduce_sum_kernel);
    RELEASE_KERNEL(broadcast_add_kernel);
    RELEASE_KERNEL(transpose_batched_kernel);
    RELEASE_KERNEL(transpose_12_batched_kernel);
    RELEASE_KERNEL(matmul_batched_kernel);
    RELEASE_KERNEL(matmul_batched_backward_da_kernel);
    RELEASE_KERNEL(matmul_batched_backward_db_kernel);
    RELEASE_KERNEL(log_softmax_kernel); // Release new CE kernel
    RELEASE_KERNEL(cross_entropy_kernel); // Release new CE kernel
    RELEASE_KERNEL(add_broadcast_pe_kernel); // Release new PE kernel
    #undef RELEASE_KERNEL

    // Release Programs
    #define RELEASE_PROGRAM(p) if (p) { clReleaseProgram(p); p = NULL; }
    RELEASE_PROGRAM(matmul_program);
    RELEASE_PROGRAM(softmax_program);
    RELEASE_PROGRAM(gelu_program);
    RELEASE_PROGRAM(add_program);
    RELEASE_PROGRAM(mul_program);
    RELEASE_PROGRAM(layernorm_program);
    RELEASE_PROGRAM(transpose_program);
    RELEASE_PROGRAM(gelu_backward_program);
    RELEASE_PROGRAM(matmul_backward_da_program);
    RELEASE_PROGRAM(matmul_backward_db_program);
    RELEASE_PROGRAM(layernorm_backward_program);
    RELEASE_PROGRAM(adam_program);
    RELEASE_PROGRAM(softmax_backward_program);
    RELEASE_PROGRAM(mul_backward_program);
    RELEASE_PROGRAM(transpose_backward_program);
    RELEASE_PROGRAM(embedding_lookup_program);
    RELEASE_PROGRAM(embedding_backward_program);
    RELEASE_PROGRAM(reduce_sum_program);
    RELEASE_PROGRAM(broadcast_add_program);
    RELEASE_PROGRAM(transpose_batched_program);
    RELEASE_PROGRAM(transpose_12_batched_program);
    RELEASE_PROGRAM(matmul_batched_program);
    RELEASE_PROGRAM(matmul_batched_backward_da_program);
    RELEASE_PROGRAM(matmul_batched_backward_db_program);
    RELEASE_PROGRAM(log_softmax_program); // Release new CE program
    RELEASE_PROGRAM(cross_entropy_program); // Release new CE program
    RELEASE_PROGRAM(add_broadcast_pe_program); // Release new PE program
    #undef RELEASE_PROGRAM

    // Release Command Queue and Context
    if (queue) {
        clFinish(queue); // Ensure all commands are finished before releasing
        clReleaseCommandQueue(queue);
        queue = NULL;
    }
    if (context) {
        clReleaseContext(context);
        context = NULL;
    }

    // Reset global pointers and flags
    device_id = NULL;
    platform_id = NULL;
    has_fp64_support = 0;
    has_atomics_support = 0;
    printf("[C] shutdown_driver: Cleanup finished.\n");
}

// Function to get the number of compute units on the device
unsigned int get_compute_unit_count(int gpu_index) {
    if (!device_id) {
        // fprintf(stderr, "[C] get_compute_unit_count: No device selected.\n"); // Optional warning
        return 0; // Return 0 if no device is initialized
    }
    cl_uint cu_count = 0;
    cl_int err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &cu_count, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[C] get_compute_unit_count: clGetDeviceInfo failed for CL_DEVICE_MAX_COMPUTE_UNITS: %s (%d)\n", clGetErrorString(err), err);
        return 0; // Return 0 on error
    }
    return (unsigned int)cu_count;
}


// --- Exported Functions ---

DLLEXPORT int initialize_gpu(int gpu_index) {
    cl_int err;
    // --- Find Platform ---
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) { fprintf(stderr, "[C] initialize_gpu: No OpenCL platforms found.\n"); return 0; }
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) { fprintf(stderr, "[C] initialize_gpu: Failed to allocate memory for platforms.\n"); return 0; }
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "[C] initialize_gpu: Error getting platform IDs: %s (%d)\n", clGetErrorString(err), err); free(platforms); return 0; }
    platform_id = platforms[0]; // Use the first platform found
    free(platforms); // Free the allocated platform list
    char platformName[1024] = {0}; clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(platformName)-1, platformName, NULL);
    printf("[C] initialize_gpu: Using platform: %s\n", platformName);

    // --- Find Device ---
    cl_uint num_devices;
    cl_device_type selected_device_type = CL_DEVICE_TYPE_GPU; // Prefer GPU
    err = clGetDeviceIDs(platform_id, selected_device_type, 0, NULL, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        fprintf(stderr, "[C] initialize_gpu: No GPU devices found. Trying CL_DEVICE_TYPE_ALL...\n");
        selected_device_type = CL_DEVICE_TYPE_ALL; // Fallback to any device type
        err = clGetDeviceIDs(platform_id, selected_device_type, 0, NULL, &num_devices);
        if(err != CL_SUCCESS || num_devices == 0) { fprintf(stderr, "[C] initialize_gpu: No OpenCL devices found at all.\n"); return 0; }
    }
    // Validate gpu_index
    if (gpu_index < 0 || gpu_index >= (int)num_devices) {
        fprintf(stderr, "[C] initialize_gpu: gpu_index %d out of range [0, %d). Using index 0.\n", gpu_index, num_devices);
        gpu_index = 0; // Default to the first device if index is invalid
    }
    cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
    if (!devices) { fprintf(stderr, "[C] initialize_gpu: Failed to allocate memory for devices.\n"); return 0; }
    err = clGetDeviceIDs(platform_id, selected_device_type, num_devices, devices, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "[C] initialize_gpu: Failed to get device IDs: %s (%d)\n", clGetErrorString(err), err); free(devices); return 0; }
    device_id = devices[gpu_index]; // Select the requested device
    free(devices); // Free the allocated device list
    char deviceName[1024] = {0}; clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName)-1, deviceName, NULL);
    printf("[C] initialize_gpu: Using device index %d: %s\n", gpu_index, deviceName);

    // --- Check Device Capabilities ---
    // Check FP64 support
    cl_device_fp_config fp_config;
    err = clGetDeviceInfo(device_id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp_config), &fp_config, NULL);
    if (err == CL_SUCCESS && (fp_config & (CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM))) {
        has_fp64_support = 1;
    } else { has_fp64_support = 0; }
    printf("[C] initialize_gpu: FP64 Support: %s\n", has_fp64_support ? "Yes" : "No");

    // Check Atomics support (specifically for global int compare-and-swap needed for FP32 atomics)
    cl_device_atomic_capabilities atom_caps = 0;
    err = clGetDeviceInfo(device_id, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES, sizeof(atom_caps), &atom_caps, NULL);
    // Check for basic device-scope atomic support first
    if (err == CL_SUCCESS && (atom_caps & (CL_DEVICE_ATOMIC_ORDER_RELAXED | CL_DEVICE_ATOMIC_SCOPE_DEVICE))) {
        // A more robust check might involve querying CL_DEVICE_EXTENSIONS string for "cl_khr_global_int32_base_atomics"
        // Or checking CL_DEVICE_ATOMIC_ORDER_ACQ_REL | CL_DEVICE_ATOMIC_SCOPE_DEVICE based on OpenCL spec.
        // For simplicity, assume basic device scope atomics imply necessary support for int cmpxchg.
        has_atomics_support = 1;
    } else { has_atomics_support = 0; }
    if (!has_atomics_support) {
        fprintf(stderr, "[C] WARN: Sufficient atomics support (global int cmpxchg) potentially UNSUPPORTED or not detected. Embedding backward might fail or cause race conditions!\n");
    }
    printf("[C] initialize_gpu: Atomics Support (sufficient for FP32): %s\n", has_atomics_support ? "Yes" : "No");

    // --- Create Context ---
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) { fprintf(stderr, "[C] initialize_gpu: clCreateContext failed: %s (%d)\n", clGetErrorString(err), err); shutdown_driver(); return 0; }

    // --- Create Command Queue ---
    #ifdef CL_VERSION_2_0 // Use newer API if OpenCL 2.0+
        // Example: Enable profiling if needed:
        // cl_queue_properties queue_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        cl_queue_properties queue_props[] = {0}; // No special properties
        queue = clCreateCommandQueueWithProperties(context, device_id, queue_props, &err);
    #else // Use deprecated API for OpenCL 1.x
        #pragma GCC diagnostic push // Suppress deprecation warnings (GCC/Clang)
        #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        queue = clCreateCommandQueue(context, device_id, 0, &err); // 0 for default properties
        #pragma GCC diagnostic pop
    #endif
    if (!queue || err != CL_SUCCESS) { fprintf(stderr, "[C] initialize_gpu: clCreateCommandQueue failed: %s (%d)\n", clGetErrorString(err), err); shutdown_driver(); return 0; }

    // --- Compile All Kernels ---
    printf("[C] initialize_gpu: Compiling ALL kernels...\n");
    cl_int compile_err;
    #define COMPILE_KERNEL(src, name, prog, kern) \
        printf("[C] Compiling kernel: %s...\n", name); \
        compile_err = compile_opencl_kernel(src, name, prog, kern); \
        if (compile_err != CL_SUCCESS) { \
            fprintf(stderr, "[C] FAILED to compile kernel '%s'. Shutting down.\n", name); \
            shutdown_driver(); \
            return 0; \
        }

    COMPILE_KERNEL(matmul_kernel_src, "matrix_multiply", &matmul_program, &matmul_kernel);
    COMPILE_KERNEL(softmax_kernel_src, "softmax_rowwise", &softmax_program, &softmax_kernel);
    COMPILE_KERNEL(gelu_kernel_src, "gelu_elementwise", &gelu_program, &gelu_kernel);
    COMPILE_KERNEL(add_kernel_src, "add_elementwise", &add_program, &add_kernel);
    COMPILE_KERNEL(mul_kernel_src, "mul_elementwise", &mul_program, &mul_kernel);
    COMPILE_KERNEL(layernorm_kernel_src, "layer_norm", &layernorm_program, &layernorm_kernel);
    COMPILE_KERNEL(transpose_kernel_src, "transpose", &transpose_program, &transpose_kernel);
    COMPILE_KERNEL(gelu_backward_kernel_src, "gelu_backward_elementwise", &gelu_backward_program, &gelu_backward_kernel);
    COMPILE_KERNEL(matmul_backward_dA_kernel_src, "matmul_backward_da", &matmul_backward_da_program, &matmul_backward_da_kernel);
    COMPILE_KERNEL(matmul_backward_dB_kernel_src, "matmul_backward_db", &matmul_backward_db_program, &matmul_backward_db_kernel);
    COMPILE_KERNEL(layernorm_backward_kernel_src, "layer_norm_backward", &layernorm_backward_program, &layernorm_backward_kernel);
    COMPILE_KERNEL(adam_kernel_src, "adam_update", &adam_program, &adam_kernel);
    COMPILE_KERNEL(softmax_backward_kernel_src, "softmax_backward", &softmax_backward_program, &softmax_backward_kernel);
    COMPILE_KERNEL(mul_backward_kernel_src, "mul_backward", &mul_backward_program, &mul_backward_kernel);
    COMPILE_KERNEL(transpose_backward_kernel_src, "transpose_backward", &transpose_backward_program, &transpose_backward_kernel);
    COMPILE_KERNEL(embedding_lookup_kernel_src, "embedding_lookup", &embedding_lookup_program, &embedding_lookup_kernel);
    COMPILE_KERNEL(embedding_backward_kernel_src, "embedding_backward_scatter_add", &embedding_backward_program, &embedding_backward_kernel);
    COMPILE_KERNEL(reduce_sum_kernel_src, "reduce_sum_axis01", &reduce_sum_program, &reduce_sum_kernel);
    COMPILE_KERNEL(broadcast_add_kernel_src, "broadcast_add_bias", &broadcast_add_program, &broadcast_add_kernel);
    COMPILE_KERNEL(transpose_batched_kernel_src, "transpose_batched_last_two", &transpose_batched_program, &transpose_batched_kernel);
    COMPILE_KERNEL(transpose_12_batched_kernel_src, "transpose_12_batched", &transpose_12_batched_program, &transpose_12_batched_kernel);
    COMPILE_KERNEL(matmul_batched_kernel_src, "matmul_batched", &matmul_batched_program, &matmul_batched_kernel);
    COMPILE_KERNEL(matmul_batched_backward_dA_kernel_src, "matmul_batched_backward_da", &matmul_batched_backward_da_program, &matmul_batched_backward_da_kernel);
    COMPILE_KERNEL(matmul_batched_backward_dB_kernel_src, "matmul_batched_backward_db", &matmul_batched_backward_db_program, &matmul_batched_backward_db_kernel);
    // --- NEU: Compile Cross Entropy & PE Kernels ---
    COMPILE_KERNEL(log_softmax_stable_kernel_src, "log_softmax_stable_rowwise", &log_softmax_program, &log_softmax_kernel);
    COMPILE_KERNEL(cross_entropy_loss_grad_kernel_src, "cross_entropy_loss_grad", &cross_entropy_program, &cross_entropy_kernel);
    COMPILE_KERNEL(add_broadcast_pe_kernel_src, "add_broadcast_pe", &add_broadcast_pe_program, &add_broadcast_pe_kernel);
    // ----------------------------------------
    #undef COMPILE_KERNEL

    printf("[C] initialize_gpu: Initialization OK for GPU %d.\n", gpu_index);
    return 1; // Success
}

DLLEXPORT void *allocate_gpu_memory(int gpu_index, size_t size) {
    cl_int err;
    if (!context) { fprintf(stderr, "[C] allocate_gpu_memory: No context available.\n"); return NULL; }
    if (size == 0) { fprintf(stderr, "[C] allocate_gpu_memory: Warning - Attempted to allocate 0 bytes.\n"); return NULL; } // Disallow 0-byte allocation

    // CL_MEM_READ_WRITE is standard, allows kernel read/write.
    // Consider CL_MEM_HOST_NO_ACCESS if host only writes initially and reads finally (potential perf gain).
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (!buffer || err != CL_SUCCESS) {
        fprintf(stderr, "[C] allocate_gpu_memory: Error clCreateBuffer: %s (%d) for size %zu bytes.\n", clGetErrorString(err), err, size);
        return NULL; // Return NULL on failure
    }
    // printf("[C] allocate_gpu_memory: Allocated buffer %p with size %zu\n", (void*)buffer, size); // Debug
    return (void*)buffer; // Return the opaque buffer handle
}

DLLEXPORT void free_gpu_memory(int gpu_index, void* buffer_handle) {
    if (!buffer_handle) { return; } // Nothing to free if handle is NULL
    cl_mem buffer = (cl_mem)buffer_handle; // Cast handle back to cl_mem

    if (!context) {
        // This might happen if called after shutdown or before successful init
        fprintf(stderr, "[C] free_gpu_memory: No context available. Cannot free buffer %p.\n", buffer_handle);
        return;
    }
    // printf("[C] free_gpu_memory: Releasing buffer %p\n", buffer_handle); // Debug
    cl_int err = clReleaseMemObject(buffer);
    if (err != CL_SUCCESS) {
        // CL_INVALID_MEM_OBJECT often means it was already freed (or never valid).
        // Avoid crashing on double free attempts, just issue a warning.
        if (err == CL_INVALID_MEM_OBJECT) {
             fprintf(stderr, "[C] free_gpu_memory: Warning - clReleaseMemObject returned CL_INVALID_MEM_OBJECT (possibly already freed?): %p\n", buffer_handle);
        } else {
            // Report other errors more seriously
            fprintf(stderr, "[C] free_gpu_memory: Error clReleaseMemObject for buffer %p: %s (%d)\n", buffer_handle, clGetErrorString(err), err);
        }
    }
    // Note: The handle `buffer_handle` in the calling code (Python) should be considered invalid after this.
}

DLLEXPORT int write_host_to_gpu_blocking(int gpu_index, void* gpu_buffer_handle, size_t offset, size_t size, const void* host_source_ptr) {
    // --- Input Validation ---
    if (!gpu_buffer_handle) { fprintf(stderr, "[C] write_host_to_gpu_blocking: Invalid GPU buffer handle (NULL).\n"); return 0; }
    if (size > 0 && !host_source_ptr) { fprintf(stderr, "[C] write_host_to_gpu_blocking: Host source pointer is NULL but size > 0 (%zu).\n", size); return 0; }
    if (!queue) { fprintf(stderr, "[C] write_host_to_gpu_blocking: Command queue is NULL.\n"); return 0; }
    if (size == 0) { return 1; } // Nothing to write, operation successful
    // --- End Validation ---

    cl_mem gpu_buffer = (cl_mem)gpu_buffer_handle;
    // CL_TRUE makes this call blocking - it waits for the write operation to complete.
    cl_int err = clEnqueueWriteBuffer(queue, gpu_buffer, CL_TRUE, offset, size, host_source_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[C] write_host_to_gpu_blocking: Error clEnqueueWriteBuffer: %s (%d) [offset=%zu, size=%zu]\n", clGetErrorString(err), err, offset, size);
        return 0; // Failure
    }
    return 1; // Success
}

DLLEXPORT int read_gpu_to_host_blocking(int gpu_index, void* gpu_buffer_handle, size_t offset, size_t size, void* host_destination_ptr) {
    // --- Input Validation ---
     if (!gpu_buffer_handle) { fprintf(stderr, "[C] read_gpu_to_host_blocking: Invalid GPU buffer handle (NULL).\n"); return 0; }
     if (size > 0 && !host_destination_ptr) { fprintf(stderr, "[C] read_gpu_to_host_blocking: Host destination pointer is NULL but size > 0 (%zu).\n", size); return 0; }
     if (!queue) { fprintf(stderr, "[C] read_gpu_to_host_blocking: Command queue is NULL.\n"); return 0; }
     if (size == 0) { return 1; } // Nothing to read, operation successful
    // --- End Validation ---

    cl_mem gpu_buffer = (cl_mem)gpu_buffer_handle;
    // CL_TRUE makes this call blocking - it waits for the read operation to complete.
    cl_int err = clEnqueueReadBuffer(queue, gpu_buffer, CL_TRUE, offset, size, host_destination_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[C] read_gpu_to_host_blocking: Error clEnqueueReadBuffer: %s (%d) [offset=%zu, size=%zu]\n", clGetErrorString(err), err, offset, size);
        return 0; // Failure
    }
    return 1; // Success
}

// --- Command Data Structures (Defined based on kernel arguments) ---
typedef struct { void* buffer_a; void* buffer_b; void* buffer_c; int B; int M; int N; int K; } BMMCommandData;
typedef struct { void* buffer_input; void* buffer_output; int num_rows; int row_size; } SoftmaxCommandData;
typedef struct { void* buffer_input; void* buffer_output; int num_elements; } GeluCommandData;
typedef struct { void* buffer_a; void* buffer_b; void* buffer_c; int num_elements; } AddCommandData;
typedef struct { void* buffer_a; void* buffer_b; void* buffer_c; int num_elements; } MulCommandData;
typedef struct { void* buffer_input; void* buffer_output; int num_rows; int row_size; float eps; } LayerNormCommandData;
typedef struct { void* src_buffer; void* dst_buffer; size_t size; } CloneCommandData;
typedef struct { void* buffer_input; void* buffer_output; int rows; int cols; } TransposeCommandData; // Basic 2D
typedef struct { void* buffer_input; void* buffer_grad_output; void* buffer_grad_input; int num_elements; } GeluBackwardCommandData;
typedef struct { void* buffer_a; void* buffer_b; void* buffer_dc; void* buffer_da; void* buffer_db; int B, M, N, K; } MatMulBackwardData; // For standard matmul backward
typedef struct { void* buffer_dy; void* buffer_x; void* buffer_dx; int num_rows; int row_size; float eps; } LayerNormBackwardCommandData;
typedef struct { void* param_buffer; void* grad_buffer; void* m_buffer; void* v_buffer; int num_elements; int t_step; float lr,beta1,beta2,eps,weight_decay,beta1_t,beta2_t; } AdamCommandData; // Renamed 't' to 't_step' to avoid potential conflict
typedef struct { void* buffer_dy; void* buffer_y; void* buffer_dx; int num_rows; int row_size; } SoftmaxBackwardCommandData;
typedef struct { void* buffer_dC; void* buffer_A; void* buffer_B; void* buffer_dA; void* buffer_dB; int num_elements; } MulBackwardCommandData;
typedef struct { void* buffer_dC; void* buffer_dA; int rows_A; int cols_A; } TransposeBackwardCommandData; // Basic 2D backward
typedef struct { void* idx; void* w; void* o; int b, s, d, v; } EmbeddingLookupCommandData;
typedef struct { void* d_o; void* idx; void* d_w; int b, s, d, v; } EmbeddingBackwardCommandData;
typedef struct { void* in; void* out; int B, M, N; size_t lmem_sz; } ReduceSumCommandData; // For Bias grad
typedef struct { void* a; void* b; void* c; int B, M, N; } BroadcastAddCommandData; // General bias add
typedef struct { void* in; void* out; int B_flat, d1, d2; } TransposeBatchedCommandData; // For last two dims
typedef struct { void* in; void* out; int B, D1, D2, D3; } Transpose12BatchedCommandData; // For transpose(1,2) 4D
typedef struct { void* buffer_a; void* buffer_b; void* buffer_c; int B; int M; int N; int K; } BMMBatchedCommandData; // For batched 3D @ 3D
typedef struct { void* buffer_a; void* buffer_b; void* buffer_dc; void* buffer_da; void* buffer_db; int B, M, N, K; } BMMBatchedBackwardData; // For batched 3D @ 3D backward
// --- NEU: Structs for Cross Entropy & PE Add ---
typedef struct { void* input_logits; void* output_log_probs; int B_S_rows; int V_cols; } LogSoftmaxStableCommandData; // Combined B*S
typedef struct { void* log_probs; void* target_indices; void* grad_input; void* loss_per_sample; int B_S_rows; int V_cols; } CrossEntropyLossGradCommandData; // Combined B*S
typedef struct { void* input; void* pe_slice; void* output; int B; int S; int E; } AddBroadcastPECommandData;
// ------------------------------------

// --- Kernel Submission Logic ---
int submit_kernel_command(int gpu_index, GPUCommand command, void *data) {
    cl_int err = CL_SUCCESS;
    if (!queue) { fprintf(stderr, "[C] submit_kernel_command: Invalid command queue.\n"); return 0; }

    // Macro to check OpenCL errors after calls
    #define CHECK_CL_ERR(call, kernel_name_str) \
        err = (call); \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "[C] OpenCL Error (%s): %s (%d) during '%s' in %s line %d\n", \
                    kernel_name_str, clGetErrorString(err), err, #call, __FILE__, __LINE__); \
            return 0; \
        }

    switch(command) {
        // --- Standard Matmul (Handles 3D@2D broadcast) ---
        case COMMAND_MATRIX_MULTIPLY: {
            BMMCommandData* cmd = (BMMCommandData*)data;
            if (!matmul_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_b || !cmd->buffer_c) { fprintf(stderr, "[C] Submit BMM Fwd (Standard): Invalid args.\n"); return 0; }
            cl_mem a = (cl_mem)cmd->buffer_a, b = (cl_mem)cmd->buffer_b, c = (cl_mem)cmd->buffer_c;
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 0, sizeof(cl_mem), &a), "BMM Fwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 1, sizeof(cl_mem), &b), "BMM Fwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 2, sizeof(cl_mem), &c), "BMM Fwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 3, sizeof(cl_int), &cmd->B), "BMM Fwd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 4, sizeof(cl_int), &cmd->M), "BMM Fwd Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 5, sizeof(cl_int), &cmd->N), "BMM Fwd Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 6, sizeof(cl_int), &cmd->K), "BMM Fwd Arg 6");
            size_t gws[3] = { (size_t)cmd->N, (size_t)cmd->M, (size_t)cmd->B }; // GWS maps to output C[B, M, N] -> (N, M, B)
            if (gws[0] == 0 || gws[1] == 0 || gws[2] == 0) return 1; // Zero elements to compute
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "BMM Fwd Enqueue");
            return 1;
        }
        // --- Softmax ---
        case COMMAND_SOFTMAX_ROWWISE: {
            SoftmaxCommandData* cmd = (SoftmaxCommandData*)data;
            if (!softmax_kernel || !cmd || !cmd->buffer_input || !cmd->buffer_output) { fprintf(stderr, "[C] Submit Softmax Fwd: Invalid args.\n"); return 0; }
            cl_mem in = (cl_mem)cmd->buffer_input, out = (cl_mem)cmd->buffer_output;
            CHECK_CL_ERR(clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &in), "Softmax Fwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &out), "Softmax Fwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(softmax_kernel, 2, sizeof(cl_int), &cmd->num_rows), "Softmax Fwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(softmax_kernel, 3, sizeof(cl_int), &cmd->row_size), "Softmax Fwd Arg 3");
            size_t gws[1] = { (size_t)cmd->num_rows }; // 1D global work size, one item per row
            if (gws[0] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, softmax_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Softmax Fwd Enqueue");
            return 1;
        }
        // --- GELU ---
         case COMMAND_GELU_ELEMENTWISE: {
             GeluCommandData* cmd = (GeluCommandData*)data;
             if (!gelu_kernel || !cmd || !cmd->buffer_input || !cmd->buffer_output) { fprintf(stderr, "[C] Submit GELU Fwd: Invalid args.\n"); return 0; }
             cl_mem in = (cl_mem)cmd->buffer_input, out = (cl_mem)cmd->buffer_output;
             CHECK_CL_ERR(clSetKernelArg(gelu_kernel, 0, sizeof(cl_mem), &in), "GELU Fwd Arg 0");
             CHECK_CL_ERR(clSetKernelArg(gelu_kernel, 1, sizeof(cl_mem), &out), "GELU Fwd Arg 1");
             CHECK_CL_ERR(clSetKernelArg(gelu_kernel, 2, sizeof(cl_int), &cmd->num_elements), "GELU Fwd Arg 2");
             size_t gws[1] = { (size_t)cmd->num_elements }; // 1D GWS, one item per element
             if (gws[0] == 0) return 1;
             CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, gelu_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "GELU Fwd Enqueue");
            return 1;
        }
        // --- Add (Elementwise) ---
        case COMMAND_ADD_ELEMENTWISE: {
             AddCommandData* cmd = (AddCommandData*)data;
             if (!add_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_b || !cmd->buffer_c) { fprintf(stderr, "[C] Submit Add Fwd: Invalid args.\n"); return 0; }
             cl_mem a = (cl_mem)cmd->buffer_a, b = (cl_mem)cmd->buffer_b, c = (cl_mem)cmd->buffer_c;
             CHECK_CL_ERR(clSetKernelArg(add_kernel, 0, sizeof(cl_mem), &a), "Add Fwd Arg 0");
             CHECK_CL_ERR(clSetKernelArg(add_kernel, 1, sizeof(cl_mem), &b), "Add Fwd Arg 1");
             CHECK_CL_ERR(clSetKernelArg(add_kernel, 2, sizeof(cl_mem), &c), "Add Fwd Arg 2");
             CHECK_CL_ERR(clSetKernelArg(add_kernel, 3, sizeof(cl_int), &cmd->num_elements), "Add Fwd Arg 3");
             size_t gws[1] = { (size_t)cmd->num_elements }; // 1D GWS
             if (gws[0] == 0) return 1;
             CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, add_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Add Fwd Enqueue");
            return 1;
        }
        // --- Multiply (Elementwise) ---
        case COMMAND_MUL_ELEMENTWISE: {
             MulCommandData* cmd = (MulCommandData*)data;
             if (!mul_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_b || !cmd->buffer_c) { fprintf(stderr, "[C] Submit Mul Fwd: Invalid args.\n"); return 0; }
             cl_mem a = (cl_mem)cmd->buffer_a, b = (cl_mem)cmd->buffer_b, c = (cl_mem)cmd->buffer_c;
             CHECK_CL_ERR(clSetKernelArg(mul_kernel, 0, sizeof(cl_mem), &a), "Mul Fwd Arg 0");
             CHECK_CL_ERR(clSetKernelArg(mul_kernel, 1, sizeof(cl_mem), &b), "Mul Fwd Arg 1");
             CHECK_CL_ERR(clSetKernelArg(mul_kernel, 2, sizeof(cl_mem), &c), "Mul Fwd Arg 2");
             CHECK_CL_ERR(clSetKernelArg(mul_kernel, 3, sizeof(cl_int), &cmd->num_elements), "Mul Fwd Arg 3");
             size_t gws[1] = { (size_t)cmd->num_elements }; // 1D GWS
             if (gws[0] == 0) return 1;
             CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, mul_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Mul Fwd Enqueue");
            return 1;
        }
        // --- Layer Normalization ---
        case COMMAND_LAYER_NORM: {
             LayerNormCommandData* cmd = (LayerNormCommandData*)data;
             if (!layernorm_kernel || !cmd || !cmd->buffer_input || !cmd->buffer_output) { fprintf(stderr, "[C] Submit LayerNorm Fwd: Invalid args.\n"); return 0; }
             cl_mem in = (cl_mem)cmd->buffer_input, out = (cl_mem)cmd->buffer_output;
             CHECK_CL_ERR(clSetKernelArg(layernorm_kernel, 0, sizeof(cl_mem), &in), "LayerNorm Fwd Arg 0");
             CHECK_CL_ERR(clSetKernelArg(layernorm_kernel, 1, sizeof(cl_mem), &out), "LayerNorm Fwd Arg 1");
             CHECK_CL_ERR(clSetKernelArg(layernorm_kernel, 2, sizeof(cl_int), &cmd->num_rows), "LayerNorm Fwd Arg 2");
             CHECK_CL_ERR(clSetKernelArg(layernorm_kernel, 3, sizeof(cl_int), &cmd->row_size), "LayerNorm Fwd Arg 3");
             CHECK_CL_ERR(clSetKernelArg(layernorm_kernel, 4, sizeof(cl_float), &cmd->eps), "LayerNorm Fwd Arg 4");
             size_t gws[1] = { (size_t)cmd->num_rows }; // 1D GWS, one item per row
             if (gws[0] == 0) return 1;
             CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, layernorm_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "LayerNorm Fwd Enqueue");
            return 1;
        }
        // --- Clone (GPU Buffer Copy) ---
        case COMMAND_CLONE: {
            CloneCommandData* cmd = (CloneCommandData*)data;
            if (!cmd || !cmd->src_buffer || !cmd->dst_buffer) { fprintf(stderr, "[C] Submit Clone: Invalid args.\n"); return 0; }
            if (cmd->size == 0) { return 1; } // Nothing to copy
            cl_mem src = (cl_mem)cmd->src_buffer; cl_mem dst = (cl_mem)cmd->dst_buffer;
            // Use efficient buffer copy command
            CHECK_CL_ERR(clEnqueueCopyBuffer(queue, src, dst, 0, 0, cmd->size, 0, NULL, NULL), "Clone Enqueue (CopyBuffer)");
            return 1;
        }
        // --- Transpose (Basic 2D) ---
        case COMMAND_TRANSPOSE: {
            TransposeCommandData* cmd = (TransposeCommandData*)data;
            if (!transpose_kernel || !cmd || !cmd->buffer_input || !cmd->buffer_output) { fprintf(stderr, "[C] Submit Transpose Fwd (2D): Invalid args.\n"); return 0; }
            cl_mem in = (cl_mem)cmd->buffer_input, out = (cl_mem)cmd->buffer_output;
            CHECK_CL_ERR(clSetKernelArg(transpose_kernel, 0, sizeof(cl_mem), &in), "Transpose Fwd (2D) Arg 0");
            CHECK_CL_ERR(clSetKernelArg(transpose_kernel, 1, sizeof(cl_mem), &out), "Transpose Fwd (2D) Arg 1");
            CHECK_CL_ERR(clSetKernelArg(transpose_kernel, 2, sizeof(cl_int), &cmd->rows), "Transpose Fwd (2D) Arg 2");
            CHECK_CL_ERR(clSetKernelArg(transpose_kernel, 3, sizeof(cl_int), &cmd->cols), "Transpose Fwd (2D) Arg 3");
            // GWS maps to output dimensions (cols, rows)
            size_t gws[2] = { (size_t)cmd->cols, (size_t)cmd->rows };
            if (gws[0] == 0 || gws[1] == 0) { return 1; }
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, transpose_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "Transpose Fwd (2D) Enqueue");
            return 1;
        }
        // --- GELU Backward ---
        case COMMAND_GELU_BACKWARD_ELEMENTWISE: {
            GeluBackwardCommandData* cmd = (GeluBackwardCommandData*)data;
            if (!gelu_backward_kernel || !cmd || !cmd->buffer_input || !cmd->buffer_grad_output || !cmd->buffer_grad_input) { fprintf(stderr, "[C] Submit GELU Bwd: Invalid args.\n"); return 0; }
            cl_mem input_mem = (cl_mem)cmd->buffer_input; cl_mem grad_output_mem = (cl_mem)cmd->buffer_grad_output; cl_mem grad_input_mem = (cl_mem)cmd->buffer_grad_input;
            CHECK_CL_ERR(clSetKernelArg(gelu_backward_kernel, 0, sizeof(cl_mem), &input_mem), "GELU Bwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(gelu_backward_kernel, 1, sizeof(cl_mem), &grad_output_mem), "GELU Bwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(gelu_backward_kernel, 2, sizeof(cl_mem), &grad_input_mem), "GELU Bwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(gelu_backward_kernel, 3, sizeof(cl_int), &cmd->num_elements), "GELU Bwd Arg 3");
            size_t gws[1] = { (size_t)cmd->num_elements }; // 1D GWS
            if (gws[0] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, gelu_backward_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "GELU Bwd Enqueue");
            return 1;
        }
        // --- Matmul Backward dA (Standard) ---
        case COMMAND_MATMUL_BACKWARD_DA: {
            MatMulBackwardData* cmd = (MatMulBackwardData*)data;
            if (!matmul_backward_da_kernel || !cmd || !cmd->buffer_dc || !cmd->buffer_b || !cmd->buffer_da) { fprintf(stderr, "[C] Submit MatMul Bwd dA (Standard): Invalid args.\n"); return 0; }
            cl_mem dc = (cl_mem)cmd->buffer_dc, b_mem = (cl_mem)cmd->buffer_b, da = (cl_mem)cmd->buffer_da;
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 0, sizeof(cl_mem), &dc), "MatMul dA Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 1, sizeof(cl_mem), &b_mem), "MatMul dA Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 2, sizeof(cl_mem), &da), "MatMul dA Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 3, sizeof(cl_int), &cmd->B), "MatMul dA Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 4, sizeof(cl_int), &cmd->M), "MatMul dA Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 5, sizeof(cl_int), &cmd->N), "MatMul dA Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 6, sizeof(cl_int), &cmd->K), "MatMul dA Arg 6");
            // GWS maps to output dA[B, M, K] -> (K, M, B)
            size_t gws[3] = { (size_t)cmd->K, (size_t)cmd->M, (size_t)cmd->B };
            if (gws[0] == 0 || gws[1] == 0 || gws[2] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_backward_da_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "MatMul dA Enqueue");
            return 1;
        }
        // --- Matmul Backward dB (Standard) ---
        case COMMAND_MATMUL_BACKWARD_DB: {
            MatMulBackwardData* cmd = (MatMulBackwardData*)data;
             if (!matmul_backward_db_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_dc || !cmd->buffer_db) { fprintf(stderr, "[C] Submit MatMul Bwd dB (Standard): Invalid args.\n"); return 0; }
            cl_mem a_mem = (cl_mem)cmd->buffer_a, dc = (cl_mem)cmd->buffer_dc, db = (cl_mem)cmd->buffer_db;
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 0, sizeof(cl_mem), &a_mem), "MatMul dB Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 1, sizeof(cl_mem), &dc), "MatMul dB Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 2, sizeof(cl_mem), &db), "MatMul dB Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 3, sizeof(cl_int), &cmd->B), "MatMul dB Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 4, sizeof(cl_int), &cmd->M), "MatMul dB Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 5, sizeof(cl_int), &cmd->N), "MatMul dB Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 6, sizeof(cl_int), &cmd->K), "MatMul dB Arg 6");
            // GWS maps to output dB[K, N] -> (N, K)
            size_t gws[2] = { (size_t)cmd->N, (size_t)cmd->K };
             if (gws[0] == 0 || gws[1] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_backward_db_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "MatMul dB Enqueue");
            return 1;
        }
        // --- Layer Normalization Backward ---
        case COMMAND_LAYER_NORM_BACKWARD: {
            LayerNormBackwardCommandData* cmd = (LayerNormBackwardCommandData*)data;
            if (!layernorm_backward_kernel || !cmd || !cmd->buffer_dy || !cmd->buffer_x || !cmd->buffer_dx) { fprintf(stderr, "[C] Submit LayerNorm Bwd: Invalid args.\n"); return 0; }
            cl_mem dy_mem = (cl_mem)cmd->buffer_dy; cl_mem x_mem = (cl_mem)cmd->buffer_x; cl_mem dx_mem = (cl_mem)cmd->buffer_dx;
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 0, sizeof(cl_mem), &dy_mem), "LayerNorm Bwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 1, sizeof(cl_mem), &x_mem), "LayerNorm Bwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 2, sizeof(cl_mem), &dx_mem), "LayerNorm Bwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 3, sizeof(cl_int), &cmd->num_rows), "LayerNorm Bwd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 4, sizeof(cl_int), &cmd->row_size), "LayerNorm Bwd Arg 4");
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 5, sizeof(cl_float), &cmd->eps), "LayerNorm Bwd Arg 5");
            size_t gws[1] = { (size_t)cmd->num_rows }; // 1D GWS, one item per row
            if (gws[0] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, layernorm_backward_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "LayerNorm Bwd Enqueue");
            return 1;
        }
        // --- Softmax Backward ---
        case COMMAND_SOFTMAX_BACKWARD: {
            SoftmaxBackwardCommandData* cmd = (SoftmaxBackwardCommandData*)data;
            if (!softmax_backward_kernel || !cmd || !cmd->buffer_dy || !cmd->buffer_y || !cmd->buffer_dx) { fprintf(stderr, "[C] Submit Softmax Bwd: Invalid args.\n"); return 0; }
            cl_mem dy = (cl_mem)cmd->buffer_dy; cl_mem y = (cl_mem)cmd->buffer_y; cl_mem dx = (cl_mem)cmd->buffer_dx;
            CHECK_CL_ERR(clSetKernelArg(softmax_backward_kernel, 0, sizeof(cl_mem), &dy), "Softmax Bwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(softmax_backward_kernel, 1, sizeof(cl_mem), &y), "Softmax Bwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(softmax_backward_kernel, 2, sizeof(cl_mem), &dx), "Softmax Bwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(softmax_backward_kernel, 3, sizeof(cl_int), &cmd->num_rows), "Softmax Bwd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(softmax_backward_kernel, 4, sizeof(cl_int), &cmd->row_size), "Softmax Bwd Arg 4");
            size_t gws[1] = { (size_t)cmd->num_rows }; // 1D GWS, one item per row
            if (gws[0] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, softmax_backward_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Softmax Bwd Enqueue");
            return 1;
        }
        // --- Multiply Backward ---
        case COMMAND_MUL_BACKWARD: {
            MulBackwardCommandData* cmd = (MulBackwardCommandData*)data;
            // Check required inputs (dC, A, B) and at least one output (dA or dB)
            if (!mul_backward_kernel || !cmd || !cmd->buffer_dC || !cmd->buffer_A || !cmd->buffer_B || (!cmd->buffer_dA && !cmd->buffer_dB)) {
                fprintf(stderr, "[C] Submit Mul Bwd: Invalid args (Need dC, A, B, and at least dA or dB).\n"); return 0;
            }
            cl_mem dC = (cl_mem)cmd->buffer_dC; cl_mem A_mem = (cl_mem)cmd->buffer_A; cl_mem B_mem = (cl_mem)cmd->buffer_B;
            // Pass NULL if buffer not provided, kernel handles this internally (but checks should exist)
            cl_mem dA = (cl_mem)cmd->buffer_dA; cl_mem dB = (cl_mem)cmd->buffer_dB;
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 0, sizeof(cl_mem), &dC), "Mul Bwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 1, sizeof(cl_mem), &A_mem), "Mul Bwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 2, sizeof(cl_mem), &B_mem), "Mul Bwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 3, sizeof(cl_mem), &dA), "Mul Bwd Arg 3"); // Can be NULL
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 4, sizeof(cl_mem), &dB), "Mul Bwd Arg 4"); // Can be NULL
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 5, sizeof(cl_int), &cmd->num_elements), "Mul Bwd Arg 5");
            size_t gws[1] = { (size_t)cmd->num_elements }; // 1D GWS
            if (gws[0] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, mul_backward_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Mul Bwd Enqueue");
            return 1;
        }
        // --- Transpose Backward (Basic 2D) ---
        case COMMAND_TRANSPOSE_BACKWARD: {
            TransposeBackwardCommandData* cmd = (TransposeBackwardCommandData*)data;
            if (!transpose_backward_kernel || !cmd || !cmd->buffer_dC || !cmd->buffer_dA ) { fprintf(stderr, "[C] Submit Transpose Bwd (2D): Invalid args.\n"); return 0; }
            cl_mem dC = (cl_mem)cmd->buffer_dC; cl_mem dA = (cl_mem)cmd->buffer_dA;
            CHECK_CL_ERR(clSetKernelArg(transpose_backward_kernel, 0, sizeof(cl_mem), &dC), "Transpose Bwd (2D) Arg 0");
            CHECK_CL_ERR(clSetKernelArg(transpose_backward_kernel, 1, sizeof(cl_mem), &dA), "Transpose Bwd (2D) Arg 1");
            CHECK_CL_ERR(clSetKernelArg(transpose_backward_kernel, 2, sizeof(cl_int), &cmd->rows_A), "Transpose Bwd (2D) Arg 2");
            CHECK_CL_ERR(clSetKernelArg(transpose_backward_kernel, 3, sizeof(cl_int), &cmd->cols_A), "Transpose Bwd (2D) Arg 3");
            // GWS maps to output dA[rows_A, cols_A] -> (rows_A, cols_A)
            size_t gws[2] = { (size_t)cmd->rows_A, (size_t)cmd->cols_A };
            if (gws[0] == 0 || gws[1] == 0) { return 1; }
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, transpose_backward_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "Transpose Bwd (2D) Enqueue");
            return 1;
        }
        // --- Embedding Lookup ---
        case COMMAND_EMBEDDING_LOOKUP: {
            EmbeddingLookupCommandData* cmd = (EmbeddingLookupCommandData*)data;
            if (!embedding_lookup_kernel || !cmd || !cmd->idx || !cmd->w || !cmd->o) { fprintf(stderr, "[C] Submit Embedding Lookup: Invalid args.\n"); return 0; }
            cl_mem idx_mem = (cl_mem)cmd->idx, w_mem = (cl_mem)cmd->w, o_mem = (cl_mem)cmd->o;
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 0, sizeof(cl_mem), &idx_mem), "Embedding Lookup Arg 0");
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 1, sizeof(cl_mem), &w_mem), "Embedding Lookup Arg 1");
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 2, sizeof(cl_mem), &o_mem), "Embedding Lookup Arg 2");
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 3, sizeof(cl_int), &cmd->s), "Embedding Lookup Arg 3"); // Seq Len
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 4, sizeof(cl_int), &cmd->d), "Embedding Lookup Arg 4"); // Embed Dim
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 5, sizeof(cl_int), &cmd->v), "Embedding Lookup Arg 5"); // Vocab Size
            // GWS maps to output (B, S) -> (S, B)
            size_t gws[2] = { (size_t)cmd->s, (size_t)cmd->b };
            if (gws[0] == 0 || gws[1] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, embedding_lookup_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "Embedding Lookup Enqueue");
            return 1;
        }
        // --- Embedding Backward ---
        case COMMAND_EMBEDDING_BACKWARD: {
            EmbeddingBackwardCommandData* cmd = (EmbeddingBackwardCommandData*)data;
            if (!embedding_backward_kernel || !cmd || !cmd->d_o || !cmd->idx || !cmd->d_w) { fprintf(stderr, "[C] Submit Embedding Bwd: Invalid args.\n"); return 0; }
            // Runtime check for atomics support (crucial!)
            if (!has_atomics_support) {
                fprintf(stderr, "[C] ERROR: Embedding Backward requires atomics, but support was not detected or is insufficient. Cannot execute.\n");
                return 0; // Fail explicitly if atomics are required but not supported
            }
            cl_mem d_o_mem = (cl_mem)cmd->d_o, idx_mem = (cl_mem)cmd->idx, d_w_mem = (cl_mem)cmd->d_w;
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_kernel, 0, sizeof(cl_mem), &d_o_mem), "Embedding Bwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_kernel, 1, sizeof(cl_mem), &idx_mem), "Embedding Bwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_kernel, 2, sizeof(cl_mem), &d_w_mem), "Embedding Bwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_kernel, 3, sizeof(cl_int), &cmd->s), "Embedding Bwd Arg 3"); // Seq Len
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_kernel, 4, sizeof(cl_int), &cmd->d), "Embedding Bwd Arg 4"); // Embed Dim
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_kernel, 5, sizeof(cl_int), &cmd->v), "Embedding Bwd Arg 5"); // Vocab Size
            // GWS maps to input grad/indices (B, S) -> (S, B)
            size_t gws[2] = { (size_t)cmd->s, (size_t)cmd->b };
            if (gws[0] == 0 || gws[1] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, embedding_backward_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "Embedding Bwd Enqueue");
            return 1;
        }
        // --- Reduce Sum (For Bias Gradient) ---
        case COMMAND_REDUCE_SUM_AXIS01: {
            ReduceSumCommandData* cmd = (ReduceSumCommandData*)data;
            if (!reduce_sum_kernel || !cmd || !cmd->in || !cmd->out) { fprintf(stderr, "[C] Submit ReduceSum Axis01: Invalid args.\n"); return 0; }
            cl_mem in_mem = (cl_mem)cmd->in, out_mem = (cl_mem)cmd->out;
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 0, sizeof(cl_mem), &in_mem), "ReduceSum Arg 0");
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 1, sizeof(cl_mem), &out_mem), "ReduceSum Arg 1");
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 2, sizeof(cl_int), &cmd->B), "ReduceSum Arg 2");
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 3, sizeof(cl_int), &cmd->M), "ReduceSum Arg 3");
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 4, sizeof(cl_int), &cmd->N), "ReduceSum Arg 4");

            #ifndef WORK_GROUP_SIZE_REDUCE // Define default if not provided externally
            #define WORK_GROUP_SIZE_REDUCE 256
            #endif
            size_t local_mem_size = WORK_GROUP_SIZE_REDUCE * sizeof(FP_TYPE); // Size of local memory buffer per work-group
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 5, local_mem_size, NULL), "ReduceSum Arg 5 (Local Mem)");

            if (cmd->N <= 0 || cmd->B <= 0 || cmd->M <= 0) return 1; // Nothing to reduce

            // Determine Local and Global Work Sizes
            size_t lws[1] = { WORK_GROUP_SIZE_REDUCE };
            // TODO: Add robust check against CL_DEVICE_MAX_WORK_GROUP_SIZE and CL_KERNEL_WORK_GROUP_SIZE if necessary
            // size_t max_lws; clGetKernelWorkGroupInfo(reduce_sum_kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_lws), &max_lws, NULL); if(lws[0] > max_lws) lws[0] = max_lws;

            // Number of work-groups needed to cover the output dimension N
            size_t num_groups = (cmd->N + lws[0] - 1) / lws[0]; // Ceiling division
            size_t gws[1] = { num_groups * lws[0] }; // Total global size must be multiple of local size

            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, reduce_sum_kernel, 1, NULL, gws, lws, 0, NULL, NULL), "ReduceSum Axis01 Enqueue");
            return 1;
        }
        // --- Broadcast Add (General Bias) ---
        case COMMAND_BROADCAST_ADD_BIAS: {
            BroadcastAddCommandData* cmd = (BroadcastAddCommandData*)data;
            if (!broadcast_add_kernel || !cmd || !cmd->a || !cmd->b || !cmd->c) { fprintf(stderr, "[C] Submit BroadcastAdd Bias: Invalid args.\n"); return 0; }
            cl_mem a = (cl_mem)cmd->a, b_bias = (cl_mem)cmd->b, c = (cl_mem)cmd->c;
            CHECK_CL_ERR(clSetKernelArg(broadcast_add_kernel, 0, sizeof(cl_mem), &a), "BroadcastAdd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(broadcast_add_kernel, 1, sizeof(cl_mem), &b_bias), "BroadcastAdd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(broadcast_add_kernel, 2, sizeof(cl_mem), &c), "BroadcastAdd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(broadcast_add_kernel, 3, sizeof(cl_int), &cmd->M), "BroadcastAdd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(broadcast_add_kernel, 4, sizeof(cl_int), &cmd->N), "BroadcastAdd Arg 4");
            // GWS maps to output tensor C[B, M, N] -> (N, M, B)
            size_t gws[3] = { (size_t)cmd->N, (size_t)cmd->M, (size_t)cmd->B };
            if (gws[0] == 0 || gws[1] == 0 || gws[2] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, broadcast_add_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "BroadcastAdd Enqueue");
            return 1;
        }
        // --- Transpose Last Two Dims ---
        case COMMAND_TRANSPOSE_BATCHED: {
            TransposeBatchedCommandData* cmd = (TransposeBatchedCommandData*)data;
            if (!transpose_batched_kernel || !cmd || !cmd->in || !cmd->out) { fprintf(stderr, "[C] Submit TransposeBatched (LastTwo): Invalid args.\n"); return 0; }
            cl_mem in_mem = (cl_mem)cmd->in, out_mem = (cl_mem)cmd->out;
            CHECK_CL_ERR(clSetKernelArg(transpose_batched_kernel, 0, sizeof(cl_mem), &in_mem), "TransposeBatched Arg 0");
            CHECK_CL_ERR(clSetKernelArg(transpose_batched_kernel, 1, sizeof(cl_mem), &out_mem), "TransposeBatched Arg 1");
            CHECK_CL_ERR(clSetKernelArg(transpose_batched_kernel, 2, sizeof(cl_int), &cmd->d1), "TransposeBatched Arg 2"); // Orig Dim -2 size
            CHECK_CL_ERR(clSetKernelArg(transpose_batched_kernel, 3, sizeof(cl_int), &cmd->d2), "TransposeBatched Arg 3"); // Orig Dim -1 size
            // GWS maps to output dimensions (..., Dim2, Dim1) -> Kernel uses (d2_out=Dim1, d1_out=Dim2, b_linear)
            size_t gws[3] = { (size_t)cmd->d1, (size_t)cmd->d2, (size_t)cmd->B_flat };
            if (gws[0] == 0 || gws[1] == 0 || gws[2] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, transpose_batched_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "TransposeBatched (LastTwo) Enqueue");
            return 1;
        }
        // --- Adam Optimizer Update ---
        case COMMAND_ADAM_UPDATE: {
            AdamCommandData* cmd = (AdamCommandData*)data;
            if (!adam_kernel || !cmd || !cmd->param_buffer || !cmd->grad_buffer || !cmd->m_buffer || !cmd->v_buffer) { fprintf(stderr, "[C] Submit Adam Update: Invalid args.\n"); return 0; }
            cl_mem p = (cl_mem)cmd->param_buffer; cl_mem g = (cl_mem)cmd->grad_buffer; cl_mem m = (cl_mem)cmd->m_buffer; cl_mem v = (cl_mem)cmd->v_buffer;
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 0, sizeof(cl_mem), &p), "Adam Arg 0");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 1, sizeof(cl_mem), &g), "Adam Arg 1");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 2, sizeof(cl_mem), &m), "Adam Arg 2");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 3, sizeof(cl_mem), &v), "Adam Arg 3");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 4, sizeof(cl_int), &cmd->num_elements), "Adam Arg 4");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 5, sizeof(cl_float), &cmd->lr), "Adam Arg 5");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 6, sizeof(cl_float), &cmd->beta1), "Adam Arg 6");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 7, sizeof(cl_float), &cmd->beta2), "Adam Arg 7");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 8, sizeof(cl_float), &cmd->eps), "Adam Arg 8");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 9, sizeof(cl_float), &cmd->weight_decay), "Adam Arg 9");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 10, sizeof(cl_float), &cmd->beta1_t), "Adam Arg 10");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 11, sizeof(cl_float), &cmd->beta2_t), "Adam Arg 11");
            size_t gws[1] = { (size_t)cmd->num_elements }; // 1D GWS
            if (gws[0] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, adam_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Adam Update Enqueue");
            return 1;
        }
        // --- Batched Matmul (3D @ 3D) ---
        case COMMAND_MATRIX_MULTIPLY_BATCHED: {
            BMMBatchedCommandData* cmd = (BMMBatchedCommandData*)data;
            if (!matmul_batched_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_b || !cmd->buffer_c) { fprintf(stderr, "[C] Submit BMM Batched Fwd: Invalid args.\n"); return 0; }
            cl_mem a = (cl_mem)cmd->buffer_a, b = (cl_mem)cmd->buffer_b, c = (cl_mem)cmd->buffer_c;
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 0, sizeof(cl_mem), &a), "BMM Batched Fwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 1, sizeof(cl_mem), &b), "BMM Batched Fwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 2, sizeof(cl_mem), &c), "BMM Batched Fwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 3, sizeof(cl_int), &cmd->B), "BMM Batched Fwd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 4, sizeof(cl_int), &cmd->M), "BMM Batched Fwd Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 5, sizeof(cl_int), &cmd->N), "BMM Batched Fwd Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 6, sizeof(cl_int), &cmd->K), "BMM Batched Fwd Arg 6");
            // GWS maps to output C[B, M, N] -> (N, M, B)
            size_t gws[3] = { (size_t)cmd->N, (size_t)cmd->M, (size_t)cmd->B };
            if (gws[0] == 0 || gws[1] == 0 || gws[2] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_batched_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "BMM Batched Fwd Enqueue");
            return 1;
        }
        // --- Batched Matmul Backward dA ---
        case COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DA: {
            BMMBatchedBackwardData* cmd = (BMMBatchedBackwardData*)data;
            if (!matmul_batched_backward_da_kernel || !cmd || !cmd->buffer_dc || !cmd->buffer_b || !cmd->buffer_da) { fprintf(stderr, "[C] Submit MatMul Batched Bwd dA: Invalid args.\n"); return 0; }
            cl_mem dc = (cl_mem)cmd->buffer_dc, b_in = (cl_mem)cmd->buffer_b, da = (cl_mem)cmd->buffer_da;
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 0, sizeof(cl_mem), &dc), "MatMul Batched dA Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 1, sizeof(cl_mem), &b_in), "MatMul Batched dA Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 2, sizeof(cl_mem), &da), "MatMul Batched dA Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 3, sizeof(cl_int), &cmd->B), "MatMul Batched dA Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 4, sizeof(cl_int), &cmd->M), "MatMul Batched dA Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 5, sizeof(cl_int), &cmd->N), "MatMul Batched dA Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 6, sizeof(cl_int), &cmd->K), "MatMul Batched dA Arg 6");
            // GWS maps to output dA[B, M, K] -> (K, M, B)
            size_t gws[3] = { (size_t)cmd->K, (size_t)cmd->M, (size_t)cmd->B };
             if (gws[0] == 0 || gws[1] == 0 || gws[2] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_batched_backward_da_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "MatMul Batched dA Enqueue");
            return 1;
        }
        // --- Batched Matmul Backward dB ---
        case COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DB: {
            BMMBatchedBackwardData* cmd = (BMMBatchedBackwardData*)data;
             if (!matmul_batched_backward_db_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_dc || !cmd->buffer_db) { fprintf(stderr, "[C] Submit MatMul Batched Bwd dB: Invalid args.\n"); return 0; }
            cl_mem a_in = (cl_mem)cmd->buffer_a, dc = (cl_mem)cmd->buffer_dc, db = (cl_mem)cmd->buffer_db;
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 0, sizeof(cl_mem), &a_in), "MatMul Batched dB Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 1, sizeof(cl_mem), &dc), "MatMul Batched dB Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 2, sizeof(cl_mem), &db), "MatMul Batched dB Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 3, sizeof(cl_int), &cmd->B), "MatMul Batched dB Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 4, sizeof(cl_int), &cmd->M), "MatMul Batched dB Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 5, sizeof(cl_int), &cmd->N), "MatMul Batched dB Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 6, sizeof(cl_int), &cmd->K), "MatMul Batched dB Arg 6");
            // GWS maps to output dB[B, K, N] -> (N, K, B)
            size_t gws[3] = { (size_t)cmd->N, (size_t)cmd->K, (size_t)cmd->B };
            if (gws[0] == 0 || gws[1] == 0 || gws[2] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_batched_backward_db_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "MatMul Batched dB Enqueue");
            return 1;
        }
        // --- Transpose(1,2) for 4D ---
        case COMMAND_TRANSPOSE_12_BATCHED: {
            Transpose12BatchedCommandData* cmd = (Transpose12BatchedCommandData*)data;
            if (!transpose_12_batched_kernel || !cmd || !cmd->in || !cmd->out) { fprintf(stderr, "[C] Submit Transpose12Batched: Invalid args.\n"); return 0; }
            cl_mem in_mem = (cl_mem)cmd->in; cl_mem out_mem = (cl_mem)cmd->out;
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 0, sizeof(cl_mem), &in_mem), "Transpose12 Arg 0");
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 1, sizeof(cl_mem), &out_mem), "Transpose12 Arg 1");
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 2, sizeof(cl_int), &cmd->B), "Transpose12 Arg 2");
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 3, sizeof(cl_int), &cmd->D1), "Transpose12 Arg 3");
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 4, sizeof(cl_int), &cmd->D2), "Transpose12 Arg 4");
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 5, sizeof(cl_int), &cmd->D3), "Transpose12 Arg 5");
            // GWS mapping based on kernel logic: (D3, D1, D2*B)
            size_t gws[3] = { (size_t)cmd->D3, (size_t)cmd->D1, (size_t)cmd->D2 * cmd->B };
            if (gws[0] == 0 || gws[1] == 0 || gws[2] == 0) return 1;
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, transpose_12_batched_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "Transpose12Batched Enqueue (3D)");
            return 1;
        }

        // --- NEUE CASES für Cross Entropy & PE Add ---
        case COMMAND_LOG_SOFTMAX_STABLE: {
            LogSoftmaxStableCommandData* cmd = (LogSoftmaxStableCommandData*)data;
            if (!log_softmax_kernel || !cmd || !cmd->input_logits || !cmd->output_log_probs) { fprintf(stderr, "[C] Submit LogSoftmaxStable: Invalid args.\n"); return 0; }
            cl_mem in_logits = (cl_mem)cmd->input_logits; cl_mem out_log_probs = (cl_mem)cmd->output_log_probs;
            if (cmd->B_S_rows <= 0 || cmd->V_cols <= 0) return 1; // Nothing to compute
            CHECK_CL_ERR(clSetKernelArg(log_softmax_kernel, 0, sizeof(cl_mem), &in_logits), "LogSoftmaxStable Arg 0");
            CHECK_CL_ERR(clSetKernelArg(log_softmax_kernel, 1, sizeof(cl_mem), &out_log_probs), "LogSoftmaxStable Arg 1");
            CHECK_CL_ERR(clSetKernelArg(log_softmax_kernel, 2, sizeof(cl_int), &cmd->B_S_rows), "LogSoftmaxStable Arg 2"); // num_rows = B*S
            CHECK_CL_ERR(clSetKernelArg(log_softmax_kernel, 3, sizeof(cl_int), &cmd->V_cols), "LogSoftmaxStable Arg 3");   // row_size = V
            size_t gws[1] = { (size_t)cmd->B_S_rows }; // 1D GWS, one item per row
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, log_softmax_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "LogSoftmaxStable Enqueue");
            return 1;
        }
        case COMMAND_CROSS_ENTROPY_LOSS_GRAD: {
            CrossEntropyLossGradCommandData* cmd = (CrossEntropyLossGradCommandData*)data;
            if (!cross_entropy_kernel || !cmd || !cmd->log_probs || !cmd->target_indices || !cmd->grad_input || !cmd->loss_per_sample) { fprintf(stderr, "[C] Submit CrossEntropyLossGrad: Invalid args.\n"); return 0; }
            cl_mem log_probs_mem = (cl_mem)cmd->log_probs; cl_mem target_indices_mem = (cl_mem)cmd->target_indices;
            cl_mem grad_input_mem = (cl_mem)cmd->grad_input; cl_mem loss_per_sample_mem = (cl_mem)cmd->loss_per_sample;
            if (cmd->B_S_rows <= 0 || cmd->V_cols <= 0) return 1; // Nothing to compute
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 0, sizeof(cl_mem), &log_probs_mem), "CrossEntropyLossGrad Arg 0");
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 1, sizeof(cl_mem), &target_indices_mem), "CrossEntropyLossGrad Arg 1");
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 2, sizeof(cl_mem), &grad_input_mem), "CrossEntropyLossGrad Arg 2");
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 3, sizeof(cl_mem), &loss_per_sample_mem), "CrossEntropyLossGrad Arg 3");
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 4, sizeof(cl_int), &cmd->B_S_rows), "CrossEntropyLossGrad Arg 4"); // num_rows = B*S
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 5, sizeof(cl_int), &cmd->V_cols), "CrossEntropyLossGrad Arg 5");   // V
            size_t gws[1] = { (size_t)cmd->B_S_rows }; // 1D GWS, one item per row (sample/token)
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, cross_entropy_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "CrossEntropyLossGrad Enqueue");
            return 1;
        }
        case COMMAND_ADD_BROADCAST_PE: {
            AddBroadcastPECommandData* cmd = (AddBroadcastPECommandData*)data;
            if (!add_broadcast_pe_kernel || !cmd || !cmd->input || !cmd->pe_slice || !cmd->output) { fprintf(stderr, "[C] Submit AddBroadcastPE: Invalid args.\n"); return 0; }
            cl_mem input_mem = (cl_mem)cmd->input; cl_mem pe_slice_mem = (cl_mem)cmd->pe_slice; cl_mem output_mem = (cl_mem)cmd->output;
            if (cmd->B <= 0 || cmd->S <= 0 || cmd->E <= 0) return 1; // Nothing to compute
            CHECK_CL_ERR(clSetKernelArg(add_broadcast_pe_kernel, 0, sizeof(cl_mem), &input_mem), "AddBroadcastPE Arg 0");
            CHECK_CL_ERR(clSetKernelArg(add_broadcast_pe_kernel, 1, sizeof(cl_mem), &pe_slice_mem), "AddBroadcastPE Arg 1");
            CHECK_CL_ERR(clSetKernelArg(add_broadcast_pe_kernel, 2, sizeof(cl_mem), &output_mem), "AddBroadcastPE Arg 2");
            CHECK_CL_ERR(clSetKernelArg(add_broadcast_pe_kernel, 3, sizeof(cl_int), &cmd->S), "AddBroadcastPE Arg 3");
            CHECK_CL_ERR(clSetKernelArg(add_broadcast_pe_kernel, 4, sizeof(cl_int), &cmd->E), "AddBroadcastPE Arg 4");
            // GWS maps to output (B, S, E) -> Kernel uses (E, S, B)
            size_t gws[3] = { (size_t)cmd->E, (size_t)cmd->S, (size_t)cmd->B };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, add_broadcast_pe_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "AddBroadcastPE Enqueue");
            return 1;
        }
        // ----------------------------------------

        default:
            fprintf(stderr, "[C] submit_kernel_command: Error - Unknown command code: %d\n", command);
            return 0; // Unknown command
    }
    #undef CHECK_CL_ERR // Undefine the macro after the switch block
}

// Helper function to wait for queue completion and check for errors
int finish_queue_and_check(int gpu_index, const char* func_name) {
    if (!queue) { fprintf(stderr, "[C] %s: Command queue is NULL. Cannot finish.\n", func_name); return 0; }
    // clFinish blocks until all previously queued commands for this queue have completed.
    cl_int err = clFinish(queue);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[C] %s: Error during clFinish after submitting commands: %s (%d)\n", func_name, clGetErrorString(err), err);
        return 0; // Indicate failure
    }
    return 1; // Indicate success
}

// --- Exported Function Definitions (Wrappers around submit_kernel_command) ---

DLLEXPORT int execute_matmul_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int B, int M, int N, int K) {
    if (!buffer_a || !buffer_b || !buffer_c) { fprintf(stderr, "[C] execute_matmul_on_gpu: Invalid buffer handles (NULL).\n"); return 0; }
    if (B <= 0 || M <= 0 || N <= 0 || K <= 0) {
        if ((size_t)B * M * N == 0 || K == 0) return 1; // Trivial if output size is 0 or K is 0
        fprintf(stderr, "[C] execute_matmul_on_gpu: Invalid non-positive dimensions (B=%d, M=%d, N=%d, K=%d).\n", B, M, N, K); return 0;
    }
    BMMCommandData cmd_data = { buffer_a, buffer_b, buffer_c, B, M, N, K };
    if (!submit_kernel_command(gpu_index, COMMAND_MATRIX_MULTIPLY, &cmd_data)) { fprintf(stderr, "[C] execute_matmul_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_matmul_on_gpu");
}

DLLEXPORT int execute_softmax_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_rows, int row_size) {
    if (!buffer_input || !buffer_output) { fprintf(stderr, "[C] execute_softmax_on_gpu: Invalid buffer handles (NULL).\n"); return 0; }
    if (num_rows <= 0 || row_size <= 0) {
         if (num_rows == 0 || row_size == 0) return 1; // Nothing to compute
         fprintf(stderr, "[C] execute_softmax_on_gpu: Invalid non-positive dimensions (num_rows=%d, row_size=%d).\n", num_rows, row_size); return 0;
    }
    SoftmaxCommandData cmd_data = { buffer_input, buffer_output, num_rows, row_size };
    if (!submit_kernel_command(gpu_index, COMMAND_SOFTMAX_ROWWISE, &cmd_data)) { fprintf(stderr, "[C] execute_softmax_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_softmax_on_gpu");
}

DLLEXPORT int execute_gelu_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_elements) {
    if (!buffer_input || !buffer_output) { fprintf(stderr, "[C] execute_gelu_on_gpu: Invalid buffer handles (NULL).\n"); return 0; }
    if (num_elements <= 0) {
        if (num_elements == 0) return 1;
        fprintf(stderr, "[C] execute_gelu_on_gpu: Invalid non-positive num_elements (%d).\n", num_elements); return 0;
    }
    GeluCommandData cmd_data = { buffer_input, buffer_output, num_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_GELU_ELEMENTWISE, &cmd_data)) { fprintf(stderr, "[C] execute_gelu_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_gelu_on_gpu");
}

DLLEXPORT int execute_add_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int num_elements) {
    if (!buffer_a || !buffer_b || !buffer_c) { fprintf(stderr, "[C] execute_add_on_gpu: Invalid buffer handles (NULL).\n"); return 0; }
    if (num_elements <= 0) {
        if (num_elements == 0) return 1;
        fprintf(stderr, "[C] execute_add_on_gpu: Invalid non-positive num_elements (%d).\n", num_elements); return 0;
    }
    AddCommandData cmd_data = { buffer_a, buffer_b, buffer_c, num_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_ADD_ELEMENTWISE, &cmd_data)) { fprintf(stderr, "[C] execute_add_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_add_on_gpu");
}

DLLEXPORT int execute_mul_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int num_elements) {
     if (!buffer_a || !buffer_b || !buffer_c) { fprintf(stderr, "[C] execute_mul_on_gpu: Invalid buffer handles (NULL).\n"); return 0; }
     if (num_elements <= 0) {
         if (num_elements == 0) return 1;
         fprintf(stderr, "[C] execute_mul_on_gpu: Invalid non-positive num_elements (%d).\n", num_elements); return 0;
     }
    MulCommandData cmd_data = { buffer_a, buffer_b, buffer_c, num_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_MUL_ELEMENTWISE, &cmd_data)) { fprintf(stderr, "[C] execute_mul_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_mul_on_gpu");
}

DLLEXPORT int execute_layernorm_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_rows, int row_size, float eps) {
    if (!buffer_input || !buffer_output) { fprintf(stderr, "[C] execute_layernorm_on_gpu: Invalid buffer handles (NULL).\n"); return 0; }
    if (num_rows <= 0 || row_size <= 0) {
        if (num_rows == 0 || row_size == 0) return 1;
         fprintf(stderr, "[C] execute_layernorm_on_gpu: Invalid non-positive dimensions (num_rows=%d, row_size=%d).\n", num_rows, row_size); return 0;
    }
    if (eps <= 0) { fprintf(stderr, "[C] execute_layernorm_on_gpu: Epsilon must be positive (%f). Using default 1e-5f.\n", eps); eps = 1e-5f; }
    LayerNormCommandData cmd_data = { buffer_input, buffer_output, num_rows, row_size, eps };
    if (!submit_kernel_command(gpu_index, COMMAND_LAYER_NORM, &cmd_data)) { fprintf(stderr, "[C] execute_layernorm_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_layernorm_on_gpu");
}

DLLEXPORT int execute_clone_on_gpu(int gpu_index, void* src_buffer, void* dst_buffer, size_t size) {
    if (!src_buffer || !dst_buffer) { fprintf(stderr, "[C] execute_clone_on_gpu: Invalid buffer handles (NULL).\n"); return 0; }
    if (size == 0) { return 1; }
    CloneCommandData cmd_data = { src_buffer, dst_buffer, size };
    if (!submit_kernel_command(gpu_index, COMMAND_CLONE, &cmd_data)) { fprintf(stderr, "[C] execute_clone_on_gpu: Failed to submit kernel command (clEnqueueCopyBuffer).\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_clone_on_gpu");
}

DLLEXPORT int execute_transpose_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int rows, int cols) {
    if (!buffer_input || !buffer_output) { fprintf(stderr, "[C] execute_transpose_on_gpu (2D): Invalid buffer handles (NULL).\n"); return 0; }
    if (rows <= 0 || cols <= 0) {
        if ((size_t)rows * cols == 0) return 1;
        fprintf(stderr, "[C] execute_transpose_on_gpu (2D): Invalid non-positive dimensions (rows=%d, cols=%d).\n", rows, cols); return 0;
    }
    TransposeCommandData cmd_data = { buffer_input, buffer_output, rows, cols };
    if (!submit_kernel_command(gpu_index, COMMAND_TRANSPOSE, &cmd_data)) { fprintf(stderr, "[C] execute_transpose_on_gpu (2D): Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_transpose_on_gpu");
}

DLLEXPORT int execute_gelu_backward_on_gpu(int gpu_index, void* buffer_input, void* buffer_grad_output, void* buffer_grad_input, int num_elements) {
    if (!buffer_input || !buffer_grad_output || !buffer_grad_input) { fprintf(stderr, "[C] execute_gelu_backward_on_gpu: Invalid buffer handles (NULL).\n"); return 0; }
     if (num_elements <= 0) {
        if (num_elements == 0) return 1;
        fprintf(stderr, "[C] execute_gelu_backward_on_gpu: Invalid non-positive num_elements (%d).\n", num_elements); return 0;
    }
    GeluBackwardCommandData cmd_data = { buffer_input, buffer_grad_output, buffer_grad_input, num_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_GELU_BACKWARD_ELEMENTWISE, &cmd_data)) { fprintf(stderr, "[C] execute_gelu_backward_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_gelu_backward_on_gpu");
}

DLLEXPORT int execute_matmul_backward_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_dc, void* buffer_da, void* buffer_db, int B, int M, int N, int K) {
    if (!buffer_a || !buffer_b || !buffer_dc) { fprintf(stderr, "[C] execute_matmul_backward_on_gpu: Invalid required input handles (A, B, dC are NULL).\n"); return 0; }
    if (!buffer_da && !buffer_db) { return 1; } // Nothing to compute if both output grads are NULL
    if (B <= 0 || M <= 0 || N <= 0 || K <= 0) {
         if ((size_t)B * M * N * K == 0) return 1; // Trivial if any relevant dimension is 0
         fprintf(stderr, "[C] execute_matmul_backward_on_gpu: Invalid non-positive dimensions (B=%d, M=%d, N=%d, K=%d).\n", B, M, N, K); return 0;
    }
    MatMulBackwardData cmd_data = { buffer_a, buffer_b, buffer_dc, buffer_da, buffer_db, B, M, N, K };
    int da_submitted = 0, db_submitted = 0; int success = 1;
    // Submit dA calculation if requested and valid
    if (buffer_da && (size_t)B * M * K > 0) {
        if (!submit_kernel_command(gpu_index, COMMAND_MATMUL_BACKWARD_DA, &cmd_data)) { fprintf(stderr, "[C] execute_matmul_backward_on_gpu: Failed to submit MatMul Bwd dA command.\n"); success = 0; }
        else { da_submitted = 1; }
    }
    // Submit dB calculation if requested and valid
    if (buffer_db && (size_t)K * N > 0) {
         if (!submit_kernel_command(gpu_index, COMMAND_MATMUL_BACKWARD_DB, &cmd_data)) { fprintf(stderr, "[C] execute_matmul_backward_on_gpu: Failed to submit MatMul Bwd dB command.\n"); success = 0; }
         else { db_submitted = 1; }
    }
    // Wait for completion only if something was submitted and no errors occurred during submission
    if ((da_submitted || db_submitted) && success) { return finish_queue_and_check(gpu_index, "execute_matmul_backward_on_gpu"); }
    else { return success; } // Return 1 if nothing was submitted, 0 if submission failed
}

DLLEXPORT int execute_layernorm_backward_on_gpu(int gpu_index, void* buffer_dy, void* buffer_x, void* buffer_dx, int num_rows, int row_size, float eps) {
    if (!buffer_dy || !buffer_x || !buffer_dx) { fprintf(stderr, "[C] execute_layernorm_backward_on_gpu: Invalid handles (NULL).\n"); return 0; }
    if (num_rows <= 0 || row_size <= 0) {
        if (num_rows == 0 || row_size == 0) return 1;
        fprintf(stderr, "[C] execute_layernorm_backward_on_gpu: Invalid non-positive dimensions (num_rows=%d, row_size=%d).\n", num_rows, row_size); return 0;
    }
     if (eps <= 0) { fprintf(stderr, "[C] execute_layernorm_backward_on_gpu: Epsilon must be positive (%f). Using default 1e-5f.\n", eps); eps = 1e-5f; }
    LayerNormBackwardCommandData cmd_data = { buffer_dy, buffer_x, buffer_dx, num_rows, row_size, eps };
    if (!submit_kernel_command(gpu_index, COMMAND_LAYER_NORM_BACKWARD, &cmd_data)) { fprintf(stderr, "[C] execute_layernorm_backward_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_layernorm_backward_on_gpu");
}

DLLEXPORT int execute_softmax_backward_on_gpu(int gpu_index, void* buffer_dy, void* buffer_y, void* buffer_dx, int num_rows, int row_size) {
    if (!buffer_dy || !buffer_y || !buffer_dx) { fprintf(stderr, "[C] execute_softmax_backward_on_gpu: Invalid handles (NULL).\n"); return 0; }
    if (num_rows <= 0 || row_size <= 0) {
        if (num_rows == 0 || row_size == 0) return 1;
        fprintf(stderr, "[C] execute_softmax_backward_on_gpu: Invalid non-positive dimensions (num_rows=%d, row_size=%d).\n", num_rows, row_size); return 0;
    }
    SoftmaxBackwardCommandData cmd_data = { buffer_dy, buffer_y, buffer_dx, num_rows, row_size };
    if (!submit_kernel_command(gpu_index, COMMAND_SOFTMAX_BACKWARD, &cmd_data)) { fprintf(stderr, "[C] execute_softmax_backward_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_softmax_backward_on_gpu");
}

DLLEXPORT int execute_mul_backward_on_gpu(int gpu_index, void* buffer_dC, void* buffer_A, void* buffer_B, void* buffer_dA, void* buffer_dB, int num_elements) {
    if (!buffer_dC || !buffer_A || !buffer_B) { fprintf(stderr, "[C] execute_mul_backward_on_gpu: Invalid required input handles (dC, A, B are NULL).\n"); return 0; }
    if (!buffer_dA && !buffer_dB) { return 1; } // Nothing to compute if both output grads are NULL
    if (num_elements <= 0) {
         if (num_elements == 0) return 1;
         fprintf(stderr, "[C] execute_mul_backward_on_gpu: Invalid non-positive num_elements (%d).\n", num_elements); return 0;
    }
    // Kernel calculates both dA and dB; caller passes valid buffers for those needed.
    MulBackwardCommandData cmd_data = { buffer_dC, buffer_A, buffer_B, buffer_dA, buffer_dB, num_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_MUL_BACKWARD, &cmd_data)) { fprintf(stderr, "[C] execute_mul_backward_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_mul_backward_on_gpu");
}

DLLEXPORT int execute_transpose_backward_on_gpu(int gpu_index, void* buffer_dC, void* buffer_dA, int rows_A, int cols_A) {
    if (!buffer_dC || !buffer_dA) { fprintf(stderr, "[C] execute_transpose_backward_on_gpu (2D): Invalid handles (NULL).\n"); return 0; }
    if (rows_A <= 0 || cols_A <= 0) {
        if ((size_t)rows_A * cols_A == 0) return 1;
        fprintf(stderr, "[C] execute_transpose_backward_on_gpu (2D): Invalid non-positive dimensions (rows_A=%d, cols_A=%d).\n", rows_A, cols_A); return 0;
    }
    TransposeBackwardCommandData cmd_data = { buffer_dC, buffer_dA, rows_A, cols_A };
    if (!submit_kernel_command(gpu_index, COMMAND_TRANSPOSE_BACKWARD, &cmd_data)) { fprintf(stderr, "[C] execute_transpose_backward_on_gpu (2D): Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_transpose_backward_on_gpu");
}

DLLEXPORT int execute_embedding_lookup_gpu(int gpu_index, void* idx, void* w, void* o, int b, int s, int d, int v) {
    if (!idx || !w || !o) { fprintf(stderr, "[C] execute_embedding_lookup_gpu: Invalid handles (NULL).\n"); return 0; }
    if (b <= 0 || s <= 0 || d <= 0 || v <= 0) {
        if ((size_t)b * s * d == 0) return 1; // Trivial if output size is zero
        fprintf(stderr, "[C] execute_embedding_lookup_gpu: Invalid non-positive dimensions (b=%d, s=%d, d=%d, v=%d).\n", b, s, d, v); return 0;
    }
    EmbeddingLookupCommandData cd = { idx, w, o, b, s, d, v };
    if (!submit_kernel_command(gpu_index, COMMAND_EMBEDDING_LOOKUP, &cd)) { fprintf(stderr, "[C] execute_embedding_lookup_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_embedding_lookup_gpu");
}

DLLEXPORT int execute_embedding_backward_gpu(int gpu_index, void* d_o, void* idx, void* d_w, int b, int s, int d, int v) {
    if (!d_o || !idx || !d_w) { fprintf(stderr, "[C] execute_embedding_backward_gpu: Invalid handles (NULL).\n"); return 0; }
    if (b <= 0 || s <= 0 || d <= 0 || v <= 0) {
         if ((size_t)b * s * d == 0 || (size_t)v * d == 0) return 1; // Trivial if grad output or weight grad is zero size
        fprintf(stderr, "[C] execute_embedding_backward_gpu: Invalid non-positive dimensions (b=%d, s=%d, d=%d, v=%d).\n", b, s, d, v); return 0;
    }
    // Runtime check: ensure atomics were detected during initialization
    if (!has_atomics_support) {
        fprintf(stderr, "[C] ERROR: execute_embedding_backward_gpu requires atomics support, but it was not detected for GPU %d. Aborting.\n", gpu_index);
        return 0;
    }
    EmbeddingBackwardCommandData cd = { d_o, idx, d_w, b, s, d, v };
    if (!submit_kernel_command(gpu_index, COMMAND_EMBEDDING_BACKWARD, &cd)) { fprintf(stderr, "[C] execute_embedding_backward_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_embedding_backward_gpu");
}

DLLEXPORT int execute_reduce_sum_gpu(int gpu_index, void* in, void* out, int B, int M, int N) {
    if (!in || !out) { fprintf(stderr, "[C] execute_reduce_sum_gpu: Invalid handles (NULL).\n"); return 0; }
     if (B <= 0 || M <= 0 || N <= 0) {
         if ((size_t)B * M * N == 0) return 1; // Trivial if input is zero size
         fprintf(stderr, "[C] execute_reduce_sum_gpu: Invalid non-positive dimensions (B=%d, M=%d, N=%d).\n", B, M, N); return 0;
     }
    #ifndef WORK_GROUP_SIZE_REDUCE
    #define WORK_GROUP_SIZE_REDUCE 256
    #endif
    size_t local_mem_size = WORK_GROUP_SIZE_REDUCE * sizeof(KERNEL_FP_TYPE);
    ReduceSumCommandData cd = { in, out, B, M, N, local_mem_size };
    if (!submit_kernel_command(gpu_index, COMMAND_REDUCE_SUM_AXIS01, &cd)) { fprintf(stderr, "[C] execute_reduce_sum_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_reduce_sum_gpu");
}

DLLEXPORT int execute_broadcast_add_gpu(int gpu_index, void* a, void* b, void* c, int B, int M, int N) {
    if (!a || !b || !c) { fprintf(stderr, "[C] execute_broadcast_add_gpu: Invalid handles (NULL).\n"); return 0; }
    if (B <= 0 || M <= 0 || N <= 0) {
        if ((size_t)B * M * N == 0) return 1; // Trivial if output is zero size
        fprintf(stderr, "[C] execute_broadcast_add_gpu: Invalid non-positive dimensions (B=%d, M=%d, N=%d).\n", B, M, N); return 0;
    }
    BroadcastAddCommandData cd = { a, b, c, B, M, N };
    if (!submit_kernel_command(gpu_index, COMMAND_BROADCAST_ADD_BIAS, &cd)) { fprintf(stderr, "[C] execute_broadcast_add_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_broadcast_add_gpu");
}

DLLEXPORT int execute_transpose_batched_gpu(int gpu_index, void* in, void* out, int B_flat, int d1, int d2) {
     if (!in || !out) { fprintf(stderr, "[C] execute_transpose_batched_gpu (LastTwo): Invalid handles (NULL).\n"); return 0; }
    if (B_flat <= 0 || d1 <= 0 || d2 <= 0) {
         if ((size_t)B_flat * d1 * d2 == 0) return 1; // Trivial if zero size
         fprintf(stderr, "[C] execute_transpose_batched_gpu (LastTwo): Invalid non-positive dimensions (B_flat=%d, d1=%d, d2=%d).\n", B_flat, d1, d2); return 0;
     }
     TransposeBatchedCommandData cd = { in, out, B_flat, d1, d2 };
     if (!submit_kernel_command(gpu_index, COMMAND_TRANSPOSE_BATCHED, &cd)) { fprintf(stderr, "[C] execute_transpose_batched_gpu (LastTwo): Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_transpose_batched_gpu");
}

DLLEXPORT int execute_adam_update_on_gpu(int gpu_index, void* param_buffer, void* grad_buffer, void* m_buffer, void* v_buffer, int num_elements, int t, float lr, float beta1, float beta2, float eps, float weight_decay) {
    if (!param_buffer || !grad_buffer || !m_buffer || !v_buffer) { fprintf(stderr, "[C] execute_adam_update_on_gpu: Invalid buffer handles (NULL).\n"); return 0; }
    if (num_elements <= 0) {
         if (num_elements == 0) return 1;
        fprintf(stderr, "[C] execute_adam_update_on_gpu: Invalid non-positive num_elements (%d).\n", num_elements); return 0;
    }
    if (t <= 0) { fprintf(stderr, "[C] execute_adam_update_on_gpu: Invalid time step t (%d). Must be > 0.\n", t); return 0; }
    if (lr < 0.0f) { fprintf(stderr, "[C] execute_adam_update_on_gpu: Invalid lr (%f).\n", lr); return 0; }
    if (beta1 < 0.0f || beta1 >= 1.0f) { fprintf(stderr, "[C] execute_adam_update_on_gpu: Invalid beta1 (%f).\n", beta1); return 0; }
    if (beta2 < 0.0f || beta2 >= 1.0f) { fprintf(stderr, "[C] execute_adam_update_on_gpu: Invalid beta2 (%f).\n", beta2); return 0; }
    if (eps < 0.0f) { fprintf(stderr, "[C] execute_adam_update_on_gpu: Invalid epsilon (%f).\n", eps); return 0; }
    if (weight_decay < 0.0f) { fprintf(stderr, "[C] execute_adam_update_on_gpu: Invalid weight_decay (%f).\n", weight_decay); return 0; }

    // Precompute powers of beta (can be done on host)
    float beta1_t = (float)pow((double)beta1, (double)t);
    float beta2_t = (float)pow((double)beta2, (double)t);

    AdamCommandData cmd_data = { param_buffer, grad_buffer, m_buffer, v_buffer, num_elements, t, lr, beta1, beta2, eps, weight_decay, beta1_t, beta2_t };
    if (!submit_kernel_command(gpu_index, COMMAND_ADAM_UPDATE, &cmd_data)) { fprintf(stderr, "[C] execute_adam_update_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_adam_update_on_gpu");
}

DLLEXPORT int execute_matmul_batched_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int B, int M, int N, int K) {
    if (!buffer_a || !buffer_b || !buffer_c) { fprintf(stderr, "[C] execute_matmul_batched_on_gpu: Invalid buffer handles (NULL).\n"); return 0; }
    if (B <= 0 || M <= 0 || N <= 0 || K <= 0) {
        if ((size_t)B * M * N == 0 || K == 0) return 1;
        fprintf(stderr, "[C] execute_matmul_batched_on_gpu: Invalid non-positive dimensions (B=%d, M=%d, N=%d, K=%d).\n", B, M, N, K); return 0;
    }
    BMMBatchedCommandData cmd_data = { buffer_a, buffer_b, buffer_c, B, M, N, K };
    if (!submit_kernel_command(gpu_index, COMMAND_MATRIX_MULTIPLY_BATCHED, &cmd_data)) { fprintf(stderr, "[C] execute_matmul_batched_on_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_matmul_batched_on_gpu");
}

DLLEXPORT int execute_matmul_batched_backward_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_dc, void* buffer_da, void* buffer_db, int B, int M, int N, int K) {
    if (!buffer_a || !buffer_b || !buffer_dc ) { fprintf(stderr, "[C] execute_matmul_batched_backward_on_gpu: Invalid required handles (A, B, dC are NULL).\n"); return 0; }
    if (!buffer_da && !buffer_db) { return 1; } // Nothing to compute
    if (B <= 0 || M <= 0 || N <= 0 || K <= 0) {
        if ((size_t)B * M * N * K == 0) return 1;
        fprintf(stderr, "[C] execute_matmul_batched_backward_on_gpu: Invalid non-positive dimensions (B=%d, M=%d, N=%d, K=%d).\n", B, M, N, K); return 0;
    }
    BMMBatchedBackwardData cmd_data = { buffer_a, buffer_b, buffer_dc, buffer_da, buffer_db, B, M, N, K };
    int da_submitted = 0, db_submitted = 0; int success = 1;
    if (buffer_da && (size_t)B * M * K > 0) {
        if (!submit_kernel_command(gpu_index, COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DA, &cmd_data)) { fprintf(stderr, "[C] execute_matmul_batched_backward_on_gpu: Failed to submit dA command.\n"); success = 0; }
        else { da_submitted = 1; }
    }
    if (buffer_db && (size_t)B * K * N > 0) {
         if (!submit_kernel_command(gpu_index, COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DB, &cmd_data)) { fprintf(stderr, "[C] execute_matmul_batched_backward_on_gpu: Failed to submit dB command.\n"); success = 0; }
         else { db_submitted = 1; }
    }
    if ((da_submitted || db_submitted) && success) { return finish_queue_and_check(gpu_index, "execute_matmul_batched_backward_on_gpu"); }
    else { return success; }
}

DLLEXPORT int execute_transpose_12_batched_gpu(int gpu_index, void* buffer_in, void* buffer_out, int B, int D1, int D2, int D3) {
    if (!buffer_in || !buffer_out) { fprintf(stderr, "[C] execute_transpose_12_batched_gpu (GPU %d): Invalid buffer handles (in=%p, out=%p).\n", gpu_index, buffer_in, buffer_out); return 0; }
    if (B <= 0 || D1 <= 0 || D2 <= 0 || D3 <= 0) {
         size_t num_elements = (size_t)B * D1 * D2 * D3;
         if (num_elements == 0) return 1; // Nothing to do
        fprintf(stderr, "[C] execute_transpose_12_batched_gpu (GPU %d): Invalid non-positive dimensions (B=%d, D1=%d, D2=%d, D3=%d).\n", gpu_index, B, D1, D2, D3); return 0;
    }
    Transpose12BatchedCommandData cmd_data = { buffer_in, buffer_out, B, D1, D2, D3 };
    if (!submit_kernel_command(gpu_index, COMMAND_TRANSPOSE_12_BATCHED, &cmd_data)) { fprintf(stderr, "[C] execute_transpose_12_batched_gpu (GPU %d): Failed to submit kernel command.\n", gpu_index); return 0; }
    return finish_queue_and_check(gpu_index, "execute_transpose_12_batched_gpu");
}

// --- NEUE Export Funktionen für Cross Entropy & PE Add ---

DLLEXPORT int execute_log_softmax_stable_gpu(int gpu_index, void* input_logits, void* output_log_probs, int B, int S, int V) {
    if (!input_logits || !output_log_probs) { fprintf(stderr, "[C] execute_log_softmax_stable_gpu: Invalid buffer handles (NULL).\n"); return 0; }
     if (B <= 0 || S <= 0 || V <= 0) {
         if ((size_t)B * S * V == 0) return 1;
        fprintf(stderr, "[C] execute_log_softmax_stable_gpu: Invalid non-positive dimensions (B=%d, S=%d, V=%d).\n", B, S, V); return 0;
    }
    // Combine B and S for the kernel which operates row-wise
    int num_rows = B * S;
    LogSoftmaxStableCommandData cmd_data = { input_logits, output_log_probs, num_rows, V };
    if (!submit_kernel_command(gpu_index, COMMAND_LOG_SOFTMAX_STABLE, &cmd_data)) { fprintf(stderr, "[C] execute_log_softmax_stable_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_log_softmax_stable_gpu");
}

DLLEXPORT int execute_cross_entropy_loss_grad_gpu(int gpu_index, void* log_probs, void* target_indices, void* grad_input, void* loss_per_sample, int B, int S, int V) {
    if (!log_probs || !target_indices || !grad_input || !loss_per_sample) { fprintf(stderr, "[C] execute_cross_entropy_loss_grad_gpu: Invalid buffer handles (NULL).\n"); return 0; }
     if (B <= 0 || S <= 0 || V <= 0) {
         if ((size_t)B * S * V == 0) return 1;
        fprintf(stderr, "[C] execute_cross_entropy_loss_grad_gpu: Invalid non-positive dimensions (B=%d, S=%d, V=%d).\n", B, S, V); return 0;
    }
    // Combine B and S for the kernel
    int num_rows = B * S;
    CrossEntropyLossGradCommandData cmd_data = { log_probs, target_indices, grad_input, loss_per_sample, num_rows, V };
     if (!submit_kernel_command(gpu_index, COMMAND_CROSS_ENTROPY_LOSS_GRAD, &cmd_data)) { fprintf(stderr, "[C] execute_cross_entropy_loss_grad_gpu: Failed to submit kernel command.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_cross_entropy_loss_grad_gpu");
}

DLLEXPORT int execute_add_broadcast_pe_gpu(int gpu_index, void* input, void* pe_slice, void* output, int B, int S, int E) {
    if (!input || !pe_slice || !output) { fprintf(stderr, "[C] execute_add_broadcast_pe_gpu: Invalid handles (NULL).\n"); return 0; }
    if (B <= 0 || S <= 0 || E <= 0) {
        if ((size_t)B * S * E == 0) return 1; // Trivial if zero size
        fprintf(stderr, "[C] execute_add_broadcast_pe_gpu: Invalid non-positive dimensions (B=%d, S=%d, E=%d).\n", B, S, E); return 0;
    }
    AddBroadcastPECommandData cmd_data = { input, pe_slice, output, B, S, E };
    if (!submit_kernel_command(gpu_index, COMMAND_ADD_BROADCAST_PE, &cmd_data)) { fprintf(stderr, "[C] execute_add_broadcast_pe_gpu: Failed submit.\n"); return 0; }
    return finish_queue_and_check(gpu_index, "execute_add_broadcast_pe_gpu");
}
// ---------------------------------------------

// --- Simulation Layer (Dummy implementations) ---
// These allow basic testing or compilation on systems without OpenCL drivers.
DLLEXPORT unsigned long long simulated_kernel_allocate(int gpu_index, size_t size) {
    if (size == 0) return 0;
    void* ptr = malloc(size);
    if (!ptr) { fprintf(stderr, "[C SIM] simulated_kernel_allocate: malloc failed for size %zu.\n", size); return 0; }
    // printf("[C SIM] Allocated %zu bytes at %p\n", size, ptr); // Debug
    return (unsigned long long)ptr; // Return address as integer
}

DLLEXPORT void simulated_kernel_free(int gpu_index, unsigned long long address, size_t size) {
    if (address == 0) return;
    // printf("[C SIM] Freeing address %p (size %zu)\n", (void*)address, size); // Debug
    free((void*)address); // Free the host memory
}

DLLEXPORT void simulated_kernel_write(int gpu_index, unsigned long long address, size_t size, const void *source) {
    if (address == 0 || size == 0 || source == NULL) return;
     // printf("[C SIM] Writing %zu bytes to %p from %p\n", size, (void*)address, source); // Debug
    memcpy((void*)address, source, size); // Copy host to host memory
}

DLLEXPORT unsigned int simulated_get_compute_unit_count(int gpu_index) {
    // Try to return the actual count if OpenCL was initialized, otherwise a default.
    if (device_id) { return get_compute_unit_count(gpu_index); }
    else {
        // printf("[C SIM] Returning default CU count (16)\n"); // Debug
        return 16; // Return a plausible default value (e.g., 16 CUs)
    }
}
