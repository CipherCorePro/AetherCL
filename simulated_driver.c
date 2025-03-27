#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For memcpy
#include <time.h>   // For srand/rand

// --- OpenCL Headers ---
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// --- Platform Specific Headers ---
// Assuming Windows based on the path separator in the Python output
#ifndef __linux__
    // Dummy mmap/munmap/PCI for non-Linux (Keep as is if not needed)
    #define PROT_READ 1
    #define PROT_WRITE 2
    #define MAP_SHARED 1
    #define MAP_FAILED ((void *) -1)
    void* mmap(void* addr, size_t length, int prot, int flags, int fd, long offset) { return MAP_FAILED; }
    int munmap(void* addr, size_t length) { return -1; }
    // Dummy PCI read
    unsigned int read_pci_config(int gpu_index, int offset) {
        printf("[C Driver-Dummy] Reading PCI config for GPU %d, offset %d (returning dummy 0)\n", gpu_index, offset);
        return 0; // Return a dummy value
    }
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/mman.h>
    // Implement real read_pci_config for Linux if needed
    unsigned int read_pci_config(int gpu_index, int offset) { /* ... Linux implementation ... */ return 0;}
#endif

// --- Globals ---
// MMAP (Not really used in the current Python flow, but kept for structure)
#define SIMULATED_MMIO_SIZE (1024 * 1024 * 256)
void *simulated_mmio_base = NULL; // Example, not used by malloc below
size_t simulated_mmio_allocated = 0; // Example, not used by malloc below

// REAL OpenCL Objects
cl_context       context = NULL;
cl_command_queue queue = NULL;
cl_device_id     device_id = NULL; // Assuming single device for now
cl_platform_id   platform_id = NULL;

// --- Utility Functions ---
void initialize_rng() { srand((unsigned int)time(NULL)); }

// Simple clGetErrorString Implementation
const char* clGetErrorString(cl_int error) {
    switch (error) {
        // Run-time and JIT compiler errors
        case CL_SUCCESS:                            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // Compile-time errors
        case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";

        default:                                    return "Unknown OpenCL error";
    }
}


// --- GPU Initialization (Real OpenCL) ---
// Export needed for Python ctypes
__declspec(dllexport)
int initialize_gpu(int gpu_index) {
    cl_int err;
    cl_uint num_platforms;

    // Plattform-IDs abrufen
    err = clGetPlatformIDs(0, NULL, &num_platforms);
     if (err != CL_SUCCESS || num_platforms == 0) {
        printf("initialize_gpu: Fehler bei clGetPlatformIDs (count): %s (%d), num_platforms=%u\n", clGetErrorString(err), err, num_platforms);
        return 0; // Use 0 for failure consistent with other funcs
    }
    // Nur die erste Plattform verwenden
    err = clGetPlatformIDs(1, &platform_id, NULL);
    if (err != CL_SUCCESS) {
        printf("initialize_gpu: Fehler bei clGetPlatformIDs (get): %s (%d)\n", clGetErrorString(err), err);
        return 0;
    }

    // Geräte-ID abrufen (nur die erste GPU)
    cl_uint num_devices;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
     if (err != CL_SUCCESS || num_devices == 0) {
        printf("initialize_gpu: Fehler bei clGetDeviceIDs (count): %s (%d), num_devices=%u\n", clGetErrorString(err), err, num_devices);
        // Try CPU as fallback? For now, fail.
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        printf("initialize_gpu: Trying CL_DEVICE_TYPE_ALL, num_devices=%u\n", num_devices);
        if (num_devices == 0) return 0; // Still no devices
         err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
    } else {
         err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    }

    if (err != CL_SUCCESS) {
        printf("initialize_gpu: Fehler bei clGetDeviceIDs (get specific): %s (%d)\n", clGetErrorString(err), err);
        return 0;
    }

    // Kontext erstellen
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) {
        printf("initialize_gpu: Fehler bei clCreateContext: %s (%d)\n", clGetErrorString(err), err);
        return 0;
    }

    // Kommando-Queue erstellen (deprecierte Version oft kompatibler)
    #ifdef CL_VERSION_2_0
        queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    #else
        queue = clCreateCommandQueue(context, device_id, 0, &err);
    #endif

    if (!queue || err != CL_SUCCESS) {
        printf("initialize_gpu: Fehler bei clCreateCommandQueue: %s (%d)\n", clGetErrorString(err), err);
        // Cleanup context if queue fails
        clReleaseContext(context);
        context = NULL;
        return 0;
    }

    printf("initialize_gpu: Initialisierung erfolgreich (Context: %p, Queue: %p).\n", (void*)context, (void*)queue);
    return 1; // Erfolg
}


// --- GPU Memory Management (Using OpenCL Buffers) ---
// THESE ARE THE *REAL* GPU ALLOCATIONS via OpenCL
// Note: Python currently calls simulated_kernel_allocate, not this one directly.
void *allocate_gpu_memory(int gpu_index, size_t size) {
    cl_int err;
    if (!context) {
        printf("[C Driver][allocate_gpu_memory] Fehler: Kein gültiger OpenCL-Kontext.\n");
        return NULL;
    }
    if (size == 0) {
        printf("[C Driver][allocate_gpu_memory] Warnung: Allokation von Größe 0 angefordert.\n");
        return NULL; // Or handle as appropriate
    }
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (!buffer || err != CL_SUCCESS) {
        printf("[C Driver][allocate_gpu_memory] Fehler bei clCreateBuffer: %s (%d)\n", clGetErrorString(err), err);
        return NULL;
    }
    printf("[C Driver][allocate_gpu_memory] OpenCL Buffer erstellt (handle=%p), Size=%zu\n", (void*)buffer, size);
    return (void*)buffer; // Return the cl_mem handle cast to void*
}

void free_gpu_memory(int gpu_index, void* buffer_handle) {
    if (!buffer_handle) return; // Ignore NULL frees

    cl_mem buffer = (cl_mem)buffer_handle;
    cl_int err = clReleaseMemObject(buffer);
    if (err != CL_SUCCESS) {
         // Log error but don't crash
         fprintf(stderr, "[C Driver][free_gpu_memory] Warnung: clReleaseMemObject fehlgeschlagen für Handle %p: %s (%d)\n", buffer_handle, clGetErrorString(err), err);
    } else {
         printf("[C Driver][free_gpu_memory] OpenCL Buffer freigegeben (handle=%p)\n", buffer_handle);
    }
}

// --- Data Transfer ---

// simulated_kernel_write: Host Python Buffer -> Host Malloc'd Buffer
// This is called by Python *before* the kernel.
__declspec(dllexport) // Export for Python
void simulated_kernel_write(int gpu_index, unsigned long long address, size_t size, const void *source) {
    if (address == 0 || source == NULL || size == 0) {
        printf("[C Driver][simulated_kernel_write] Fehler: Ungültige Argumente (addr=%llu, src=%p, size=%zu)\n", address, source, size);
        return;
    }
    void* dest_ptr = (void*)address; // The address from simulated_kernel_allocate (malloc)
    printf("[C Driver][simulated_kernel_write] Kopiere %zu Bytes von Host-Quelle %p nach Host-Ziel %p (addr %llu)\n", size, source, dest_ptr, address);
    memcpy(dest_ptr, source, size);
    printf("[C Driver]   Host-zu-Host-Kopie (simulated_kernel_write) abgeschlossen.\n");
}

// write_to_gpu: Host Malloc'd Buffer -> OpenCL GPU Buffer
// This is called *inside* simulated_matrix_multiply.
// *** THIS WAS THE MISSING IMPLEMENTATION ***
int write_to_gpu(int gpu_index, void* gpu_buffer_handle, unsigned long long host_src_address, size_t offset, size_t size) {
    if (!queue || !gpu_buffer_handle || host_src_address == 0 || size == 0) {
        printf("[C Driver][write_to_gpu] Fehler: Ungültige Argumente (queue=%p, handle=%p, host_addr=%llu, size=%zu)\n",
                (void*)queue, gpu_buffer_handle, host_src_address, size);
        return 0; // Failure
    }

    // Cast host_src_address (which is from simulated_kernel_allocate's malloc)
    void* host_ptr = (void*)host_src_address;
    cl_mem gpu_buffer = (cl_mem)gpu_buffer_handle;

    printf("[C Driver][write_to_gpu] Starte clEnqueueWriteBuffer: %zu Bytes von Host %p nach GPU %p\n",
           size, host_ptr, gpu_buffer_handle);

    // Perform the write (blocking for simplicity and debugging)
    cl_int err = clEnqueueWriteBuffer(queue,        // Command queue
                                    gpu_buffer,     // GPU buffer
                                    CL_TRUE,        // Blocking write - IMPORTANT
                                    offset,         // Offset in GPU buffer
                                    size,           // Size to write
                                    host_ptr,       // Pointer to host data (from malloc)
                                    0, NULL, NULL); // Events

    if (err != CL_SUCCESS) {
        printf("[C Driver][write_to_gpu] *** FEHLER *** bei clEnqueueWriteBuffer: %s (%d)\n", clGetErrorString(err), err);
        return 0; // Failure
    }

    printf("[C Driver][write_to_gpu] Erfolgreich %zu Bytes von Host %p nach GPU Puffer %p geschrieben\n",
           size, host_ptr, gpu_buffer_handle);
    return 1; // Success
}


// --- Command Submission Abstraction ---
typedef enum { COMMAND_MATRIX_MULTIPLY = 1, } GPUCommand;
typedef struct {
    void* buffer_a;   // cl_mem A
    void* buffer_b;   // cl_mem B
    void* buffer_c;   // cl_mem C
    size_t size_a;
    size_t size_b;
    size_t size_c;
    int cols_a;
    int cols_b;
    int rows_res;
    int cols_res;
} MatrixMultiplyCommandData;


// --- submit_command (Simulates Hardware Command Execution via CPU+OpenCL Read/Write) ---
int submit_command(int gpu_index, GPUCommand command, void *data) {
    cl_int err = CL_SUCCESS;

    switch(command) {
        case COMMAND_MATRIX_MULTIPLY: {
            // printf("[C Driver] submit_command: Starte Matrixmultiplikation...\n"); // Less verbose

            MatrixMultiplyCommandData* cmd_data = (MatrixMultiplyCommandData*)data;

            if (!context || !queue || !cmd_data || !cmd_data->buffer_a || !cmd_data->buffer_b || !cmd_data->buffer_c) {
                fprintf(stderr, "[C Driver][submit_command] Fehler: Kontext (%p), Queue (%p), cmd_data (%p) oder Puffer ungültig!\n", (void*)context, (void*)queue, (void*)cmd_data);
                return 0;
            }

            // Debug: Strukturwerte ausgeben
            // printf("[C Driver][submit_command] cmd_data: size_a=%zu, size_b=%zu, size_c=%zu\n", cmd_data->size_a, cmd_data->size_b, cmd_data->size_c);
            // printf("[C Driver][submit_command] cols_a=%d, cols_b=%d, rows_res=%d, cols_res=%d\n",
            //       cmd_data->cols_a, cmd_data->cols_b, cmd_data->rows_res, cmd_data->cols_res);

            // Host-Puffer für die *Berechnung* allokieren
            double* host_a = (double*)malloc(cmd_data->size_a);
            double* host_b = (double*)malloc(cmd_data->size_b);
            double* host_c = (double*)malloc(cmd_data->size_c); // Ergebnis-Puffer auf Host

            if (!host_a || !host_b || !host_c) {
                fprintf(stderr, "[C Driver][submit_command] Fehler bei malloc für Host-Berechnungspuffer!\n");
                err = CL_OUT_OF_HOST_MEMORY; // Simulate error type
                goto cmd_matmul_hw_sim_cleanup;
            }

            // GPU -> Host lesen (Blocking Reads)
            // printf("[C Driver][submit_command] Lese Puffer A von GPU nach Host...\n");
            err = clEnqueueReadBuffer(queue, (cl_mem)cmd_data->buffer_a, CL_TRUE, 0, cmd_data->size_a, host_a, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "[C Driver][submit_command] Fehler bei clEnqueueReadBuffer A: %s (%d)\n", clGetErrorString(err), err);
                goto cmd_matmul_hw_sim_cleanup;
            }

            // printf("[C Driver][submit_command] Lese Puffer B von GPU nach Host...\n");
            err = clEnqueueReadBuffer(queue, (cl_mem)cmd_data->buffer_b, CL_TRUE, 0, cmd_data->size_b, host_b, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "[C Driver][submit_command] Fehler bei clEnqueueReadBuffer B: %s (%d)\n", clGetErrorString(err), err);
                goto cmd_matmul_hw_sim_cleanup;
            }

            // Matrixmultiplikation (CPU)
            printf("[C Driver][submit_command] Starte CPU-Matrixmultiplikation (Simulation)...\n");
            for (int i = 0; i < cmd_data->rows_res; ++i) {
                for (int j = 0; j < cmd_data->cols_res; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < cmd_data->cols_a; ++k) {
                        // Row-major access
                        sum += host_a[i * cmd_data->cols_a + k] * host_b[k * cmd_data->cols_b + j];
                    }
                    host_c[i * cmd_data->cols_res + j] = sum;
                }
            }
            printf("[C Driver][submit_command] CPU-Matrixmultiplikation abgeschlossen.\n");

            // Host -> GPU schreiben (Blocking Write für das Ergebnis)
            // printf("[C Driver][submit_command] Schreibe Ergebnis-Puffer C von Host nach GPU...\n");
            err = clEnqueueWriteBuffer(queue, (cl_mem)cmd_data->buffer_c, CL_TRUE, 0, cmd_data->size_c, host_c, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "[C Driver][submit_command] Fehler bei clEnqueueWriteBuffer C: %s (%d)\n", clGetErrorString(err), err);
                goto cmd_matmul_hw_sim_cleanup;
            }

            printf("[C Driver][submit_command] Matrixmultiplikations-Simulation erfolgreich abgeschlossen.\n");

        cmd_matmul_hw_sim_cleanup:
            // Temporäre Host-Puffer für Berechnung freigeben
            free(host_a);
            free(host_b);
            free(host_c);
            return (err == CL_SUCCESS); // Erfolg, wenn kein Fehler bei OpenCL
        }

        default:
            fprintf(stderr, "[C Driver][submit_command] Fehler: Unbekannter Befehl %d\n", command);
            return 0;
    }
}


// --- Asynchronous Readback Callback Logic ---

typedef struct {
    int gpu_index;
    // No host pointer needed here anymore
} CallbackUserData;

void CL_CALLBACK readback_callback(cl_event event, cl_int event_command_exec_status, void *user_data) {
    CallbackUserData *cb_data = (CallbackUserData*)user_data;
    int gpu_idx = cb_data ? cb_data->gpu_index : -1;

    printf("[C Driver][Callback GPU %d] Readback event (%p) abgeschlossen. Status: %s (%d)\n",
           gpu_idx, (void*)event, clGetErrorString(event_command_exec_status), event_command_exec_status);

    if (event_command_exec_status != CL_COMPLETE) {
        fprintf(stderr, "[C Driver][Callback GPU %d] Fehler: Read buffer Befehl nicht erfolgreich abgeschlossen.\n", gpu_idx);
    } else {
         printf("[C Driver][Callback GPU %d] Daten lesen in Host-Puffer abgeschlossen.\n", gpu_idx);
         // Data is in the 'result_host_ptr' allocated in simulated_matrix_multiply.
         // simulated_kernel_read will copy it and free it later.
    }

    // --- Cleanup for THIS callback invocation ---
    if (event) {
        clReleaseEvent(event); // Release the event object
    }
    if (cb_data) {
        free(cb_data); // Free the user_data structure
    }
    printf("[C Driver][Callback GPU %d] Callback beendet.\n", gpu_idx);
}

// --- Main Kernel Function (Exported) ---
// Takes "MMAP" addresses (actually host malloc pointers from simulated_kernel_allocate)
__declspec(dllexport)
unsigned long long simulated_matrix_multiply(int gpu_index,
                                             unsigned long long mmap_address_a, // Host ptr from sim_alloc
                                             unsigned long long mmap_address_b, // Host ptr from sim_alloc
                                             size_t size_a, size_t size_b,
                                             int *shape_a, int *shape_b) // Pass shape info
{
    cl_int err = CL_SUCCESS;
    int submit_ok = 0;
    void* result_host_ptr = NULL; // Host buffer for final result readback
    cl_event read_event = NULL;
    CallbackUserData *cb_data = NULL;

    // REAL OpenCL GPU Buffers
    cl_mem buffer_a_gpu = NULL;
    cl_mem buffer_b_gpu = NULL;
    cl_mem buffer_c_gpu = NULL;

    printf("[C Driver] simulated_matrix_multiply gestartet für GPU %d.\n", gpu_index);

    if (!context || !queue) {
        fprintf(stderr, "[C Driver][sim_matmul] Fehler: Kein gültiger Kontext oder Queue vorhanden.\n");
        return 0;
    }
    if (!shape_a || !shape_b) {
        fprintf(stderr, "[C Driver][sim_matmul] Fehler: Ungültige Shapes übergeben (NULL).\n");
        return 0;
    }

    // --- Dimension checks ---
    int rows_a = shape_a[0]; int cols_a = shape_a[1];
    int rows_b = shape_b[0]; int cols_b = shape_b[1];
    printf("[C Driver][sim_matmul] Matrix A Shape: (%d, %d), Size: %zu\n", rows_a, cols_a, size_a);
    printf("[C Driver][sim_matmul] Matrix B Shape: (%d, %d), Size: %zu\n", rows_b, cols_b, size_b);

    if (cols_a != rows_b) {
        fprintf(stderr, "[C Driver][sim_matmul] Fehler: Unvereinbare Matrizenmaße (%dx%d) * (%dx%d)\n", rows_a, cols_a, rows_b, cols_b);
        return 0;
    }
    if (size_a != (size_t)rows_a * cols_a * sizeof(double) || size_b != (size_t)rows_b * cols_b * sizeof(double)) {
         fprintf(stderr, "[C Driver][sim_matmul] Fehler: Übergebene Size stimmt nicht mit Shape*sizeof(double) überein!\n");
         fprintf(stderr, "  A: size=%zu, shape_size=%zu\n", size_a, (size_t)rows_a * cols_a * sizeof(double));
         fprintf(stderr, "  B: size=%zu, shape_size=%zu\n", size_b, (size_t)rows_b * cols_b * sizeof(double));
         return 0;
    }


    int rows_res = rows_a; int cols_res = cols_b;
    size_t result_size = (size_t)rows_res * cols_res * sizeof(double);
    printf("[C Driver][sim_matmul] Ergebnis Matrix C Shape: (%d, %d), Size: %zu\n", rows_res, cols_res, result_size);


    // --- Allocate REAL GPU Buffers ---
    // Use the *real* allocation function here
    buffer_a_gpu = allocate_gpu_memory(gpu_index, size_a);
    buffer_b_gpu = allocate_gpu_memory(gpu_index, size_b);
    buffer_c_gpu = allocate_gpu_memory(gpu_index, result_size);

    if (!buffer_a_gpu || !buffer_b_gpu || !buffer_c_gpu) {
        fprintf(stderr, "[C Driver][sim_matmul] Fehler bei der Allokation von OpenCL GPU Puffern.\n");
        goto matmul_cleanup; // Cleanup partially allocated buffers
    }

    // --- Transfer Data (Host Malloc -> GPU Buffer) ---
    // *** THIS IS WHERE THE ERROR OCCURRED ***
    printf("[C Driver][sim_matmul] Übertrage Daten von Host nach GPU...\n");
    if (!write_to_gpu(gpu_index, buffer_a_gpu, mmap_address_a, 0, size_a)) { // Uses host ptr from sim_alloc
        fprintf(stderr,"[C Driver][sim_matmul] Fehler beim Schreiben von Puffer A zur GPU.\n");
        goto matmul_cleanup;
    }
     if (!write_to_gpu(gpu_index, buffer_b_gpu, mmap_address_b, 0, size_b)) { // Uses host ptr from sim_alloc
        fprintf(stderr,"[C Driver][sim_matmul] Fehler beim Schreiben von Puffer B zur GPU.\n");
        goto matmul_cleanup;
    }
    printf("[C Driver][sim_matmul] Datenübertragung Host->GPU abgeschlossen.\n");


    // --- Prepare and Submit Command (CPU Simulation) ---
    MatrixMultiplyCommandData cmd_data;
    cmd_data.buffer_a = buffer_a_gpu; // Pass REAL cl_mem handles
    cmd_data.buffer_b = buffer_b_gpu;
    cmd_data.buffer_c = buffer_c_gpu;
    cmd_data.size_a = size_a;
    cmd_data.size_b = size_b;
    cmd_data.size_c = result_size;
    cmd_data.cols_a = cols_a;
    cmd_data.cols_b = cols_b;
    cmd_data.rows_res = rows_res;
    cmd_data.cols_res = cols_res;

    printf("[C Driver][sim_matmul] Sende Matrixmultiplikations-Befehl (Simulation)...\n");
    submit_ok = submit_command(gpu_index, COMMAND_MATRIX_MULTIPLY, &cmd_data);
     printf("[C Driver][sim_matmul] Matrixmultiplikations-Befehl abgeschlossen (Status: %d).\n", submit_ok);

    // --- Initiate Asynchronous Read (GPU -> Host) and Set Callback ---
    if (submit_ok) {
        printf("[C Driver][sim_matmul] Starte asynchronen Readback GPU Puffer C -> Host...\n");
        result_host_ptr = malloc(result_size); // Allocate host memory for the final result
        if (!result_host_ptr) {
            fprintf(stderr, "[C Driver][sim_matmul] Fehler: malloc für Host-Ergebnispuffer fehlgeschlagen.\n");
            submit_ok = 0; // Mark failure
            goto matmul_cleanup;
        }
        printf("[C Driver]   Host-Ergebnispuffer allokiert bei %p\n", result_host_ptr);

        // Initiate NON-BLOCKING read
        err = clEnqueueReadBuffer(queue, (cl_mem)buffer_c_gpu, CL_FALSE, // <--- NON-BLOCKING
                                  0, result_size, result_host_ptr,
                                  0, NULL, &read_event); // Get event handle

        if (err != CL_SUCCESS) {
            fprintf(stderr, "[C Driver][sim_matmul] Fehler: clEnqueueReadBuffer (non-blocking) fehlgeschlagen: %s (%d)\n", clGetErrorString(err), err);
            free(result_host_ptr); // Free host buffer if read fails
            result_host_ptr = NULL;
            submit_ok = 0;
            goto matmul_cleanup;
        }
        printf("[C Driver]   Non-blocking Read eingereiht (Event: %p). Setze Callback.\n", (void*)read_event);

        // Prepare user data for the callback
        cb_data = (CallbackUserData*)malloc(sizeof(CallbackUserData));
        if (!cb_data) {
            fprintf(stderr, "[C Driver][sim_matmul] Fehler: malloc für Callback User Data fehlgeschlagen.\n");
            clFinish(queue); // Wait for the read to finish anyway
            free(result_host_ptr); result_host_ptr = NULL;
            if(read_event) clReleaseEvent(read_event); read_event = NULL; // Release event if created
            submit_ok = 0;
            goto matmul_cleanup;
        }
        cb_data->gpu_index = gpu_index;

        // Set the callback
        err = clSetEventCallback(read_event, CL_COMPLETE, readback_callback, cb_data);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[C Driver][sim_matmul] Fehler: clSetEventCallback fehlgeschlagen: %s (%d)\n", clGetErrorString(err), err);
            clFinish(queue); // Wait for pending ops
            free(result_host_ptr); result_host_ptr = NULL;
            free(cb_data); cb_data = NULL; // Free user data as callback won't run
            if(read_event) clReleaseEvent(read_event); read_event = NULL; // Release event
            submit_ok = 0;
            goto matmul_cleanup;
        }
        printf("[C Driver]   Callback registriert für Event %p.\n", (void*)read_event);

        // IMPORTANT Synchronization: Wait for queue to finish before returning to Python
        printf("[C Driver]   Warte auf Beendigung der Command Queue (clFinish)...\n");
        err = clFinish(queue); // This waits for the readback AND its callback to complete
        printf("[C Driver]   Command Queue beendet (Status: %s).\n", clGetErrorString(err));
         if (err != CL_SUCCESS) {
            fprintf(stderr, "[C Driver][sim_matmul] Fehler: clFinish nach Callback-Setup fehlgeschlagen: %s (%d)\n", clGetErrorString(err), err);
            // Callback might or might not have run. result_host_ptr might leak if callback didn't run/free.
            // Mark as failure. Python side won't get a valid address.
            submit_ok = 0;
            // Don't free result_host_ptr here, callback *might* be responsible or simulated_kernel_read might expect it
         }

    } else {
         fprintf(stderr, "[C Driver][sim_matmul] Befehlssimulation fehlgeschlagen. Überspringe Ergebnis-Readback.\n");
    }

matmul_cleanup:
    // --- Cleanup GPU Buffers (always attempt using the *real* free function) ---
    printf("[C Driver][sim_matmul] Bereinige GPU Puffer...\n");
    free_gpu_memory(gpu_index, buffer_a_gpu); // Use the correct free function
    free_gpu_memory(gpu_index, buffer_b_gpu);
    free_gpu_memory(gpu_index, buffer_c_gpu);

    // --- Release Event if callback setup failed ---
    // Callback releases the event on success. If setup failed *after* event creation but *before* callback registration, release here.
    if (read_event != NULL && cb_data == NULL && err != CL_SUCCESS) {
         printf("[C Driver]   Gebe Event Objekt %p frei wegen Callback Setup Fehler.\n", (void*)read_event);
         clReleaseEvent(read_event);
         read_event = NULL;
    }
    // If clFinish failed, the event might already be released by a successful callback, or it might be leaked.

    // --- Return Host Pointer (or 0 on failure) ---
    if (submit_ok && result_host_ptr && err == CL_SUCCESS) { // Ensure clFinish was also OK
        unsigned long long result_address_host = (unsigned long long)result_host_ptr;
        printf("[C Driver][sim_matmul] Matrixmultiplikation erfolgreich. Gebe Host-Adresse zurück: %llu (%p)\n", result_address_host, result_host_ptr);
        return result_address_host; // Return the pointer to the host buffer holding the result
    } else {
         fprintf(stderr, "[C Driver][sim_matmul] Matrixmultiplikation fehlgeschlagen. Gebe 0 zurück.\n");
         // Free host pointer if it was allocated but something failed *before* successful callback registration OR clFinish failed
         if (result_host_ptr && (cb_data == NULL || err != CL_SUCCESS) ) {
              printf("[C Driver]   Gebe Host Ergebnispuffer %p frei wegen Fehler.\n", result_host_ptr);
              free(result_host_ptr);
              result_host_ptr = NULL;
         } else if (result_host_ptr && cb_data != NULL && err == CL_SUCCESS) {
             // Should not happen - if submit_ok=1 and err=CL_SUCCESS, we should return the pointer.
             // If submit_ok=0 but cb_data exists and err=CL_SUCCESS (e.g. submit_command failed), rely on callback? Risky.
             fprintf(stderr, "[C Driver]   Warnung: Unerwarteter Zustand am Ende von sim_matmul. Host Puffer %p könnte lecken.\n", result_host_ptr);
         }
         return 0; // Indicate failure
    }
}

// --- Kernel Read (Host Pointer -> Python Buffer) ---
__declspec(dllexport)
void simulated_kernel_read(int gpu_index, unsigned long long host_address, size_t size, void *destination) {
    if (host_address == 0 || destination == NULL || size == 0) {
        fprintf(stderr, "[C Driver][simulated_kernel_read] Fehler: Ungültige Argumente (host_addr=%llu, dest=%p, size=%zu)\n", host_address, destination, size);
        return;
    }
    void* host_ptr = (void*)host_address; // Pointer returned by simulated_matrix_multiply
    printf("[C Driver][simulated_kernel_read] Kopiere %zu Bytes von Host Addr %llu (%p) nach Python Dest %p\n", size, host_address, host_ptr, destination);
    memcpy(destination, host_ptr, size);

    // Free the host buffer that was allocated in simulated_matrix_multiply for the readback
    printf("[C Driver]   Gebe Host Ergebnispuffer frei bei %p\n", host_ptr);
    free(host_ptr);
    printf("[C Driver]   Lesen/Kopieren von Host abgeschlossen.\n");
}

// --- Shutdown ---
__declspec(dllexport)
void shutdown_driver() {
    printf("[C Driver] shutdown_driver: Starte Cleanup...\n");
    if (queue) {
        clFinish(queue); // Ensure all commands are done
        clReleaseCommandQueue(queue);
        queue = NULL;
        printf("[C Driver]   Command Queue freigegeben.\n");
    }
    if (context) {
        clReleaseContext(context);
        context = NULL;
        printf("[C Driver]   Context freigegeben.\n");
    }
    // Release platform/device IDs? Not usually necessary.

    // MMAP cleanup (if it were used)
    if (simulated_mmio_base && simulated_mmio_base != MAP_FAILED) {
        #ifdef __linux__
            munmap(simulated_mmio_base, SIMULATED_MMIO_SIZE);
        #endif
        simulated_mmio_base = NULL;
        printf("[C Driver]   Simuliertes MMIO freigegeben.\n");
    }
    printf("[C Driver] shutdown_driver: Cleanup abgeschlossen.\n");
}

// --- Host Malloc/Free Simulation (Called by Python) ---
__declspec(dllexport)
unsigned long long simulated_kernel_allocate(int gpu_index, size_t size) {
    if (size == 0) {
        printf("[C Driver][simulated_kernel_allocate] Warnung: Allokation von Größe 0 angefordert.\n");
        return 0;
    }
    // Simulate allocation by just using host malloc
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "[C Driver][simulated_kernel_allocate] Fehler: malloc fehlgeschlagen für %zu bytes\n", size);
        return 0;
    }
    printf("[C Driver][simulated_kernel_allocate] GPU %d - Simuliert (malloc) %zu bytes bei %p\n", gpu_index, size, ptr);
    return (unsigned long long)ptr; // Return host pointer cast to integer
}

__declspec(dllexport)
void simulated_kernel_free(int gpu_index, unsigned long long address, size_t size) {
    if (address == 0) return;
    void* ptr = (void*)address;
    printf("[C Driver][simulated_kernel_free] GPU %d - Simuliert (free) %zu bytes bei %p\n", gpu_index, size, ptr);
    free(ptr); // Free the host pointer
}

// Add dllexport if needed by Python directly, otherwise keep static
unsigned int get_compute_unit_count(int gpu_index) {
    if (!device_id) return 0; // Ensure device_id is valid

    cl_uint cu_count = 0;
    cl_int err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &cu_count, NULL);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "[C Driver] Fehler bei clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS): %s (%d)\n", clGetErrorString(err), err);
        return 0; // Return 0 on error
    }
    printf("[C Driver] get_compute_unit_count für GPU %d: %u CUs\n", gpu_index, cu_count);
    return (unsigned int)cu_count;
}

// Export for Python if you want to call it directly
__declspec(dllexport)
unsigned int simulated_get_compute_unit_count(int gpu_index) {
    // If OpenCL is initialized, try to get the real value
    if (device_id) {
        return get_compute_unit_count(gpu_index);
    }
    // Fallback to a fixed value if OpenCL isn't ready or failed
    printf("[C Driver] simulated_get_compute_unit_count: OpenCL nicht bereit, gebe festen Wert 2560 zurück.\n");
    return 2560; // Your original fixed value
}