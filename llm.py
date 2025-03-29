import sys
import math
import ctypes
import numpy as np
import os
import time
import weakref
import pickle  # Wird für das Speichern von Python-Objekten wie dicts benötigt
from ctypes import c_void_p, c_int, c_size_t, c_float, CDLL

# --- Update Version String ---
print("--- Starting OCL LLM Framework [Ultimate Build Attempt 7 - PE GPU Add] ---")

# --------------------------------------------------------------------------
# 1. OpenCL-DLL binden & Globale Variablen
# --------------------------------------------------------------------------
try:
    # Finde die DLL im gleichen Verzeichnis wie das Skript oder im aktuellen Arbeitsverzeichnis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dll_filename = "driver.dll"  # Name deiner kompilierten DLL
    dll_path = os.path.join(script_dir, dll_filename)
    if not os.path.exists(dll_path):
        print(f"WARN: DLL not found in script directory ({script_dir}), checking current directory...")
        dll_path = os.path.abspath(dll_filename)  # Suche im aktuellen Verzeichnis
    if not os.path.exists(dll_path):
        raise FileNotFoundError(
            f"DLL not found at specified paths: {os.path.join(script_dir, dll_filename)} or {os.path.abspath(dll_filename)}"
        )
    ocl = CDLL(dll_path)
    print(f"[ocl_framework] Successfully loaded: {dll_path}")
except OSError as e:
    print(f"[ocl_framework] FATAL: Error loading '{dll_filename}': {e}")
    print("  Ensure the DLL exists and all dependencies (like OpenCL runtime) are installed and accessible.")
    raise
except FileNotFoundError as e:
    print(f"[ocl_framework] FATAL: {e}")
    raise

# --- Globale Typ- und Geräte-Konfiguration ---
FP_TYPE = np.float32
FP_SIZE = np.dtype(FP_TYPE).itemsize
ADAM_STATE_TYPE = np.float32  # Typ für Adam m/v Zustände (oft float32)
ADAM_STATE_SIZE = np.dtype(ADAM_STATE_TYPE).itemsize
GPU_ID = 0  # Standard GPU Index (wird durch Kommandozeilenargument überschrieben)

# --- Globale Flags für Kernel-Verfügbarkeit (werden in ocl_initialize gesetzt) ---
HAS_BMM_BATCHED = False
HAS_TRANSPOSE_LAST_TWO = False
HAS_TRANSPOSE_12_BATCHED = False
HAS_REDUCE_SUM = False           # Flag für Bias-Gradienten-Reduktion
HAS_ADD_BROADCAST_PE = False     # Flag für dedizierten PE Add Kernel
HAS_EMBEDDING_LOOKUP = False     # Flag für GPU Embedding Lookup
HAS_EMBEDDING_BACKWARD = False   # Flag für GPU Embedding Backward (mit Atomics)
# HAS_BROADCAST_ADD_BIAS = False  # Flag für allgemeinen Broadcast Add Kernel (optional)

# --------------------------------------------------------------------------
# 2. Signaturen für alle C-Funktionen definieren
# --------------------------------------------------------------------------
try:
    # Init / Shutdown / Mem
    ocl.initialize_gpu.argtypes = [c_int]
    ocl.initialize_gpu.restype = c_int
    ocl.shutdown_driver.restype = None
    ocl.allocate_gpu_memory.argtypes = [c_int, c_size_t]
    ocl.allocate_gpu_memory.restype = c_void_p
    ocl.free_gpu_memory.argtypes = [c_int, c_void_p]
    ocl.write_host_to_gpu_blocking.argtypes = [c_int, c_void_p, c_size_t, c_size_t, c_void_p]
    ocl.write_host_to_gpu_blocking.restype = c_int
    ocl.read_gpu_to_host_blocking.argtypes = [c_int, c_void_p, c_size_t, c_size_t, c_void_p]
    ocl.read_gpu_to_host_blocking.restype = c_int
    ocl.execute_clone_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_size_t]
    ocl.execute_clone_on_gpu.restype = c_int

    # --- Kern Operationen Signaturen ---
    ocl.execute_matmul_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    ocl.execute_matmul_on_gpu.restype = c_int
    ocl.execute_matmul_backward_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    ocl.execute_matmul_backward_on_gpu.restype = c_int

    ocl.execute_add_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_int]
    ocl.execute_add_on_gpu.restype = c_int
    ocl.execute_gelu_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_int]
    ocl.execute_gelu_on_gpu.restype = c_int
    ocl.execute_gelu_backward_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_int]
    ocl.execute_gelu_backward_on_gpu.restype = c_int
    ocl.execute_layernorm_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_int, c_int, c_float]
    ocl.execute_layernorm_on_gpu.restype = c_int
    ocl.execute_layernorm_backward_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_int, c_int, c_float]
    ocl.execute_layernorm_backward_on_gpu.restype = c_int
    ocl.execute_softmax_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_int, c_int]
    ocl.execute_softmax_on_gpu.restype = c_int
    ocl.execute_softmax_backward_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_int, c_int]
    ocl.execute_softmax_backward_on_gpu.restype = c_int
    ocl.execute_mul_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_int]
    ocl.execute_mul_on_gpu.restype = c_int
    ocl.execute_mul_backward_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int]
    ocl.execute_mul_backward_on_gpu.restype = c_int
    ocl.execute_transpose_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_int, c_int]
    ocl.execute_transpose_on_gpu.restype = c_int  # Basic 2D
    ocl.execute_transpose_backward_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_int, c_int]
    ocl.execute_transpose_backward_on_gpu.restype = c_int  # Basic 2D backward
    ocl.execute_adam_update_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_float, c_float, c_float, c_float, c_float]
    ocl.execute_adam_update_on_gpu.restype = c_int

    # --- Optionale / Spezialisierte Kernel Signaturen (mit Check & Flag Setting) ---
    if hasattr(ocl, 'execute_matmul_batched_on_gpu') and hasattr(ocl, 'execute_matmul_batched_backward_on_gpu'):
        ocl.execute_matmul_batched_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        ocl.execute_matmul_batched_on_gpu.restype = c_int
        ocl.execute_matmul_batched_backward_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        ocl.execute_matmul_batched_backward_on_gpu.restype = c_int
        HAS_BMM_BATCHED = True

    if hasattr(ocl, 'execute_embedding_lookup_gpu'):
        ocl.execute_embedding_lookup_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        ocl.execute_embedding_lookup_gpu.restype = c_int
        HAS_EMBEDDING_LOOKUP = True
    if hasattr(ocl, 'execute_embedding_backward_gpu'):
        ocl.execute_embedding_backward_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        ocl.execute_embedding_backward_gpu.restype = c_int
        HAS_EMBEDDING_BACKWARD = True

    if hasattr(ocl, 'execute_reduce_sum_gpu'):
        ocl.execute_reduce_sum_gpu.argtypes = [c_int, c_void_p, c_void_p, c_int, c_int, c_int]  # B, M, N
        ocl.execute_reduce_sum_gpu.restype = c_int
        HAS_REDUCE_SUM = True

    # NEU: Signatur für AddBroadcastPE
    if hasattr(ocl, 'execute_add_broadcast_pe_gpu'):
        ocl.execute_add_broadcast_pe_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]  # B, S, E
        ocl.execute_add_broadcast_pe_gpu.restype = c_int
        HAS_ADD_BROADCAST_PE = True

    if hasattr(ocl, 'execute_transpose_batched_gpu'):
        ocl.execute_transpose_batched_gpu.argtypes = [c_int, c_void_p, c_void_p, c_int, c_int, c_int]  # B_flat, d1, d2
        ocl.execute_transpose_batched_gpu.restype = c_int
        HAS_TRANSPOSE_LAST_TWO = True

    if hasattr(ocl, 'execute_transpose_12_batched_gpu'):
        ocl.execute_transpose_12_batched_gpu.argtypes = [c_int, c_void_p, c_void_p, c_int, c_int, c_int, c_int]  # B, D1, D2, D3
        ocl.execute_transpose_12_batched_gpu.restype = c_int
        HAS_TRANSPOSE_12_BATCHED = True

except AttributeError as e:
    print(f"[ocl_framework] FATAL: Sig Error: Function signature mismatch or missing function in DLL.")
    print(f"  Missing/Mismatch: {e}")
    print(f"  Please ensure the DLL '{dll_filename}' is compiled correctly and matches the expected interface.")
    raise

# --------------------------------------------------------------------------
# 3. Hilfsfunktionen (unverändert)
# --------------------------------------------------------------------------
def check_success(result_code, operation_name=""):
    """Prüft, ob ein C-Funktionsaufruf erfolgreich war (Rückgabewert 1)."""
    if result_code != 1:
        raise RuntimeError(f"OpenCL op '{operation_name}' failed (Code: {result_code})")

# --------------------------------------------------------------------------
# 4. GPU Puffer Klasse (unverändert)
# --------------------------------------------------------------------------
class GPUBuffer:
    """Wrapper class for OpenCL memory buffers."""
    def __init__(self, size_in_bytes, tag="buffer", _init_zeros=False, dtype=FP_TYPE):
        self.size = size_in_bytes
        self.ptr = None  # Will hold the c_void_p from OpenCL
        self.tag = tag  # Identifier for debugging
        self._allocated = False

        if size_in_bytes > 0:
            try:
                self.ptr = ocl.allocate_gpu_memory(GPU_ID, self.size)
            except Exception as e:
                # Catch potential errors during the C call itself
                raise MemoryError(f"GPU allocation call failed for '{tag}' (size {self.size}): {e}")

            if not self.ptr:
                # Check if the returned pointer is NULL
                raise MemoryError(f"GPU allocation failed for {self.size} bytes ('{tag}'). OpenCL returned NULL.")
            self._allocated = True  # Mark as successfully allocated

            # Optional: Initialize buffer with zeros on GPU
            if _init_zeros:
                # Calculate number of elements based on specified dtype
                itemsize = np.dtype(dtype).itemsize
                if itemsize == 0:
                    raise ValueError("dtype itemsize cannot be zero")
                num_elements = size_in_bytes // itemsize
                # Check for potential size mismatch if not perfectly divisible
                if num_elements * itemsize != size_in_bytes:
                    print(f"WARN GPUBuffer InitZeros: Size {size_in_bytes} not divisible by dtype size {itemsize} for '{tag}'. May not zero completely.")

                if num_elements > 0:
                    # Use a temporary host array for zeroing (efficient for larger buffers)
                    zeros_host = np.zeros(num_elements, dtype=dtype)
                    self.write(zeros_host)  # Use the write method to transfer zeros
                    del zeros_host  # Allow garbage collection
        # else: size is 0, ptr remains None, _allocated remains False

    def write(self, host_data: np.ndarray):
        """Writes data from a NumPy array (host) to the GPU buffer."""
        if not self._allocated:
            raise RuntimeError(f"Cannot write to buffer '{self.tag}': freed or not allocated")
        nbytes = host_data.nbytes
        if nbytes > self.size:
            raise ValueError(f"Data size {nbytes} exceeds buffer size {self.size} for '{self.tag}'")
        if nbytes > 0:  # Only call OpenCL if there's data to write
            check_success(
                ocl.write_host_to_gpu_blocking(
                    GPU_ID,
                    self.ptr,
                    0,  # offset
                    nbytes,
                    host_data.ctypes.data_as(c_void_p)  # Pointer to numpy data
                ),
                f"write '{self.tag}'"
            )

    def read(self, shape, dtype=FP_TYPE) -> np.ndarray:
        """Reads data from the GPU buffer into a NumPy array (host)."""
        if not self._allocated:
            raise RuntimeError(f"Cannot read from buffer '{self.tag}': freed or not allocated")
        num_elements = np.prod(shape, dtype=np.int64)
        itemsize = np.dtype(dtype).itemsize
        if itemsize == 0:
            raise ValueError("dtype itemsize cannot be zero")
        expected_nbytes = num_elements * itemsize
        if expected_nbytes > self.size:
            raise ValueError(f"Requested read size {expected_nbytes} (shape {shape}) exceeds buffer size {self.size} for '{self.tag}'")

        # Create an empty NumPy array on the host with the target shape and dtype
        host_data = np.empty(shape, dtype=dtype)
        nbytes = host_data.nbytes  # Actual bytes to read based on host_data

        if nbytes > 0:  # Only call OpenCL if there's data to read
            check_success(
                ocl.read_gpu_to_host_blocking(
                    GPU_ID,
                    self.ptr,
                    0,  # offset
                    nbytes,
                    host_data.ctypes.data_as(c_void_p)  # Pointer to numpy data destination
                ),
                f"read '{self.tag}'"
            )
        return host_data

    def clone(self) -> "GPUBuffer":
        """Creates a new GPUBuffer with the same size and copies the content."""
        if not self._allocated:
            raise RuntimeError(f"Cannot clone freed or unallocated buffer '{self.tag}'")
        # Create a new buffer of the same size
        new_buffer = GPUBuffer(self.size, tag=f"clone_{self.tag}")
        # Copy data from the original buffer to the new one on the GPU
        if self.ptr and new_buffer.ptr and self.size > 0:
            check_success(
                ocl.execute_clone_on_gpu(GPU_ID, self.ptr, new_buffer.ptr, self.size),
                f"clone '{self.tag}'"
            )
        return new_buffer

    def free(self):
        """Releases the allocated GPU memory."""
        if self._allocated and self.ptr:
            try:
                # Call the C function to free the memory
                ocl.free_gpu_memory(GPU_ID, self.ptr)
            except Exception as e:
                # Catch potential errors during the C call, but don't crash
                print(f"Warning: Exception during GPUBuffer free for '{self.tag}': {e}")
            # Mark as freed regardless of C call success to prevent double free attempts
            self.ptr = None
            self._allocated = False
        # else: Buffer was already freed or never allocated

    def __del__(self):
        """Destructor attempts to free GPU memory if not already freed."""
        # Check necessary conditions carefully to avoid errors during interpreter shutdown
        if hasattr(OclTensor, '_ocl_initialized') and OclTensor._ocl_initialized and \
           hasattr(self, '_allocated') and self._allocated and \
           hasattr(self, 'ptr') and self.ptr:
            try:
                # Check if the 'ocl' object and the function still exist
                if hasattr(ocl, 'free_gpu_memory'):
                    self.free()  # Use the existing free method
            except Exception as e:
                # Suppress exceptions during __del__ as they can cause issues
                pass  # print(f"Warning: Exception suppressed in GPUBuffer.__del__ for {self.tag}: {e}")

# --------------------------------------------------------------------------
# 5. Tensor-Klasse (OclTensor) mit Operationen und Autograd
# --------------------------------------------------------------------------
class OclTensor:
    """GPU-accelerated tensor class with automatic differentiation support."""
    _ocl_initialized = False  # Class variable to track OpenCL initialization
    _enable_grad = True       # Class variable to enable/disable gradient tracking globally

    class NoGradContext:
        """Context manager to temporarily disable gradient calculation."""
        def __enter__(self):
            self.prev = OclTensor._enable_grad  # Store previous state
            OclTensor._enable_grad = False      # Disable grad
        def __exit__(self, *args):
            OclTensor._enable_grad = self.prev   # Restore previous state

    @classmethod
    def no_grad(cls):
        """Returns a context manager that disables gradient calculations."""
        return cls.NoGradContext()

    def __init__(self, data: np.ndarray, requires_grad=False, _gpu_buffer=None, _shape=None):
        """
        Initializes an OclTensor.

        Args:
            data (np.ndarray): Initial data from host (can be empty if _gpu_buffer is provided).
            requires_grad (bool): If True, enables gradient tracking for this tensor.
            _gpu_buffer (GPUBuffer, optional): An existing GPUBuffer to use. If None, creates a new one.
            _shape (tuple, optional): The desired shape. If None, inferred from `data`.
                                       Must be provided if `data` is empty and _gpu_buffer is used.
        """
        if not OclTensor._ocl_initialized:
            raise RuntimeError("OpenCL not initialized. Call ocl_initialize() first.")

        # Determine shape and calculate size
        current_shape = tuple(_shape if _shape is not None else data.shape)
        self.shape = current_shape
        self.numel = np.prod(self.shape, dtype=np.int64)  # Use int64 for large tensors
        self.nbytes = self.numel * FP_SIZE

        # Manage GPU buffer
        if _gpu_buffer:  # Use provided buffer
            if not isinstance(_gpu_buffer, GPUBuffer):
                raise TypeError("_gpu_buffer must be a GPUBuffer instance")
            if _gpu_buffer.size != self.nbytes:
                raise ValueError(f"Provided GPU buffer size ({_gpu_buffer.size}) does not match required size ({self.nbytes}) for shape {self.shape}")
            self.data = _gpu_buffer
        else:  # Create a new buffer
            self.data = GPUBuffer(self.nbytes, f"tensor_{id(self)}")
            # Write initial data if provided and buffer allocated successfully
            if data.size > 0 and self.data._allocated:
                # Ensure data type matches FP_TYPE before checking size and writing
                if data.dtype != FP_TYPE:
                    data = data.astype(FP_TYPE)
                if data.nbytes != self.nbytes:
                    raise ValueError(f"Numpy data size ({data.nbytes}) mismatch after type conversion (expected {self.nbytes}).")
                self.data.write(data)
            elif data.size > 0 and not self.data._allocated:
                raise MemoryError(f"Failed to allocate GPU buffer, cannot write initial data for tensor {id(self)}")
            # No write needed if data.size is 0

        # Gradient tracking setup
        self.requires_grad = requires_grad and OclTensor._enable_grad  # Respect global flag
        self._grad_buffer = None  # GPUBuffer for the gradient (lazily allocated)
        self.grad = None          # OclTensor view of the gradient (lazily created)
        self._ctx = None          # Context for backward pass (stores operation info)

        register_tensor(self)  # Register for cleanup tracking

    @staticmethod
    def empty(shape, requires_grad=False) -> "OclTensor":
        """Creates an OclTensor with uninitialized data on the GPU."""
        nbytes = np.prod(shape, dtype=np.int64) * FP_SIZE
        gpu_buf = GPUBuffer(nbytes, "tensor_empty")
        # Pass empty numpy array, shape is determined by _shape argument
        return OclTensor(np.empty(0, dtype=FP_TYPE), requires_grad=requires_grad, _gpu_buffer=gpu_buf, _shape=shape)

    @staticmethod
    def zeros(shape, requires_grad=False) -> "OclTensor":
        """Creates an OclTensor initialized with zeros on the GPU."""
        nbytes = np.prod(shape, dtype=np.int64) * FP_SIZE
        # GPUBuffer handles zero initialization efficiently
        gpu_buf = GPUBuffer(nbytes, "tensor_zeros", _init_zeros=True, dtype=FP_TYPE)
        # Pass empty numpy array, shape is determined by _shape argument
        return OclTensor(np.empty(0, dtype=FP_TYPE), requires_grad=requires_grad, _gpu_buffer=gpu_buf, _shape=shape)

    def _ensure_grad_buffer(self):
        """Allocates the gradient buffer on GPU if needed and initializes it with zeros."""
        if self.requires_grad and self._grad_buffer is None:
            if self.nbytes > 0:
                self._grad_buffer = GPUBuffer(self.nbytes, f"grad_{id(self)}", _init_zeros=True, dtype=FP_TYPE)
            # else: No need for buffer if tensor has 0 elements

    def get_grad_tensor(self) -> "OclTensor | None":
        """Returns the gradient as an OclTensor, creating it if necessary."""
        if not self.requires_grad:
            return None
        self._ensure_grad_buffer()  # Ensure the GPUBuffer exists
        if self._grad_buffer is None or not self._grad_buffer._allocated:
            return None  # Buffer might be freed or failed alloc
        # Create the OclTensor view for the gradient buffer if it doesn't exist yet
        if self.grad is None:
            self.grad = OclTensor(np.empty(0, dtype=FP_TYPE), requires_grad=False, _gpu_buffer=self._grad_buffer, _shape=self.shape)
        return self.grad

    def zero_grad(self):
        """Resets the gradient of this tensor to zeros."""
        if self.requires_grad and self._grad_buffer:
            # Option 1: Reallocate (ensures clean state)
            self._grad_buffer.free()
            self._grad_buffer = GPUBuffer(self.nbytes, f"grad_{id(self)}", _init_zeros=True, dtype=FP_TYPE)
            # Option 2: Overwrite with zeros (might be faster if buffer reuse is preferred)
            # if self.nbytes > 0:
            #    zeros_host = np.zeros(self.numel, dtype=FP_TYPE)
            #    self._grad_buffer.write(zeros_host)
            self.grad = None  # Reset the OclTensor view

    def _accumulate_grad(self, incoming_grad: "OclTensor"):
        """Adds the incoming gradient to the tensor's current gradient."""
        if not self.requires_grad or self.numel == 0:
            return  # Nothing to do
        if incoming_grad.shape != self.shape:
            raise ValueError(f"Gradient shape mismatch: Expected {self.shape}, got {incoming_grad.shape}")

        self._ensure_grad_buffer()  # Ensure self._grad_buffer exists and is zeroed if new
        if self._grad_buffer is None or not self._grad_buffer._allocated:
            print(f"WARN _accumulate_grad: Grad buffer for tensor {id(self)} invalid, cannot accumulate.")
            return
        if incoming_grad.data is None or not incoming_grad.data._allocated:
            print(f"WARN _accumulate_grad: Incoming gradient tensor {id(incoming_grad)} buffer invalid.")
            return

        # Perform inplace accumulation: current_grad = current_grad + incoming_grad
        temp_res_buf = None
        try:
            # Need a temporary buffer for the result of the addition
            temp_res_buf = GPUBuffer(self.nbytes, f"accum_grad_temp_{id(self)}")
            if not temp_res_buf.ptr:
                raise MemoryError("Failed to alloc temp buffer for grad accum")

            # Add current gradient (self._grad_buffer) and incoming gradient
            check_success(
                ocl.execute_add_on_gpu(GPU_ID, self._grad_buffer.ptr, incoming_grad.data.ptr, temp_res_buf.ptr, self.numel),
                "accum_grad_add"
            )

            # Copy the result back to the primary gradient buffer
            check_success(
                ocl.execute_clone_on_gpu(GPU_ID, temp_res_buf.ptr, self._grad_buffer.ptr, self.nbytes),
                "accum_grad_copyback"
            )
        finally:
            if temp_res_buf:
                temp_res_buf.free()

        self.grad = None  # Reset OclTensor view as buffer content changed

    def to_host(self) -> np.ndarray:
        """Copies tensor data from GPU to a NumPy array on the host."""
        if not self.data or not self.data._allocated:
            raise RuntimeError("Tensor data buffer is not allocated or has been freed.")
        return self.data.read(self.shape, dtype=FP_TYPE)

    def detach(self):
        """Creates a new OclTensor that shares the data buffer but is detached from the computation graph."""
        return OclTensor(np.empty(0, dtype=FP_TYPE), requires_grad=False, _gpu_buffer=self.data, _shape=self.shape)

    def clone(self) -> "OclTensor":
        """Creates a new OclTensor with a separate copy of the data and gradient buffer on the GPU."""
        if not self.data or not self.data._allocated:
            raise RuntimeError("Cannot clone tensor with unallocated data.")
        new_data = self.data.clone()  # Clones the GPUBuffer for data
        # Create new tensor instance using the cloned data buffer
        new_tensor = OclTensor(np.empty(0, dtype=FP_TYPE), requires_grad=self.requires_grad, _gpu_buffer=new_data, _shape=self.shape)
        # Clone gradient buffer if it exists and grad is required
        if self.requires_grad:
            self._ensure_grad_buffer()  # Make sure original grad buffer exists if needed
            if self._grad_buffer and self._grad_buffer._allocated:
                new_tensor._grad_buffer = self._grad_buffer.clone()
            # new_tensor.grad remains None, will be lazily created if accessed
        return new_tensor

    def backward(self, gradient=None):
        """
        Initiates the backward pass to compute gradients for this tensor and its ancestors.

        Args:
            gradient (OclTensor | np.ndarray | None): The gradient of the loss with respect to this tensor.
                If None (for scalar tensors), assumes a gradient of 1.0.
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on tensor that does not require gradients.")
        if not OclTensor._enable_grad:
            return  # Respect NoGrad context

        # --- Build backward graph (Topological Sort) ---
        visited, nodes_in_pass = set(), []

        def build_graph(node):
            if node is None or id(node) in visited or not isinstance(node, OclTensor):
                return
            visited.add(id(node))  # Mark visited
            # Only explore parents and add to pass if the node requires grad
            if node.requires_grad:
                if hasattr(node, '_ctx') and node._ctx and hasattr(node._ctx, 'parents'):
                    for parent_ref in node._ctx.parents:
                        parent = parent_ref()
                        if isinstance(parent, OclTensor):
                            build_graph(parent)  # Recurse
                nodes_in_pass.append(node)  # Add node to the processing list

        build_graph(self)
        # --- End Build Graph ---

        # --- Initialize gradient accumulation for the starting tensor ---
        gradient_tensor_to_accum = None
        temp_gradient_created = False
        if gradient is None:  # Default gradient for scalar output
            if self.numel != 1:
                raise RuntimeError("Gradient can only be implicitly created for scalar outputs")
            gradient_tensor_to_accum = OclTensor(np.ones(self.shape, dtype=FP_TYPE))
            temp_gradient_created = True
        elif not isinstance(gradient, OclTensor):  # Convert numpy/scalar to OclTensor
            gradient_tensor_to_accum = OclTensor(np.asarray(gradient, dtype=FP_TYPE))
            temp_gradient_created = True
        else:  # Use provided OclTensor gradient
            gradient_tensor_to_accum = gradient

        # Check shape compatibility
        if self.shape != gradient_tensor_to_accum.shape:
            if temp_gradient_created:
                gradient_tensor_to_accum.free_memory()  # Cleanup temp tensor
            raise ValueError(f"Gradient shape mismatch: Expected {self.shape}, got {gradient_tensor_to_accum.shape}")

        # Accumulate the initial gradient (dL/dself)
        self._accumulate_grad(gradient_tensor_to_accum)

        # Free temporary gradient tensor if created
        if temp_gradient_created:
            gradient_tensor_to_accum.free_memory()
        # --- End Init Gradient ---

        # --- Execute backward pass through the graph ---
        # Process nodes in reverse topological order
        for node in reversed(nodes_in_pass):
            if node.requires_grad and node._ctx:  # Ensure node needs grad and has context
                grad_tensor = node.get_grad_tensor()  # Get OclTensor view of its gradient
                # Check if gradient tensor and its buffer are valid
                if grad_tensor and grad_tensor.data and grad_tensor.data.ptr:
                    try:
                        # Execute the backward function defined in the node's context
                        node._ctx.backward(grad_tensor)
                    except Exception as e:
                        print(f"!!! ERROR during backward pass for {node._ctx.__class__.__name__} (Tensor {id(node)}) !!!")
                        import traceback
                        traceback.print_exc()
                        raise  # Re-raise after printing details
        # --- End Backward Pass ---

    # --- Operation Implementations ---
    def matmul(self, other: "OclTensor") -> "OclTensor":
        # (Implementation unverändert, siehe vorherige Antwort)
        if not isinstance(other, OclTensor):
            return NotImplemented
        a_shape = self.shape
        b_shape = other.shape
        a_ndim = len(a_shape)
        b_ndim = len(b_shape)
        B, M, K, N = 0, 0, 0, 0
        use_batched_kernel, use_standard_kernel = False, False
        matmul_func, backward_context_class = None, None
        K_other = 0

        if a_ndim == 2 and b_ndim == 2:
            M, K = a_shape
            K_other, N = b_shape
            B = 1
            use_standard_kernel = True
            backward_context_class = MatMulBackwardContext
        elif a_ndim == 3 and b_ndim == 2:
            B, M, K = a_shape
            K_other, N = b_shape
            use_standard_kernel = True
            backward_context_class = MatMulBackwardContext
        elif a_ndim == 3 and b_ndim == 3:
            B_a, M, K = a_shape
            B_b, K_other, N = b_shape
            if B_a != B_b:
                raise ValueError(f"Matmul batch mismatch: {a_shape} @ {b_shape}")
            B = B_a
            if HAS_BMM_BATCHED:
                use_batched_kernel = True
                backward_context_class = MatMulBatchedBackwardContext
            else:
                # Fallback to CPU loop
                print(f">> WARNING: execute_matmul_batched_on_gpu missing! CPU loop for {a_shape} @ {b_shape} <<")
                req_grad = (self.requires_grad or other.requires_grad) and OclTensor._enable_grad
                out_shape = (B, M, N)
                out = OclTensor.empty(out_shape, req_grad)
                sl_a = np.empty((M, K), FP_TYPE)
                sl_b = np.empty((K, N), FP_TYPE)
                sl_c = np.empty((M, N), FP_TYPE)
                tmp_a = OclTensor.empty((M, K))
                tmp_b = OclTensor.empty((K, N))
                tmp_c = OclTensor.empty((M, N))
                a_sn = M * K * FP_SIZE
                b_sn = K * N * FP_SIZE
                c_sn = M * N * FP_SIZE
                try:
                    for i in range(B):
                        check_success(ocl.read_gpu_to_host_blocking(GPU_ID, self.data.ptr, i * a_sn, a_sn, sl_a.ctypes.data), 'wa_ra')
                        tmp_a.data.write(sl_a)
                        check_success(ocl.read_gpu_to_host_blocking(GPU_ID, other.data.ptr, i * b_sn, b_sn, sl_b.ctypes.data), 'wa_rb')
                        tmp_b.data.write(sl_b)
                        check_success(ocl.execute_matmul_on_gpu(GPU_ID, tmp_a.data.ptr, tmp_b.data.ptr, tmp_c.data.ptr, 1, M, N, K), 'wa_k')
                        check_success(ocl.read_gpu_to_host_blocking(GPU_ID, tmp_c.data.ptr, 0, c_sn, sl_c.ctypes.data), 'wa_rc')
                        check_success(ocl.write_host_to_gpu_blocking(GPU_ID, out.data.ptr, i * c_sn, c_sn, sl_c.ctypes.data), 'wa_wc')
                    if out.requires_grad:
                        print("ERROR: BW for CPU loop BMM missing.")
                        new_out = out.detach()
                        out.free_memory()
                        out = new_out
                finally:
                    tmp_a.free_memory()
                    tmp_b.free_memory()
                    tmp_c.free_memory()
                return out
        else:
            raise ValueError(f"Matmul unsupported shapes: {a_shape} @ {b_shape}")
        if K != K_other:
            raise ValueError(f"Matmul inner K mismatch: {a_shape} @ {b_shape}")

        out_shape = (B, M, N) if B > 1 or a_ndim == 3 else (M, N)
        req_grad = (self.requires_grad or other.requires_grad) and OclTensor._enable_grad
        out = OclTensor.empty(out_shape, req_grad)
        if self.numel > 0 and other.numel > 0 and out.numel > 0:
            matmul_func_c = ocl.execute_matmul_batched_on_gpu if use_batched_kernel else ocl.execute_matmul_on_gpu
            check_success(matmul_func_c(GPU_ID, self.data.ptr, other.data.ptr, out.data.ptr, B, M, N, K), "matmul gpu")
        if out.requires_grad and backward_context_class:
            out._ctx = backward_context_class(self, other)
        return out

    def add(self, other: "OclTensor") -> "OclTensor":
        # (Implementation unverändert, siehe vorherige Antwort)
        if not isinstance(other, OclTensor):
            if isinstance(other, (int, float)):  # Allow adding python scalars
                scalar_tensor = None
                try:
                    scalar_tensor = OclTensor(np.full(self.shape, float(other), dtype=FP_TYPE))
                    return self.add(scalar_tensor)  # Reuse elementwise add
                finally:
                    if scalar_tensor:
                        scalar_tensor.free_memory()
            else:
                return NotImplemented

        original_other_shape = other.shape
        broadcasting_occurred = False
        use_cpu_broadcast = False

        if self.shape == other.shape:
            # Elementwise case (GPU)
            requires_grad = (self.requires_grad or other.requires_grad) and OclTensor._enable_grad
            out = OclTensor.empty(self.shape, requires_grad=requires_grad)
            if self.numel > 0:
                assert self.data and self.data.ptr, "Add Input A buffer invalid"
                assert other.data and other.data.ptr, "Add Input B buffer invalid"
                assert out.data and out.data.ptr, "Add Output C buffer invalid"
                check_success(ocl.execute_add_on_gpu(GPU_ID, self.data.ptr, other.data.ptr, out.data.ptr, self.numel), "add (elementwise)")
            if out.requires_grad:
                out._ctx = AddBackwardContext(self, other, False, original_other_shape)  # No broadcasting
            return out
        else:
            # --- Check for CPU Broadcast Scenarios ---
            try:
                broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
                if broadcast_shape != self.shape:
                    raise NotImplementedError(f"Add: CPU broadcasting only supported if result shape matches first operand. Got {self.shape} + {other.shape} -> {broadcast_shape}")
                is_bias_like = (len(self.shape) >= 2 and (len(other.shape) == 1 or all(s == 1 for s in other.shape[:-1])) and self.shape[-1] == other.shape[-1])
                is_posenc_like = (len(self.shape) == 3 and len(other.shape) == 3 and other.shape[0] == 1 and self.shape[1:] == other.shape[1:])

                if is_bias_like or is_posenc_like:
                    use_cpu_broadcast = True
                    broadcasting_occurred = True
                else:
                    raise NotImplementedError(f"Add: Unsupported broadcast scenario for CPU fallback: {self.shape} + {other.shape} -> {broadcast_shape}")
            except ValueError:
                raise ValueError(f"Add: Incompatible shapes for addition: {self.shape} + {other.shape}")

            if use_cpu_broadcast:
                self_host = self.to_host()
                other_host = other.to_host()
                result_host = self_host + other_host
                requires_grad = (self.requires_grad or other.requires_grad) and OclTensor._enable_grad
                out = OclTensor(result_host, requires_grad=requires_grad)
                if out.requires_grad:
                    out._ctx = AddBackwardContext(self, other, broadcasting_occurred, original_other_shape)
                return out
            else:
                raise RuntimeError("Internal error in OclTensor.add broadcasting logic")

    def mul(self, other: "OclTensor") -> "OclTensor":
        # (Implementation unverändert, siehe vorherige Antwort)
        if not isinstance(other, OclTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise NotImplementedError(f"Broadcasting for mul not implemented: {self.shape} vs {other.shape}")
        requires_grad = (self.requires_grad or other.requires_grad) and OclTensor._enable_grad
        out = OclTensor.empty(self.shape, requires_grad=requires_grad)
        if self.numel > 0:
            assert self.data and self.data.ptr, "Mul Input A buffer invalid"
            assert other.data and other.data.ptr, "Mul Input B buffer invalid"
            assert out.data and out.data.ptr, "Mul Output C buffer invalid"
            check_success(ocl.execute_mul_on_gpu(GPU_ID, self.data.ptr, other.data.ptr, out.data.ptr, self.numel), "mul")
        if out.requires_grad:
            out._ctx = MulBackwardContext(self, other)
        return out

    def mul_scalar(self, scalar: float) -> "OclTensor":
        # (Implementation unverändert, siehe vorherige Antwort)
        scalar_tensor = None
        try:
            scalar_host = np.full(self.shape, scalar, dtype=FP_TYPE)
            scalar_tensor = OclTensor(scalar_host, requires_grad=False)
            out = self.mul(scalar_tensor)
            if self.requires_grad and OclTensor._enable_grad:
                out._ctx = ScalarMulBackwardContext(self, scalar)
            return out
        finally:
            if scalar_tensor:
                scalar_tensor.free_memory()

    def div_scalar(self, scalar: float) -> "OclTensor":
        # (Implementation unverändert, siehe vorherige Antwort)
        if scalar == 0:
            raise ZeroDivisionError("Scalar division by zero")
        return self.mul_scalar(1.0 / scalar)

    def gelu(self) -> "OclTensor":
        # (Implementation unverändert, siehe vorherige Antwort)
        requires_grad = self.requires_grad and OclTensor._enable_grad
        out = OclTensor.empty(self.shape, requires_grad=requires_grad)
        if self.numel > 0:
            assert self.data and self.data.ptr, "Gelu Input buffer invalid"
            assert out.data and out.data.ptr, "Gelu Output buffer invalid"
            check_success(ocl.execute_gelu_on_gpu(GPU_ID, self.data.ptr, out.data.ptr, self.numel), "gelu")
        if out.requires_grad:
            out._ctx = GeluBackwardContext(self)
        return out

    def softmax(self, dim=-1) -> "OclTensor":
        # (Implementation unverändert, siehe vorherige Antwort)
        if dim != -1 and dim != len(self.shape) - 1:
            raise NotImplementedError("Softmax only supported for the last dimension")
        if len(self.shape) < 1:
            raise ValueError("Softmax expects >= 1D tensor")
        if self.numel == 0:
            return OclTensor.empty(self.shape, requires_grad=self.requires_grad)
        row_size = self.shape[-1]
        num_rows = self.numel // row_size if row_size > 0 else 0
        requires_grad = self.requires_grad and OclTensor._enable_grad
        out = OclTensor.empty(self.shape, requires_grad=requires_grad)
        if num_rows > 0 and row_size > 0:
            assert self.data and self.data.ptr, "Softmax Input buffer invalid"
            assert out.data and out.data.ptr, "Softmax Output buffer invalid"
            check_success(ocl.execute_softmax_on_gpu(GPU_ID, self.data.ptr, out.data.ptr, int(num_rows), int(row_size)), "softmax")
        if out.requires_grad:
            output_clone = out.clone()
            out._ctx = SoftmaxBackwardContext(self, output_clone)
        return out

    def layer_norm(self, eps=1e-5) -> "OclTensor":
        # (Implementation unverändert, siehe vorherige Antwort - wird durch ocl_initialize gepatcht)
        raise NotImplementedError("LayerNorm should be monkey-patched during initialization")

    def transpose(self, dim0, dim1) -> "OclTensor":
        # (Implementation unverändert, siehe vorherige Antwort)
        ndim = len(self.shape)
        if ndim < 2:
            raise ValueError("Transpose requires at least 2 dimensions")
        dim0 = dim0 + ndim if dim0 < 0 else dim0
        dim1 = dim1 + ndim if dim1 < 0 else dim1
        if not (0 <= dim0 < ndim and 0 <= dim1 < ndim):
            raise IndexError("Transpose dimensions out of range")
        if dim0 == dim1:
            return self.clone()
        new_shape_list = list(self.shape)
        new_shape_list[dim0], new_shape_list[dim1] = new_shape_list[dim1], new_shape_list[dim0]
        new_shape = tuple(new_shape_list)
        requires_grad = self.requires_grad and OclTensor._enable_grad
        out = OclTensor.empty(new_shape, requires_grad=requires_grad)
        use_kernel = None
        if HAS_TRANSPOSE_12_BATCHED and ndim == 4 and ((dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1)):
            use_kernel = 'transpose12'
        elif HAS_TRANSPOSE_LAST_TWO and ndim >= 2 and ((dim0 == ndim - 2 and dim1 == ndim - 1) or (dim0 == ndim - 1 and dim1 == ndim - 2)):
            use_kernel = 'last_two'
        elif ndim == 2:
            use_kernel = 'basic'
        else:
            use_kernel = 'cpu'
        if self.numel > 0 and out.numel > 0:
            assert self.data and self.data.ptr, f"Input tensor {id(self)} buffer invalid before transpose ({use_kernel})"
            assert out.data and out.data.ptr, f"Output tensor {id(out)} buffer invalid before transpose ({use_kernel})"
            if use_kernel == 'transpose12':
                B, D1, D2, D3 = self.shape
                check_success(ocl.execute_transpose_12_batched_gpu(GPU_ID, self.data.ptr, out.data.ptr, B, D1, D2, D3), "transpose_12_batched")
            elif use_kernel == 'last_two':
                d1, d2 = self.shape[-2], self.shape[-1]
                b_flat = int(np.prod(self.shape[:-2])) if ndim > 2 else 1
                check_success(ocl.execute_transpose_batched_gpu(GPU_ID, self.data.ptr, out.data.ptr, b_flat, d1, d2), "transpose_batched_last_two")
            elif use_kernel == 'basic':
                rows_A, cols_A = self.shape[0], self.shape[1]
                check_success(ocl.execute_transpose_on_gpu(GPU_ID, self.data.ptr, out.data.ptr, rows_A, cols_A), "transpose2D")
            else:  # use_kernel == 'cpu'
                print(f">> WARNING: Using CPU transpose for {self.shape} swapping dims {dim0}<->{dim1} <<")
                sys.stdout.flush()
                input_host = self.to_host()
                axes = list(range(ndim))
                axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
                output_host = np.transpose(input_host, axes=tuple(axes))
                out.data.write(np.ascontiguousarray(output_host, dtype=FP_TYPE))
        if out.requires_grad:
            if use_kernel == 'transpose12':
                out._ctx = Transpose12BatchedBackwardContext(self)
            elif use_kernel == 'last_two':
                out._ctx = BatchedTransposeLastTwoBackwardContext(self)
            elif use_kernel == 'basic':
                rows_A, cols_A = self.shape[0], self.shape[1]
                out._ctx = TransposeBackwardContext(self, rows_A, cols_A)
            else:
                out._ctx = TransposeCPUFallbackBackwardContext(self, dim0, dim1)
        return out

    def reshape(self, *shape) -> "OclTensor":
        # (Implementation unverändert, siehe vorherige Antwort)
        inferred_shape = list(shape)
        if -1 in inferred_shape:
            if inferred_shape.count(-1) > 1:
                raise ValueError("Can only infer one dimension")
            inf_idx = inferred_shape.index(-1)
            known_prod = np.prod([s for s in shape if s != -1], dtype=np.int64)
            if self.numel == 0:
                inferred_shape[inf_idx] = 0 if known_prod == 0 else ValueError("Cannot infer dim for empty tensor")
            elif known_prod == 0:
                raise ValueError(f"Cannot reshape non-empty tensor to shape with zero {shape}")
            elif self.numel % known_prod != 0:
                raise ValueError(f"Cannot reshape size {self.numel} into {shape}")
            else:
                inferred_shape[inf_idx] = self.numel // known_prod
        final_shape = tuple(inferred_shape)
        if np.prod(final_shape, dtype=np.int64) != self.numel:
            raise ValueError(f"Cannot reshape {self.numel} elements to {final_shape}")
        requires_grad = self.requires_grad and OclTensor._enable_grad
        out = OclTensor(np.empty(0, dtype=FP_TYPE), requires_grad=requires_grad, _gpu_buffer=self.data, _shape=final_shape)
        if out.requires_grad:
            out._ctx = ReshapeBackwardContext(self)
        return out

    # --- Operator Überladung (unverändert) ---
    def __matmul__(self, other):
        return self.matmul(other)
    def __add__(self, other):
        return self.add(other)
    def __radd__(self, other):
        return self.add(other) if isinstance(other, OclTensor) else self.add(OclTensor(np.full(self.shape, other, dtype=FP_TYPE)))
    def __mul__(self, other):
        return self.mul_scalar(float(other)) if isinstance(other, (int, float)) else self.mul(other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        return self.div_scalar(float(other)) if isinstance(other, (int, float)) else NotImplemented
    def __sub__(self, other):
        return self.add(other.mul_scalar(-1.0)) if isinstance(other, OclTensor) else self.add(OclTensor(np.full(self.shape, -float(other), dtype=FP_TYPE)))
    def __rsub__(self, other):
        return OclTensor(np.full(self.shape, float(other), dtype=FP_TYPE)).add(self.mul_scalar(-1.0))

    # --- Repräsentation & Speicher (unverändert) ---
    def __repr__(self):
        return f"OclTensor(shape={self.shape}, req_grad={self.requires_grad}, id={id(self)})"
    def free_memory(self):
        if hasattr(self, "data") and self.data:
            self.data.free()
            self.data = None
        if hasattr(self, "_grad_buffer") and self._grad_buffer:
            self._grad_buffer.free()
            self._grad_buffer = None
        self.grad = None
        self._ctx = None
        remove_tensor_from_registry(self)
    @property
    def T(self):
        ndim = len(self.shape)
        return self.transpose(ndim - 2, ndim - 1) if ndim >= 2 else self

# --------------------------------------------------------------------------
# 6. Autograd Context Klassen (AddBackwardContext angepasst, AddBroadcastPEBackwardContext neu)
# --------------------------------------------------------------------------
class FunctionContext:
    """Base class for backward pass context. Stores parent tensors and potentially saved data."""
    def __init__(self, *tensors_saved_for_backward):
        # Store weak references to avoid circular dependencies that prevent garbage collection
        self.parents = [weakref.ref(t) for t in tensors_saved_for_backward if isinstance(t, OclTensor)]
        self.saved_data = {}  # Dictionary to store intermediate values or shapes needed for backward

    def save_for_backward(self, **kwargs):
        """Saves data needed for the backward pass."""
        self.saved_data.update(kwargs)

    def unpack_saved_data(self):
        """Retrieves the saved data dictionary."""
        return self.saved_data

    def apply_gradients(self, *gradients):
        """Applies the calculated gradients to the corresponding parent tensors."""
        if len(gradients) != len(self.parents):
            raise RuntimeError(f"Internal Error: Gradient count ({len(gradients)}) mismatch parent count ({len(self.parents)}) in {self.__class__.__name__}")
        for i, parent_ref in enumerate(self.parents):
            parent = parent_ref()  # Get the actual tensor from the weak reference
            grad = gradients[i]    # Get the corresponding calculated gradient
            if parent and parent.requires_grad and grad is not None and isinstance(grad, OclTensor):
                parent._accumulate_grad(grad)

# --- Context für Standard und Batched MatMul (unverändert) ---
class MatMulBackwardContext(FunctionContext):
    def __init__(self, a: OclTensor, b: OclTensor):
        super().__init__(a, b)
        self.save_for_backward(a_ptr=a.data.ptr, b_ptr=b.data.ptr, a_shape=a.shape, b_shape=b.shape)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        a_shape, b_shape = saved['a_shape'], saved['b_shape']
        a_ptr, b_ptr = saved['a_ptr'], saved['b_ptr']
        B = a_shape[0] if len(a_shape) == 3 else 1
        M = a_shape[-2]
        K = a_shape[-1]
        N = b_shape[-1]
        a_parent, b_parent = self.parents[0](), self.parents[1]()
        grad_a, grad_b = None, None
        grad_a_ptr, grad_b_ptr = None, None
        temp_tensors = []

        if a_parent and a_parent.requires_grad:
            grad_a = OclTensor.empty(a_shape, False)
            temp_tensors.append(grad_a)
            grad_a_ptr = grad_a.data.ptr if grad_a.data else None
        if b_parent and b_parent.requires_grad:
            grad_b = OclTensor.empty(b_shape, False)
            temp_tensors.append(grad_b)
            grad_b_ptr = grad_b.data.ptr if grad_b.data else None

        if grad_a_ptr or grad_b_ptr:
            if not grad_output.data or not grad_output.data.ptr:
                raise ValueError("grad_output invalid in MatMul BW")
            if not a_ptr or not b_ptr:
                raise ValueError("Original A/B pointer invalid in MatMul BW")
            check_success(ocl.execute_matmul_backward_on_gpu(GPU_ID, a_ptr, b_ptr, grad_output.data.ptr, grad_a_ptr, grad_b_ptr, B, M, N, K), "matmul.bw")
        self.apply_gradients(grad_a, grad_b)
        for t in temp_tensors:
            t.free_memory()

class MatMulBatchedBackwardContext(FunctionContext):
    def __init__(self, a: OclTensor, b: OclTensor):
        super().__init__(a, b)
        self.save_for_backward(a_ptr=a.data.ptr, b_ptr=b.data.ptr, a_shape=a.shape, b_shape=b.shape)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        a_shape, b_shape = saved['a_shape'], saved['b_shape']
        B, M, K = a_shape
        _, _, N = b_shape
        a_parent, b_parent = self.parents[0](), self.parents[1]()
        grad_a, grad_b = None, None
        grad_a_ptr, grad_b_ptr = None, None
        temp_tensors = []

        if a_parent and a_parent.requires_grad:
            grad_a = OclTensor.empty(a_shape, False)
            temp_tensors.append(grad_a)
            grad_a_ptr = grad_a.data.ptr if grad_a.data else None
        if b_parent and b_parent.requires_grad:
            grad_b = OclTensor.empty(b_shape, False)
            temp_tensors.append(grad_b)
            grad_b_ptr = grad_b.data.ptr if grad_b.data else None

        if grad_a_ptr or grad_b_ptr:
            if not grad_output.data or not grad_output.data.ptr:
                raise ValueError("grad_output invalid in MatMulBatched BW")
            if not saved['a_ptr'] or not saved['b_ptr']:
                raise ValueError("Original A/B pointer invalid in MatMulBatched BW")
            check_success(ocl.execute_matmul_batched_backward_on_gpu(GPU_ID, saved['a_ptr'], saved['b_ptr'], grad_output.data.ptr, grad_a_ptr, grad_b_ptr, B, M, N, K), "matmul_batched.bw")
        self.apply_gradients(grad_a, grad_b)
        for t in temp_tensors:
            t.free_memory()

# --- Context for Add (Handles Elementwise and CPU/GPU Broadcast Reduction) ---
class AddBackwardContext(FunctionContext):
    def __init__(self, a: OclTensor, b: OclTensor, broadcasting_occurred: bool, b_original_shape: tuple):
        """
        Context for addition.

        Args:
            a: First operand tensor.
            b: Second operand tensor.
            broadcasting_occurred (bool): True if broadcasting happened (b's shape != output shape).
            b_original_shape (tuple): The shape of 'b' before potential broadcasting.
        """
        super().__init__(a, b)
        self.save_for_backward(bcast=broadcasting_occurred, b_orig_sh=b_original_shape)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        bcast, b_orig_sh = saved['bcast'], saved['b_orig_sh']
        a_parent, b_parent = self.parents[0](), self.parents[1]()
        grad_a, grad_b = None, None
        temp_grad_b = None

        if a_parent and a_parent.requires_grad:
            grad_a = grad_output  # Direct pass-through
        if b_parent and b_parent.requires_grad:
            if bcast and grad_output.shape != b_orig_sh:
                can_use_reduce_kernel = False
                if HAS_REDUCE_SUM and len(grad_output.shape) >= 2 and len(b_orig_sh) == 1 and grad_output.shape[-1] == b_orig_sh[0]:
                    if len(grad_output.shape) == 3:
                        B, M, N = grad_output.shape
                    elif len(grad_output.shape) == 2:
                        B, N = grad_output.shape
                        M = 1
                    else:
                        B, M, N = 0, 0, 0
                    if B > 0 and M > 0 and N == b_orig_sh[0]:
                        can_use_reduce_kernel = True
                        temp_grad_b = OclTensor.empty(b_orig_sh, requires_grad=False)
                        assert grad_output.data and grad_output.data.ptr, "Grad Output Buffer invalid for ReduceSum"
                        assert temp_grad_b.data and temp_grad_b.data.ptr, "Temp Grad B Buffer invalid for ReduceSum"
                        check_success(ocl.execute_reduce_sum_gpu(GPU_ID, grad_output.data.ptr, temp_grad_b.data.ptr, B, M, N), "reduce_sum_bias.bw")
                        grad_b = temp_grad_b
                if not can_use_reduce_kernel:
                    grad_output_host = grad_output.to_host()
                    num_added_dims = len(grad_output.shape) - len(b_orig_sh)
                    axes_to_sum_list = list(range(num_added_dims))
                    for i in range(len(b_orig_sh)):
                        if b_orig_sh[i] == 1 and grad_output.shape[num_added_dims + i] > 1:
                            axes_to_sum_list.append(num_added_dims + i)
                    axes_to_sum = tuple(axes_to_sum_list) if axes_to_sum_list else None
                    grad_b_host = np.sum(grad_output_host, axis=axes_to_sum, dtype=FP_TYPE, keepdims=False)
                    try:
                        grad_b_host_reshaped = grad_b_host.reshape(b_orig_sh)
                    except ValueError as e:
                        raise ValueError(f"Cannot reshape reduced broadcast grad {grad_b_host.shape} to {b_orig_sh}: {e}")
                    temp_grad_b = OclTensor(grad_b_host_reshaped)
                    grad_b = temp_grad_b
            else:
                grad_b = grad_output
        self.apply_gradients(grad_a, grad_b)
        if temp_grad_b:
            temp_grad_b.free_memory()

# --- Context für dedizierten GPU Broadcast PE Add ---
class AddBroadcastPEBackwardContext(FunctionContext):
    def __init__(self, input_tensor: OclTensor, pe_tensor: OclTensor):
        # Only the input tensor requires gradient. PE tensor is constant.
        super().__init__(input_tensor)
    def backward(self, grad_output: OclTensor):
        input_parent = self.parents[0]()  # Get the input tensor ('x' in the forward pass)
        grad_input = None
        if input_parent and input_parent.requires_grad:
            grad_input = grad_output
        self.apply_gradients(grad_input)

# --- Andere Context Klassen (gekürzt bzw. unverändert) ---
class MulBackwardContext(FunctionContext):
    def __init__(self, a: OclTensor, b: OclTensor):
        super().__init__(a, b)
        self.save_for_backward(a_ptr=a.data.ptr, b_ptr=b.data.ptr, numel=a.numel, a_shape=a.shape, b_shape=b.shape)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        a_ptr, b_ptr, numel, a_shape, b_shape = saved['a_ptr'], saved['b_ptr'], saved['numel'], saved['a_shape'], saved['b_shape']
        a_parent, b_parent = self.parents[0](), self.parents[1]()
        grad_a, grad_b = None, None
        grad_a_ptr, grad_b_ptr = None, None
        temp_tensors = []
        if a_parent and a_parent.requires_grad:
            grad_a = OclTensor.empty(a_shape, False)
            temp_tensors.append(grad_a)
            grad_a_ptr = grad_a.data.ptr if grad_a.data else None
        if b_parent and b_parent.requires_grad:
            grad_b = OclTensor.empty(b_shape, False)
            temp_tensors.append(grad_b)
            grad_b_ptr = grad_b.data.ptr if grad_b.data else None
        if grad_a_ptr or grad_b_ptr:
            if not grad_output.data or not grad_output.data.ptr:
                raise ValueError("grad_output invalid in Mul BW")
            if not a_ptr or not b_ptr:
                raise ValueError("Original A/B pointer invalid in Mul BW")
            check_success(ocl.execute_mul_backward_on_gpu(GPU_ID, grad_output.data.ptr, a_ptr, b_ptr, grad_a_ptr, grad_b_ptr, numel), "mul.backward")
        self.apply_gradients(grad_a, grad_b)
        for t in temp_tensors:
            t.free_memory()

class GeluBackwardContext(FunctionContext):
    def __init__(self, input_tensor: OclTensor):
        super().__init__(input_tensor)
        self.save_for_backward(input_ptr=input_tensor.data.ptr, input_numel=input_tensor.numel, input_shape=input_tensor.shape)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        input_ptr, input_numel, input_shape = saved['input_ptr'], saved['input_numel'], saved['input_shape']
        input_parent = self.parents[0]()
        grad_input = None
        temp_tensors = []
        if input_parent and input_parent.requires_grad:
            grad_input = OclTensor.empty(input_shape, False)
            temp_tensors.append(grad_input)
            grad_input_ptr = grad_input.data.ptr if grad_input.data else None
            if not grad_output.data or not grad_output.data.ptr:
                raise ValueError("grad_output invalid in Gelu BW")
            if not input_ptr:
                raise ValueError("Original input pointer invalid in Gelu BW")
            if input_numel > 0 and grad_input_ptr:
                check_success(ocl.execute_gelu_backward_on_gpu(GPU_ID, input_ptr, grad_output.data.ptr, grad_input_ptr, input_numel), "gelu.bw")
            elif input_numel > 0:
                raise ValueError("Grad input buffer invalid in Gelu BW")
        self.apply_gradients(grad_input)
        for t in temp_tensors:
            t.free_memory()

class SoftmaxBackwardContext(FunctionContext):
    def __init__(self, input_tensor: OclTensor, output_tensor_clone: OclTensor):
        super().__init__(input_tensor)
        self.y_clone_ref = weakref.ref(output_tensor_clone)
        self.save_for_backward(y_ptr=output_tensor_clone.data.ptr, y_shape=output_tensor_clone.shape)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        y_ptr, y_shape = saved['y_ptr'], saved['y_shape']
        input_parent = self.parents[0]()
        grad_input = None
        temp_tensors = []
        if input_parent and input_parent.requires_grad:
            num_rows = np.prod(y_shape[:-1], dtype=np.int64)
            row_size = y_shape[-1]
            grad_input = OclTensor.empty(y_shape, False)
            temp_tensors.append(grad_input)
            grad_input_ptr = grad_input.data.ptr if grad_input.data else None
            if not grad_output.data or not grad_output.data.ptr:
                raise ValueError("grad_output invalid in Softmax BW")
            if not y_ptr:
                raise ValueError("Cloned output pointer invalid in Softmax BW")
            if num_rows > 0 and row_size > 0 and grad_input_ptr:
                check_success(ocl.execute_softmax_backward_on_gpu(GPU_ID, grad_output.data.ptr, y_ptr, grad_input_ptr, int(num_rows), int(row_size)), "softmax.bw")
            elif num_rows * row_size > 0:
                raise ValueError("Grad input buffer invalid in Softmax BW")
        self.apply_gradients(grad_input)
        for t in temp_tensors:
            t.free_memory()
    def __del__(self):
        if hasattr(self, 'y_clone_ref'):
            y_clone = self.y_clone_ref()
            if y_clone:
                try:
                    y_clone.free_memory()
                except Exception as e:
                    pass

class LayerNormBackwardContextRevised(FunctionContext):
    def __init__(self, input_tensor: OclTensor, eps: float):
        super().__init__(input_tensor)
        self.x_clone_ref = weakref.ref(input_tensor.clone())
        x_clone_temp = self.x_clone_ref()
        x_ptr_save = x_clone_temp.data.ptr if x_clone_temp and x_clone_temp.data else None
        self.save_for_backward(x_ptr=x_ptr_save, x_shape=input_tensor.shape, eps=eps)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        x_ptr, x_shape, eps = saved['x_ptr'], saved['x_shape'], saved['eps']
        input_parent = self.parents[0]()
        grad_input = None
        temp_tensors = []
        if input_parent and input_parent.requires_grad:
            num_rows = np.prod(input_parent.shape[:-1], dtype=np.int64)
            row_size = input_parent.shape[-1]
            grad_input = OclTensor.empty(input_parent.shape, False)
            temp_tensors.append(grad_input)
            grad_input_ptr = grad_input.data.ptr if grad_input.data else None
            if not grad_output.data or not grad_output.data.ptr:
                raise ValueError("grad_output invalid in LayerNorm BW")
            if not x_ptr:
                raise ValueError("Cloned input pointer invalid in LayerNorm BW")
            if num_rows > 0 and row_size > 0 and grad_input_ptr:
                check_success(ocl.execute_layernorm_backward_on_gpu(GPU_ID, grad_output.data.ptr, x_ptr, grad_input_ptr, int(num_rows), int(row_size), c_float(eps)), "ln.bw")
            elif num_rows * row_size > 0:
                raise ValueError("Grad input buffer invalid in LayerNorm BW")
        self.apply_gradients(grad_input)
        for t in temp_tensors:
            t.free_memory()
    def __del__(self):
        if hasattr(self, 'x_clone_ref'):
            x_clone = self.x_clone_ref()
            if x_clone:
                try:
                    x_clone.free_memory()
                except Exception as e:
                    pass

LayerNormBackwardContext = LayerNormBackwardContextRevised  # Global alias

class TransposeBackwardContext(FunctionContext):
    def __init__(self, input_tensor: OclTensor, rows_A, cols_A):
        super().__init__(input_tensor)
        self.save_for_backward(rows_A=rows_A, cols_A=cols_A)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        rows_A, cols_A = saved['rows_A'], saved['cols_A']
        input_parent = self.parents[0]()
        grad_input = None
        temp_tensors = []
        if input_parent and input_parent.requires_grad:
            if len(input_parent.shape) != 2:
                raise NotImplementedError("Transpose.bw (2D) only for 2D tensors")
            grad_input = OclTensor.empty(input_parent.shape, False)
            temp_tensors.append(grad_input)
            grad_input_ptr = grad_input.data.ptr if grad_input.data else None
            if not grad_output.data or not grad_output.data.ptr:
                raise ValueError("grad_output invalid in Transpose BW (2D)")
            if rows_A > 0 and cols_A > 0 and grad_input_ptr:
                check_success(ocl.execute_transpose_backward_on_gpu(GPU_ID, grad_output.data.ptr, grad_input_ptr, rows_A, cols_A), "transpose.bw (2D)")
            elif rows_A * cols_A > 0:
                raise ValueError("Grad input buffer invalid in Transpose BW (2D)")
        self.apply_gradients(grad_input)
        for t in temp_tensors:
            t.free_memory()

class TransposeCPUFallbackBackwardContext(FunctionContext):
    def __init__(self, input_tensor: OclTensor, dim0: int, dim1: int):
        super().__init__(input_tensor)
        self.save_for_backward(dim0=dim0, dim1=dim1, ndim=len(input_tensor.shape))
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        dim0, dim1, ndim = saved['dim0'], saved['dim1'], saved['ndim']
        input_parent = self.parents[0]()
        grad_input = None
        temp_grad_input_tensor = None
        try:
            if input_parent and input_parent.requires_grad:
                grad_output_host = grad_output.to_host()
                axes = list(range(ndim))
                axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
                axes_tuple = tuple(axes)
                grad_input_host = np.transpose(grad_output_host, axes=axes_tuple)
                grad_input_host_contig = np.ascontiguousarray(grad_input_host, dtype=FP_TYPE)
                temp_grad_input_tensor = OclTensor(grad_input_host_contig)
                grad_input = temp_grad_input_tensor
            self.apply_gradients(grad_input)
        finally:
            if temp_grad_input_tensor:
                temp_grad_input_tensor.free_memory()

class BatchedTransposeLastTwoBackwardContext(FunctionContext):
    def __init__(self, input_tensor: OclTensor):
        super().__init__(input_tensor)
        self.save_for_backward(original_shape=input_tensor.shape)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        original_shape = saved['original_shape']
        ndim = len(original_shape)
        input_parent = self.parents[0]()
        grad_input = None
        temp_tensors = []
        if input_parent and input_parent.requires_grad:
            if ndim < 2:
                raise ValueError("BatchedTransposeLastTwoBackward needs >= 2 dims")
            expected_transposed_shape = tuple(list(original_shape[:-2]) + [original_shape[-1], original_shape[-2]])
            if grad_output.shape != expected_transposed_shape:
                raise ValueError(f"BatchedTransposeLastTwoBW shape mismatch {grad_output.shape} != {expected_transposed_shape}")
            grad_input = OclTensor.empty(original_shape, False)
            temp_tensors.append(grad_input)
            grad_input_ptr = grad_input.data.ptr if grad_input.data else None
            if grad_input.numel > 0 and grad_input_ptr:
                if not grad_output.data or not grad_output.data.ptr:
                    raise ValueError("grad_output invalid in BatchedTransposeLastTwoBW")
                d1_out, d2_out = grad_output.shape[-2], grad_output.shape[-1]
                b_flat_out = int(np.prod(grad_output.shape[:-2])) if ndim > 2 else 1
                check_success(ocl.execute_transpose_batched_gpu(GPU_ID, grad_output.data.ptr, grad_input_ptr, b_flat_out, d1_out, d2_out), "transpose_batched_last_two.bw")
            elif grad_input.numel > 0:
                raise ValueError("Grad input buffer invalid in BatchedTransposeLastTwoBW")
        self.apply_gradients(grad_input)
        for t in temp_tensors:
            t.free_memory()

class Transpose12BatchedBackwardContext(FunctionContext):
    def __init__(self, input_tensor: OclTensor):
        super().__init__(input_tensor)
        self.save_for_backward(original_shape=input_tensor.shape)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        original_shape = saved['original_shape']
        ndim = len(original_shape)
        input_parent = self.parents[0]()
        grad_input = None
        temp_tensors = []
        if input_parent and input_parent.requires_grad:
            if ndim != 4:
                raise ValueError(f"Transpose12BatchedBackward requires 4D tensor, got {ndim}D")
            expected_transposed_shape = (original_shape[0], original_shape[2], original_shape[1], original_shape[3])
            if grad_output.shape != expected_transposed_shape:
                raise ValueError(f"Transpose12BW shape mismatch {grad_output.shape} != {expected_transposed_shape}")
            grad_input = OclTensor.empty(original_shape, False)
            temp_tensors.append(grad_input)
            grad_input_ptr = grad_input.data.ptr if grad_input.data else None
            if grad_input.numel > 0 and grad_input_ptr:
                if not grad_output.data or not grad_output.data.ptr:
                    raise ValueError("grad_output invalid in Transpose12BW")
                B_g, D1_g, D2_g, D3_g = grad_output.shape
                check_success(ocl.execute_transpose_12_batched_gpu(GPU_ID, grad_output.data.ptr, grad_input_ptr, B_g, D1_g, D2_g, D3_g), "transpose_12_batched.bw")
            elif grad_input.numel > 0:
                raise ValueError("Grad input buffer invalid in Transpose12BW")
        self.apply_gradients(grad_input)
        for t in temp_tensors:
            t.free_memory()

class ReshapeBackwardContext(FunctionContext):
    def __init__(self, input_tensor: OclTensor):
        super().__init__(input_tensor)
        self.save_for_backward(original_shape=input_tensor.shape)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        original_shape = saved['original_shape']
        input_parent = self.parents[0]()
        grad_input = None
        if input_parent and input_parent.requires_grad:
            if grad_output.numel != input_parent.numel:
                raise RuntimeError(f"Reshape BW numel mismatch {grad_output.numel} vs {input_parent.numel}")
            grad_input = grad_output.reshape(*original_shape)
        self.apply_gradients(grad_input)

class EmbeddingBackwardContext(FunctionContext):
    def __init__(self, weight_param: OclTensor, input_ids: np.ndarray):
        super().__init__(weight_param)
        self.save_for_backward(input_ids=input_ids.copy())
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        input_ids = saved['input_ids']
        weight_param = self.parents[0]()
        temp_grad_weight_tensor = None
        if not weight_param or not weight_param.requires_grad:
            return
        try:
            grad_output_host = grad_output.to_host()
            b, s, e = grad_output_host.shape
            v = weight_param.shape[0]
            grad_weight_host = np.zeros(weight_param.shape, dtype=FP_TYPE)
            for bi in range(b):
                for si in range(s):
                    idx = input_ids[bi, si]
                    if 0 <= idx < v:
                        grad_weight_host[idx, :] += grad_output_host[bi, si, :]
            temp_grad_weight_tensor = OclTensor(grad_weight_host)
            self.apply_gradients(temp_grad_weight_tensor)
        finally:
            if temp_grad_weight_tensor:
                temp_grad_weight_tensor.free_memory()

class ScalarMulBackwardContext(FunctionContext):
    def __init__(self, input_tensor: OclTensor, scalar: float):
        super().__init__(input_tensor)
        self.save_for_backward(scalar=scalar)
    def backward(self, grad_output: OclTensor):
        saved = self.unpack_saved_data()
        scalar = saved['scalar']
        input_parent = self.parents[0]()
        grad_input = None
        if input_parent and input_parent.requires_grad:
            grad_input = grad_output.mul_scalar(scalar)
        self.apply_gradients(grad_input)

# --------------------------------------------------------------------------
# 7. Layer Definitionen (PositionalEncoding angepasst)
# --------------------------------------------------------------------------
class Parameter(OclTensor):
    """A special OclTensor that is registered as a model parameter."""
    def __init__(self, data: np.ndarray):
        super().__init__(data, requires_grad=True)

class Linear:
    """Standard Linear layer (Matrix Multiply + optional Bias Add)."""
    def __init__(self, in_features, out_features, use_bias=True):
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.W = Parameter(np.random.uniform(-limit, limit, size=(in_features, out_features)).astype(FP_TYPE))
        self.b = Parameter(np.zeros((1, out_features), dtype=FP_TYPE)) if use_bias else None
    def __call__(self, x: OclTensor) -> OclTensor:
        y = x.matmul(self.W)
        return y.add(self.b) if self.b else y
    def parameters(self):
        return [self.W] + ([self.b] if self.b else [])

class Embedding:
    """Embedding layer (CPU Lookup)."""
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size, self.embed_dim = vocab_size, embed_dim
        scale = math.sqrt(1.0 / embed_dim)
        weight_data = np.random.uniform(-scale, scale, size=(vocab_size, embed_dim)).astype(FP_TYPE)
        self.weight_param = Parameter(weight_data.copy())
        self.weight_host = weight_data
        self.weight_param._is_embedding_weight = True
        self.weight_param._associated_layer = weakref.ref(self)
    def __call__(self, input_ids: np.ndarray) -> OclTensor:
        b, s = input_ids.shape
        out_shape = (b, s, self.embed_dim)
        requires_grad = self.weight_param.requires_grad and OclTensor._enable_grad
        out_host = np.zeros(out_shape, dtype=FP_TYPE)
        for bi in range(b):
            for si in range(s):
                idx = input_ids[bi, si]
                if 0 <= idx < self.vocab_size:
                    out_host[bi, si, :] = self.weight_host[idx, :]
        out = OclTensor(out_host, requires_grad=requires_grad)
        if out.requires_grad:
            out._ctx = EmbeddingBackwardContext(self.weight_param, input_ids)
        return out
    def parameters(self):
        return [self.weight_param]
    def update_host_weight_from_gpu(self):
        if self.weight_param and self.weight_param.data and self.weight_param.data._allocated:
            self.weight_host = self.weight_param.to_host()

class GeLUActivation:
    """Gaussian Error Linear Unit activation function."""
    def __call__(self, x: OclTensor) -> OclTensor:
        return x.gelu()
    def parameters(self):
        return []

class LayerNorm:
    """Layer Normalization."""
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            self.normalized_size = normalized_shape
        elif isinstance(normalized_shape, (list, tuple)) and len(normalized_shape) == 1:
            self.normalized_size = normalized_shape[0]
        else:
            raise TypeError(f"LayerNorm expects int or single-element tuple/list for normalized_shape, got {normalized_shape}")
        self.eps = eps
    def __call__(self, x: OclTensor) -> OclTensor:
        if x.shape[-1] != self.normalized_size:
            raise ValueError(f"Input feature size {x.shape[-1]} != LayerNorm size {self.normalized_size}")
        return x.layer_norm(self.eps)
    def parameters(self):
        return []

class PositionalEncoding:
    """Adds positional encoding to the input embeddings. Uses GPU kernel if available."""
    def __init__(self, d_model, max_len=512):
        self.d_model = d_model
        self.max_len = max_len
        pe_host = np.zeros((max_len, d_model), dtype=FP_TYPE)
        position = np.arange(0, max_len, dtype=FP_TYPE).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=FP_TYPE) * (-math.log(10000.0) / d_model))
        pe_host[:, 0::2] = np.sin(position * div_term)
        pe_host[:, 1::2] = np.cos(position * div_term)
        self.pe_gpu_full = None
        if max_len > 0 and d_model > 0:
            self.pe_gpu_full = GPUBuffer(pe_host.nbytes, "pe_full_buffer")
            if self.pe_gpu_full._allocated:
                self.pe_gpu_full.write(pe_host)
                print(f"[PosEnc] Created persistent GPU buffer ({max_len}, {d_model})")
            else:
                print(f"WARN [PosEnc] Failed to allocate persistent GPU buffer!")
                self.pe_gpu_full = None
    def __call__(self, x: OclTensor) -> OclTensor:
        B, S, E = x.shape
        if S <= 0:
            return x
        if S > self.max_len:
            raise ValueError(f"Input Sequence Length {S} exceeds PositionalEncoding max_len {self.max_len}")
        if E != self.d_model:
            raise ValueError(f"Input embedding dim {E} does not match PositionalEncoding d_model {self.d_model}")
        if not self.pe_gpu_full or not self.pe_gpu_full.ptr:
            raise RuntimeError("PositionalEncoding GPU buffer not initialized or invalid!")
        requires_grad = x.requires_grad and OclTensor._enable_grad
        result = OclTensor.empty(x.shape, requires_grad=requires_grad)
        if HAS_ADD_BROADCAST_PE:
            slice_nbytes = S * E * FP_SIZE
            pe_slice_gpu_temp = GPUBuffer(slice_nbytes, "pe_slice_temp")
            if not pe_slice_gpu_temp._allocated:
                raise MemoryError("Failed to allocate temporary PE slice buffer")
            try:
                check_success(ocl.execute_clone_on_gpu(GPU_ID, self.pe_gpu_full.ptr, pe_slice_gpu_temp.ptr, slice_nbytes), "pe_slice_copy")
                assert x.data and x.data.ptr, "PE Add Input buffer invalid"
                assert result.data and result.data.ptr, "PE Add Output buffer invalid"
                check_success(ocl.execute_add_broadcast_pe_gpu(GPU_ID, x.data.ptr, pe_slice_gpu_temp.ptr, result.data.ptr, B, S, E), "add_broadcast_pe")
                if result.requires_grad:
                    result._ctx = AddBroadcastPEBackwardContext(x, None)
            finally:
                pe_slice_gpu_temp.free()
        else:
            print(">> WARNING: Using CPU Broadcast Add for Positional Encoding <<")
            pe_slice_host = self.pe_gpu_full.read((S, E), dtype=FP_TYPE)
            pe_slice_gpu_temp = None
            result_cpu_add = None
            try:
                pe_slice_gpu_temp = OclTensor(pe_slice_host[np.newaxis, :, :], requires_grad=False)
                result_cpu_add = x.add(pe_slice_gpu_temp)
                result.free_memory()
                result = result_cpu_add
                result_cpu_add = None
            finally:
                if pe_slice_gpu_temp:
                    pe_slice_gpu_temp.free_memory()
                if result_cpu_add:
                    result_cpu_add.free_memory()
        return result
    def parameters(self):
        return []
    def free_memory(self):
        if hasattr(self, 'pe_gpu_full') and self.pe_gpu_full:
            self.pe_gpu_full.free()
            self.pe_gpu_full = None

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("Invalid d_model/num_heads")
        self.d_model, self.num_heads, self.d_k = d_model, num_heads, d_model // num_heads
        self.q_proj = Linear(d_model, d_model, use_bias=False)
        self.k_proj = Linear(d_model, d_model, use_bias=False)
        self.v_proj = Linear(d_model, d_model, use_bias=False)
        self.out_proj = Linear(d_model, d_model)
    def __call__(self, query: OclTensor, key: OclTensor, value: OclTensor, mask=None) -> OclTensor:
        B, T_q, C = query.shape
        B_k, T_kv, C_k = key.shape
        B_v, T_kv_v, C_v = value.shape
        if not (B == B_k == B_v and C == C_k == C_v == self.d_model and T_kv == T_kv_v):
            raise ValueError(f"MHA shape mismatch: Q{query.shape}, K{key.shape}, V{value.shape}")
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        Q_heads = Q.reshape(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K_heads = K.reshape(B, T_kv, self.num_heads, self.d_k).transpose(1, 2)
        V_heads = V.reshape(B, T_kv, self.num_heads, self.d_k).transpose(1, 2)
        BH = B * self.num_heads
        Q_flat = Q_heads.reshape(BH, T_q, self.d_k)
        K_flat = K_heads.reshape(BH, T_kv, self.d_k)
        V_flat = V_heads.reshape(BH, T_kv, self.d_k)
        K_flat_T = K_flat.transpose(-2, -1)
        attn_scores_flat = Q_flat.matmul(K_flat_T)
        attn_scores_scaled_flat = attn_scores_flat.mul_scalar(1.0 / math.sqrt(self.d_k))
        if mask is not None:
            print("WARN: Attention Masking not implemented.")
        attn_probs_flat = attn_scores_scaled_flat.softmax(dim=-1)
        attn_output_flat = attn_probs_flat.matmul(V_flat)
        attn_output_heads = attn_output_flat.reshape(B, self.num_heads, T_q, self.d_k)
        attn_output_transposed = attn_output_heads.transpose(1, 2)
        attn_output_concat = attn_output_transposed.reshape(B, T_q, C)
        output = self.out_proj(attn_output_concat)
        return output
    def parameters(self):
        return self.q_proj.parameters() + self.k_proj.parameters() + self.v_proj.parameters() + self.out_proj.parameters()

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.d_model, self.num_heads, self.d_ff = d_model, num_heads, d_ff
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn_linear1 = Linear(d_model, d_ff)
        self.ffn_activation = GeLUActivation()
        self.ffn_linear2 = Linear(d_ff, d_model)
        self.norm2 = LayerNorm(d_model)
    def __call__(self, x: OclTensor, mask=None) -> OclTensor:
        residual = x
        x_norm1 = self.norm1(x)
        attn_output = self.self_attn(x_norm1, x_norm1, x_norm1, mask)
        x = residual.add(attn_output)
        residual = x
        x_norm2 = self.norm2(x)
        ffn_output = self.ffn_linear2(self.ffn_activation(self.ffn_linear1(x_norm2)))
        x = residual.add(ffn_output)
        return x
    def parameters(self):
        params = []
        params.extend(self.norm1.parameters())
        params.extend(self.self_attn.parameters())
        params.extend(self.norm2.parameters())
        params.extend(self.ffn_linear1.parameters())
        params.extend(self.ffn_activation.parameters())
        params.extend(self.ffn_linear2.parameters())
        return params

# --------------------------------------------------------------------------
# 8. Modell Definition (SimpleModel) (Angepasst für PE cleanup)
# --------------------------------------------------------------------------
class SimpleModel:
    def __init__(self, config):
        self.config = config
        vocab_size = config['vocab_size']
        embed_dim = config['embed_dim']
        num_heads = config['num_heads']
        d_ff = config['d_ff']
        num_layers = config['num_layers']
        max_len = config['max_len']
        self.embed_dim = embed_dim
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.encoder_layers = [TransformerBlock(embed_dim, num_heads, d_ff) for _ in range(num_layers)]
        self.final_norm = LayerNorm(embed_dim)
        self.output_layer = Linear(embed_dim, vocab_size)
        print(f"[Model] Created SimpleModel: {num_layers}L, Emb:{embed_dim}, Heads:{num_heads}, dFF:{d_ff}")
    def __call__(self, input_ids: np.ndarray, mask=None) -> OclTensor:
        if input_ids.ndim != 2:
            raise ValueError("Input IDs must be 2D (Batch, SeqLen)")
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits
    def parameters(self):
        params = []
        params.extend(self.embedding.parameters())
        for layer in self.encoder_layers:
            params.extend(layer.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.output_layer.parameters())
        return params
    def free_memory(self):
        print("[Model] Freeing model parameters and PE buffer...")
        freed_count = 0
        for p in self.parameters():
            if isinstance(p, OclTensor):
                p.free_memory()
                freed_count += 1
        if hasattr(self.pos_encoding, 'free_memory'):
            self.pos_encoding.free_memory()
        print(f"  Freed {freed_count} parameter tensors and PE buffer.")
    def train(self):
        pass
    def eval(self):
        pass
    def load_state_dict(self, state_dict):
        model_params = self.parameters()
        param_map = {i: p for i, p in enumerate(model_params)}
        if len(state_dict) != len(param_map):
            raise ValueError(f"State dict size mismatch: expected {len(param_map)}, got {len(state_dict)}")
        print(f"[Model] Loading {len(state_dict)} parameters from state dict...")
        load_errors = 0
        loaded_indices = set()
        for key, param_data_host in state_dict.items():
            try:
                if not key.startswith('param_'):
                    raise KeyError(f"Invalid key format: {key}")
                param_index = int(key.split('_')[1])
                loaded_indices.add(param_index)
                if param_index not in param_map:
                    print(f"  ERROR: Index {param_index} from key '{key}' not in model.")
                    load_errors += 1
                    continue
                param = param_map[param_index]
                if param.shape != param_data_host.shape:
                    print(f"  ERROR: Shape mismatch param {param_index}: expect {param.shape}, got {param_data_host.shape}")
                    load_errors += 1
                    continue
                if param.data and param.data._allocated:
                    param.data.write(param_data_host.astype(FP_TYPE))
                else:
                    print(f"  ERROR: Target param buffer {param_index} not allocated")
                    load_errors += 1
                    continue
                if hasattr(param, '_is_embedding_weight') and param._is_embedding_weight:
                    assoc_layer_ref = getattr(param, '_associated_layer', None)
                    assoc_layer = assoc_layer_ref() if assoc_layer_ref else None
                    if assoc_layer:
                        assoc_layer.update_host_weight_from_gpu()
            except (KeyError, ValueError, IndexError) as e:
                print(f"  ERROR processing key '{key}': {e}")
                load_errors += 1
            except Exception as e:
                print(f"  UNEXPECTED ERROR loading key '{key}': {e}")
                load_errors += 1
        missing_indices = set(param_map.keys()) - loaded_indices
        if missing_indices:
            print(f"  WARN: Indices missing in state dict: {sorted(list(missing_indices))}")
        if load_errors > 0:
            raise RuntimeError(f"Failed to load state dict due {load_errors} errors.")
        else:
            print("[Model] Successfully loaded parameters.")

# --------------------------------------------------------------------------
# 9. Optimizer (AdamOptimizer) - Unverändert
# --------------------------------------------------------------------------
class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params_dict = {id(p): p for p in parameters if isinstance(p, OclTensor) and p.requires_grad}
        self.params = list(self.params_dict.values())
        self.param_ids_ordered = [id(p) for p in self.params]
        print(f"[AdamOpt] Tracking {len(self.params)} trainable parameters.")
        self.lr, self.beta1, self.beta2, self.eps, self.weight_decay = lr, betas[0], betas[1], eps, weight_decay
        self.t = 0
        self.m_buffers, self.v_buffers = {}, {}
        for p_id in self.param_ids_ordered:
            p = self.params_dict[p_id]
            if p and p.numel > 0:
                state_nbytes = p.numel * ADAM_STATE_SIZE
                self.m_buffers[p_id] = GPUBuffer(state_nbytes, f"adam_m_{p_id}", _init_zeros=True, dtype=ADAM_STATE_TYPE)
                self.v_buffers[p_id] = GPUBuffer(state_nbytes, f"adam_v_{p_id}", _init_zeros=True, dtype=ADAM_STATE_TYPE)
    def zero_grad(self):
        [p.zero_grad() for p in self.params if p.requires_grad]
    def step(self):
        self.t += 1
        for p_id in self.param_ids_ordered:
            p = self.params_dict.get(p_id)
            if not p or p.numel == 0 or not p.requires_grad or not hasattr(p, '_grad_buffer') or p._grad_buffer is None:
                continue
            param_ptr = p.data.ptr if p.data else None
            grad_ptr = p._grad_buffer.ptr if p._grad_buffer else None
            m_ptr = self.m_buffers[p_id].ptr if p_id in self.m_buffers else None
            v_ptr = self.v_buffers[p_id].ptr if p_id in self.v_buffers else None
            if not all([param_ptr, grad_ptr, m_ptr, v_ptr]):
                continue
            check_success(ocl.execute_adam_update_on_gpu(GPU_ID, param_ptr, grad_ptr, m_ptr, v_ptr, p.numel, self.t, c_float(self.lr), c_float(self.beta1), c_float(self.beta2), c_float(self.eps), c_float(self.weight_decay)), f"adam step p_id {p_id}")
            if hasattr(p, '_is_embedding_weight') and p._is_embedding_weight:
                assoc_layer_ref = getattr(p, '_associated_layer', None)
                assoc_layer = assoc_layer_ref() if assoc_layer_ref else None
                if assoc_layer:
                    assoc_layer.update_host_weight_from_gpu()
    def free_memory(self):
        print("[AdamOpt] Freeing Adam optimizer states...")
        freed_m, freed_v = 0, 0
        for buf in self.m_buffers.values():
            buf.free()
            freed_m += 1
        for buf in self.v_buffers.values():
            buf.free()
            freed_v += 1
        self.m_buffers.clear()
        self.v_buffers.clear()
        print(f"  Freed {freed_m} m-buffers, {freed_v} v-buffers.")
    def state_dict(self):
        state = {'t': self.t, 'm': {}, 'v': {}}
        read_count = 0
        for i, p_id in enumerate(self.param_ids_ordered):
            if p_id in self.m_buffers and p_id in self.v_buffers:
                p = self.params_dict.get(p_id)
                m_buf = self.m_buffers[p_id]
                v_buf = self.v_buffers[p_id]
                if p and m_buf._allocated and v_buf._allocated:
                    try:
                        m_data = m_buf.read(p.shape, dtype=ADAM_STATE_TYPE)
                        v_data = v_buf.read(p.shape, dtype=ADAM_STATE_TYPE)
                        state['m'][f'state_{i}'] = m_data
                        state['v'][f'state_{i}'] = v_data
                        read_count += 1
                    except Exception as e:
                        print(f"  WARN: Failed to read Adam state for param index {i}: {e}")
        return state
    def load_state_dict(self, state_dict):
        self.t = state_dict.get('t', 0)
        m_state = state_dict.get('m', {})
        v_state = state_dict.get('v', {})
        print(f"[AdamOpt] Loading optimizer state (t={self.t})...")
        load_errors = 0
        loaded_count = 0
        for i, p_id in enumerate(self.param_ids_ordered):
            key = f'state_{i}'
            if p_id in self.m_buffers and p_id in self.v_buffers:
                m_buf = self.m_buffers[p_id]
                v_buf = self.v_buffers[p_id]
                if not m_buf._allocated or not v_buf._allocated:
                    print(f"  ERROR: Optim buffer param {i} not allocated.")
                    load_errors += 1
                    continue
                if key in m_state and key in v_state:
                    m_data_host = m_state[key]
                    v_data_host = v_state[key]
                    p = self.params_dict.get(p_id)
                    if p and p.shape == m_data_host.shape and p.shape == v_data_host.shape:
                        try:
                            m_buf.write(m_data_host.astype(ADAM_STATE_TYPE))
                            v_buf.write(v_data_host.astype(ADAM_STATE_TYPE))
                            loaded_count += 1
                        except Exception as e:
                            print(f"  ERROR writing Adam state param {i}: {e}")
                            load_errors += 1
                    else:
                        print(f"  ERROR: Shape mismatch or param not found optim state {i}.")
                        load_errors += 1
                else:
                    print(f"  ERROR: Missing state key '{key}' param {i}.")
                    load_errors += 1
            elif key in m_state or key in v_state:
                print(f"  WARN: State found key '{key}' but no buffer exists (param_id {p_id}).")
        if load_errors > 0:
            raise RuntimeError(f"Failed load optim state due {load_errors} errors.")
        else:
            print(f"[AdamOpt] Successfully loaded optim state for {loaded_count} parameters.")

# --------------------------------------------------------------------------
# 10. Tokenizer (TinyTokenizer) - Unverändert
# --------------------------------------------------------------------------
class TinyTokenizer:
    def __init__(self, vocab=None, inv_vocab=None):
        if vocab and inv_vocab:
            self._vocab, self._inv_vocab = vocab, inv_vocab
        else:
            chars = sorted(list(set("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'\n\"-()")))
            self._vocab = {ch: i+1 for i, ch in enumerate(chars)}
            self._vocab["<pad>"] = 0
            self._vocab["<unk>"] = len(self._vocab)
            self._inv_vocab = {i: ch for ch, i in self._vocab.items()}
        self.vocab_size = len(self._vocab)
        print(f"[Tokenizer] Char vocab size: {self.vocab_size}")
    def encode(self, text, max_len):
        ids = [self._vocab.get(ch, self._vocab["<unk>"]) for ch in text.lower()][:max_len]
        padding = [self._vocab["<pad>"]] * (max_len - len(ids))
        ids.extend(padding)
        return ids[:max_len]
    def decode(self, ids):
        return ''.join(self._inv_vocab.get(i, '?') for i in ids if i != self._vocab["<pad>"])
    def get_vocab(self):
        return self._vocab.copy()
    def get_inv_vocab(self):
        return self._inv_vocab.copy()

# --------------------------------------------------------------------------
# 11. Loss Funktion & Backward Start (CPU) (unverändert)
# --------------------------------------------------------------------------
def cross_entropy_loss_and_backward(logits: OclTensor, target_ids: np.ndarray):
    if logits.numel == 0 or target_ids.size == 0:
        return 0.0
    logits_host = logits.to_host()
    b, s, v = logits_host.shape
    if target_ids.shape != (b, s):
        raise ValueError(f"Target shape {target_ids.shape} mismatch logits {logits.shape[:2]}")
    logits_flat = logits_host.reshape(-1, v)
    targets_flat = target_ids.reshape(-1)
    N = logits_flat.shape[0]
    logits_stable = logits_flat - np.max(logits_flat, axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    eps = 1e-9
    target_idx = np.arange(N)
    valid_mask = (targets_flat != 0)
    num_valid = np.sum(valid_mask)
    correct_probs = probs[target_idx[valid_mask], targets_flat[valid_mask]]
    correct_logprobs = -np.log(correct_probs + eps)
    loss = np.sum(correct_logprobs) / num_valid if num_valid > 0 else 0.0
    grad_logits_flat = probs.copy()
    grad_logits_flat[target_idx[valid_mask], targets_flat[valid_mask]] -= 1
    grad_logits_flat /= num_valid if num_valid > 0 else 1.0
    grad_logits_flat[~valid_mask, :] = 0
    grad_logits_host = grad_logits_flat.reshape(b, s, v).astype(FP_TYPE)
    grad_logits_tensor = None
    try:
        grad_logits_tensor = OclTensor(grad_logits_host)
        if OclTensor._enable_grad and logits.requires_grad:
            if not hasattr(logits, '_ctx') or logits._ctx is None:
                print(f"WARN LossBW: logits tensor {id(logits)} has no context!")
                sys.stdout.flush()
            else:
                logits.backward(gradient=grad_logits_tensor)
    finally:
        if grad_logits_tensor:
            grad_logits_tensor.free_memory()
    return float(loss)

# --------------------------------------------------------------------------
# 12. Init / Shutdown / Cleanup Funktionen (ocl_initialize prüft jetzt mehr Flags)
# --------------------------------------------------------------------------
_global_weak_tensor_refs = []
_optimizer_ref = None
_model_ref = None
def register_tensor(tensor):
    _global_weak_tensor_refs.append(weakref.ref(tensor))
def remove_tensor_from_registry(tensor_to_remove):
    global _global_weak_tensor_refs
    _global_weak_tensor_refs = [ref for ref in _global_weak_tensor_refs if ref() is not None and ref() is not tensor_to_remove]
def register_optimizer(optimizer):
    global _optimizer_ref
    _optimizer_ref = optimizer
def register_model(model):
    global _model_ref
    _model_ref = weakref.ref(model)
def cleanup_registered_tensors():
    global _global_weak_tensor_refs
    print(f"[Cleanup] Freeing ~{len(_global_weak_tensor_refs)} tensors...")
    count = 0
    refs_copy = list(_global_weak_tensor_refs)
    for ref in refs_copy:
        tensor = ref()
        if tensor:
            try:
                tensor.free_memory()
                count += 1
            except Exception as e:
                print(f"  Warn: Error freeing tensor {id(tensor)}: {e}")
    _global_weak_tensor_refs.clear()
    print(f"[Cleanup] Freed {count} tensors.")
def ocl_initialize(device_index=0):
    global LayerNormBackwardContext, GPU_ID, HAS_BMM_BATCHED, HAS_TRANSPOSE_LAST_TWO, HAS_TRANSPOSE_12_BATCHED, HAS_REDUCE_SUM, HAS_ADD_BROADCAST_PE, HAS_EMBEDDING_LOOKUP, HAS_EMBEDDING_BACKWARD
    if OclTensor._ocl_initialized:
        print("[ocl_init] Already initialized.")
        return True
    GPU_ID = device_index
    print(f"[ocl_init] Initializing OpenCL for GPU {GPU_ID}...")
    try:
        success = ocl.initialize_gpu(GPU_ID)
        check_success(success, "Init GPU")
        OclTensor._ocl_initialized = True
        OclTensor._enable_grad = True
        print(f"[ocl_init] GPU {GPU_ID} OK.")
        def fixed_layer_norm(self, eps=1e-5):
            if len(self.shape) < 1:
                raise ValueError("LayerNorm expects >= 1D tensor")
            if self.numel == 0:
                return OclTensor.empty(self.shape, requires_grad=self.requires_grad)
            row_size = self.shape[-1]
            num_rows = self.numel // row_size if row_size > 0 else 0
            requires_grad = self.requires_grad and OclTensor._enable_grad
            out = OclTensor.empty(self.shape, requires_grad=requires_grad)
            if num_rows > 0 and row_size > 0:
                assert self.data and self.data.ptr, "LayerNorm Input buffer invalid"
                assert out.data and out.data.ptr, "LayerNorm Output buffer invalid"
                check_success(ocl.execute_layernorm_on_gpu(GPU_ID, self.data.ptr, out.data.ptr, int(num_rows), int(row_size), c_float(eps)), "layer_norm gpu")
            if out.requires_grad:
                out._ctx = LayerNormBackwardContext(self, float(eps))
            return out
        OclTensor.layer_norm = fixed_layer_norm
        print("[ocl_init] Applied LayerNorm BW fix.")
        print("[ocl_init] Optional Kernel Status:")
        print(f"  - Batched MatMul: {'YES' if HAS_BMM_BATCHED else 'NO (CPU fallback)'}")
        print(f"  - Transpose Last Two (-2,-1): {'YES' if HAS_TRANSPOSE_LAST_TWO else 'NO (CPU fallback)'}")
        print(f"  - Transpose 1<->2 (4D): {'YES' if HAS_TRANSPOSE_12_BATCHED else 'NO (CPU fallback)'}")
        print(f"  - ReduceSum (Bias Grad): {'YES' if HAS_REDUCE_SUM else 'NO (CPU fallback)'}")
        print(f"  - AddBroadcastPE (PosEnc): {'YES' if HAS_ADD_BROADCAST_PE else 'NO (CPU fallback)'}")
        print(f"  - Embedding Lookup GPU: {'YES' if HAS_EMBEDDING_LOOKUP else 'NO (CPU only)'}")
        print(f"  - Embedding Backward GPU: {'YES' if HAS_EMBEDDING_BACKWARD else 'NO (CPU only)'}")
        return True
    except Exception as e:
        print(f"[ocl_init] FATAL Initialization Error: {e}")
        import traceback
        traceback.print_exc()
        OclTensor._ocl_initialized = False
        return False

def ocl_shutdown():
    global _optimizer_ref, _model_ref
    if not OclTensor._ocl_initialized:
        print("[ocl_shutdown] Not initialized or already shut down.")
        return
    print("[ocl_shutdown] Starting...")
    OclTensor._ocl_initialized = False
    if _optimizer_ref:
        print("  Freeing optimizer states...")
        try:
            _optimizer_ref.free_memory()
        except Exception as e:
            print(f"  Warn: Error freeing optimizer: {e}")
        finally:
            _optimizer_ref = None
    if _model_ref:
        model = _model_ref()
        if model:
            print("  Freeing model parameters & PE buffer...")
            try:
                model.free_memory()
            except Exception as e:
                print(f"  Warn: Error freeing model memory: {e}")
        _model_ref = None
    cleanup_registered_tensors()
    print("  Calling C driver shutdown...")
    try:
        ocl.shutdown_driver()
        print("  C driver shutdown complete.")
    except Exception as e:
        print(f"  Warn: Error during C driver shutdown: {e}")
    print("[ocl_shutdown] Finished.")

# --------------------------------------------------------------------------
# 13. Checkpoint Save/Load Funktionen (Unverändert)
# --------------------------------------------------------------------------
def save_checkpoint(filename, model, optimizer, tokenizer, epoch, best_val_loss, config):
    print(f"--> Saving checkpoint to {filename} ...")
    t_save_start = time.time()
    try:
        model_state_dict = {}
        model_params = model.parameters()
        for i, param in enumerate(model_params):
            if param.data and param.data._allocated:
                model_state_dict[f'param_{i}'] = param.to_host()
            else:
                print(f"  WARN: Param {i} no data, skipping save.")
        optimizer_state_dict = optimizer.state_dict()
        tokenizer_vocab = tokenizer.get_vocab()
        tokenizer_inv_vocab = tokenizer.get_inv_vocab()
        checkpoint_data = {
            'config': np.array([pickle.dumps(config)], dtype=object),
            'model_state': np.array([pickle.dumps(model_state_dict)], dtype=object),
            'optimizer_state': np.array([pickle.dumps(optimizer_state_dict)], dtype=object),
            'tokenizer_vocab': np.array([pickle.dumps(tokenizer_vocab)], dtype=object),
            'tokenizer_inv_vocab': np.array([pickle.dumps(tokenizer_inv_vocab)], dtype=object),
            'epoch': np.array([epoch], dtype=np.int32),
            'best_val_loss': np.array([best_val_loss], dtype=np.float64)
        }
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez_compressed(filename, **checkpoint_data)
        save_duration = time.time() - t_save_start
        print(f"--> Checkpoint saved successfully ({save_duration:.2f}s)")
        return True
    except Exception as e:
        print(f"!!! ERROR saving checkpoint to {filename}: {e} !!!")
        return False

def load_checkpoint(filename):
    if not os.path.exists(filename):
        print(f"[Load Checkpoint] File not found: {filename}")
        return None
    print(f"<-- Loading checkpoint from {filename} ...")
    t_load_start = time.time()
    try:
        data = np.load(filename, allow_pickle=True)
        config = pickle.loads(data['config'][0])
        tokenizer_vocab = pickle.loads(data['tokenizer_vocab'][0])
        tokenizer_inv_vocab = pickle.loads(data['tokenizer_inv_vocab'][0])
        tokenizer = TinyTokenizer(vocab=tokenizer_vocab, inv_vocab=tokenizer_inv_vocab)
        if 'vocab_size' not in config:
            config['vocab_size'] = tokenizer.vocab_size
        if not OclTensor._ocl_initialized:
            raise RuntimeError("OpenCL must be initialized before loading checkpoint.")
        model = SimpleModel(config)
        optimizer = AdamOptimizer(model.parameters(), lr=config.get('lr', 0.001), weight_decay=config.get('wd', 0.01))
        model_state_dict = pickle.loads(data['model_state'][0])
        model.load_state_dict(model_state_dict)
        optimizer_state_dict = pickle.loads(data['optimizer_state'][0])
        optimizer.load_state_dict(optimizer_state_dict)
        start_epoch = int(data['epoch'][0]) + 1
        best_val_loss = float(data['best_val_loss'][0])
        load_duration = time.time() - t_load_start
        print(f"<-- Checkpoint loaded (Epoch {start_epoch-1}, BestLoss {best_val_loss:.5f}, {load_duration:.2f}s)")
        return model, optimizer, tokenizer, start_epoch, best_val_loss, config
    except Exception as e:
        print(f"!!! ERROR loading checkpoint {filename}: {e} !!!")
        import traceback
        traceback.print_exc()
        return None

# --------------------------------------------------------------------------
# 14. Trainings- und Hilfsfunktionen (print_config angepasst)
# --------------------------------------------------------------------------
def print_config(config):
    print("\n--- Training Configuration [Ultimate Attempt 7 - PE GPU Add] ---")
    print(f" Model: {config['num_layers']}L | Embed:{config['embed_dim']} | Heads:{config['num_heads']} | dFF:{config['d_ff']}")
    print(f" Data: Vocab Size:{config['vocab_size']} | Max Seq Len:{config['max_len']}")
    print(f" Train: Batch Size:{config['batch_size']} | Epochs:{config['num_epochs']} | LR:{config['lr']:.1e} | WeightDecay:{config['wd']}")
    print(f" Precision: FP Type:{FP_TYPE.__name__} | Adam State:{ADAM_STATE_TYPE.__name__}")
    print(f" GPU Kernels Found:")
    print(f"  - BMM Batched: {'YES' if HAS_BMM_BATCHED else 'NO (CPU fallback)'}")
    print(f"  - Transpose Last Two (-2,-1): {'YES' if HAS_TRANSPOSE_LAST_TWO else 'NO (CPU fallback)'}")
    print(f"  - Transpose 1<->2 (4D): {'YES' if HAS_TRANSPOSE_12_BATCHED else 'NO (CPU fallback)'}")
    print(f"  - ReduceSum (Bias Grad): {'YES' if HAS_REDUCE_SUM else 'NO (CPU fallback)'}")
    print(f"  - AddBroadcastPE (PosEnc): {'YES' if HAS_ADD_BROADCAST_PE else 'NO (CPU fallback)'}")
    print(f"  - Embedding Lookup GPU: {'YES' if HAS_EMBEDDING_LOOKUP else 'NO (CPU only)'}")
    print(f"  - Embedding Backward GPU: {'YES' if HAS_EMBEDDING_BACKWARD else 'NO (CPU only)'}")
    print("-" * 50)
    print(" Known Limitations / Workarounds:")
    if not HAS_EMBEDDING_LOOKUP or not HAS_EMBEDDING_BACKWARD:
        print("  - CPU Embedding Lookup & Backward Gradient")
    if not HAS_REDUCE_SUM:
        print("  - CPU Bias Gradient Reduction")
    if not HAS_TRANSPOSE_LAST_TWO or not HAS_TRANSPOSE_12_BATCHED:
        print("  - CPU Transpose for some dimension swaps")
    print("  - CPU Cross Entropy Loss Calculation & Grad Input")
    print("  - No Dropout Layers implemented")
    print("-" * 50 + "\n")

def run_inference(model, tokenizer, max_len):
    test_input = "opencl test"
    test_ids_list = tokenizer.encode(test_input, max_len)
    test_ids_np = np.array([test_ids_list], dtype=np.int32)
    print(f"\n--- Running Inference --- | Input: '{test_input}'")
    inf_start_time = time.time()
    pred_logits_gpu = None
    try:
        model.eval()
        model.embedding.update_host_weight_from_gpu()
        with OclTensor.no_grad():
            pred_logits_gpu = model(test_ids_np)
        if pred_logits_gpu and pred_logits_gpu.data and pred_logits_gpu.data._allocated:
            pred_logits_host = pred_logits_gpu.to_host()
            pred_ids = np.argmax(pred_logits_host[0], axis=-1)
            pred_text = tokenizer.decode(pred_ids)
            print(f"Predicted Text: '{pred_text}'")
        else:
            print("Error: Inference produced invalid output.")
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        if pred_logits_gpu:
            pred_logits_gpu.free_memory()
        inf_duration = time.time() - inf_start_time
        print(f"Inference Time: {inf_duration:.4f}s")
    print(f"--------------------------\n")

def train(load_checkpoint_path=None, save_checkpoint_dir="."):
    # --- Standard Config ---
    config = {
        'max_len': 64, 'batch_size': 16, 'embed_dim': 64, 'num_heads': 4,
        'd_ff': 64 * 4, 'num_layers': 1, 'lr': 5e-4, 'num_epochs': 10, # Erhöhe num_epochs zum Testen
        'wd': 0.01, 'val_split': 0.1, 'data_file': "input.txt",
        'checkpoint_filename': "best_model.npz" # Default filename
    }
    # Diese Werte werden beim Laden überschrieben
    start_epoch = 1
    best_val_loss = float('inf')
    model = None
    optimizer = None
    tokenizer = None

    # --- Checkpoint laden, falls angegeben ---
    if load_checkpoint_path and os.path.exists(load_checkpoint_path):
        load_result = load_checkpoint(load_checkpoint_path)
        if load_result:
            model, optimizer, tokenizer, start_epoch, best_val_loss, loaded_config = load_result
            # Update config mit geladenen Werten (insbesondere vocab_size etc.)
            config.update(loaded_config)
            register_optimizer(optimizer) # Wichtig: Nach dem Laden registrieren!
            register_model(model) # Wichtig: Nach dem Laden registrieren!
        else:
            print(f"WARN: Failed to load checkpoint '{load_checkpoint_path}'. Starting from scratch.")
            load_checkpoint_path = None # Verhindert weiteren Ladeversuch
    # -----------------------------------------

    # --- Initialisierung, falls nicht geladen ---
    if tokenizer is None:
        tokenizer = TinyTokenizer()
    if 'vocab_size' not in config: # Setze vocab_size falls nicht geladen
        config['vocab_size'] = tokenizer.vocab_size
    if model is None:
        model = SimpleModel(config)
        register_model(model) # Neu erstelltes Modell registrieren
    if optimizer is None:
        optimizer = AdamOptimizer(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        register_optimizer(optimizer) # Neu erstellten Optimizer registrieren
    # ------------------------------------------

    # --- Daten laden und vorbereiten ---
    try:
        with open(config['data_file'], 'r', encoding='utf-8') as f: text_data = f.read()
        print(f"[Data] Loaded {len(text_data)} chars from {config['data_file']}"); sys.stdout.flush()
        if len(text_data) < config['max_len'] * 2: print(f"WARN: Data in {config['data_file']} maybe too short"); sys.stdout.flush()
    except Exception as e: print(f"FATAL: Error reading {config['data_file']}: {e}"); return

    input_ids_all, target_ids_all = [], []; encoded_text = tokenizer.encode(text_data, max_len=len(text_data))
    for i in range(0, len(encoded_text) - config['max_len']):
        chunk_in = encoded_text[i : i + config['max_len']]
        chunk_tgt = encoded_text[i + 1 : i + config['max_len'] + 1]
        if len(chunk_in) == config['max_len'] and len(chunk_tgt) == config['max_len']:
            input_ids_all.append(chunk_in); target_ids_all.append(chunk_tgt)
    if not input_ids_all: print(f"FATAL: No sequences of length {config['max_len']} created from {config['data_file']}"); return

    input_ids_np, target_ids_np = np.array(input_ids_all, dtype=np.int32), np.array(target_ids_all, dtype=np.int32)
    num_samples = len(input_ids_np); print(f"[Data] Created {num_samples} sequences."); sys.stdout.flush()
    indices = np.arange(num_samples); np.random.shuffle(indices); split_idx = int(num_samples * (1 - config['val_split']))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    input_train, target_train = input_ids_np[train_indices], target_ids_np[train_indices]
    input_val, target_val = input_ids_np[val_indices], target_ids_np[val_indices]
    num_train_samples, num_val_samples = len(input_train), len(input_val)
    print(f"[Data] Train: {num_train_samples}, Val: {num_val_samples}"); sys.stdout.flush()
    # --- Ende Datenladen ---

    print_config(config); # Verwende das (ggf. aktualisierte) config dict
    print(f"--- Starting Training (Epochs {start_epoch} to {config['num_epochs']}) ---"); sys.stdout.flush()
    t_total_start = time.time();
    # Best Checkpoint Path
    best_checkpoint_filename = os.path.join(save_checkpoint_dir, config['checkpoint_filename'])

    for epoch in range(start_epoch, config['num_epochs'] + 1):
        epoch_train_loss, batch_times = 0.0, []; train_perm = np.random.permutation(num_train_samples)
        t_epoch_start = time.time(); output_tensor = None;
        print(f"--- Epoch {epoch} Start ---"); sys.stdout.flush() # DEBUG: Epoch start
        model.embedding.update_host_weight_from_gpu() # Sync host weight before epoch

        model.train() # Set model to training mode (placeholder)
        OclTensor._enable_grad = True # Ensure grad is enabled for training

        # --- Training Loop ---
        for i in range(0, num_train_samples, config['batch_size']):
            batch_num = i // config['batch_size'] + 1
            print(f"Epoch {epoch}, Batch {batch_num}: Start"); sys.stdout.flush() # DEBUG: Batch Start
            t_batch_start = time.time(); batch_idx = train_perm[i:i + config['batch_size']]; bs_curr = len(batch_idx)
            if bs_curr == 0:
                print(f"Epoch {epoch}, Batch {batch_num}: Skipping (0 size)"); sys.stdout.flush() # DEBUG: Skip
                continue
            input_b, target_b = input_train[batch_idx], target_train[batch_idx]
            try:
                print(f"Epoch {epoch}, Batch {batch_num}: Zero Grad Start"); sys.stdout.flush() # DEBUG
                optimizer.zero_grad();
                print(f"Epoch {epoch}, Batch {batch_num}: Zero Grad End / Fwd Start"); sys.stdout.flush() # DEBUG
                output_tensor = model(input_b) # Forward pass
                print(f"Epoch {epoch}, Batch {batch_num}: Fwd End / Loss Start"); sys.stdout.flush() # DEBUG
                loss = cross_entropy_loss_and_backward(output_tensor, target_b) # Loss & Backward pass
                print(f"Epoch {epoch}, Batch {batch_num}: Loss End (Loss={loss:.4f}) / Optim Step Start"); sys.stdout.flush() # DEBUG
                if np.isnan(loss) or np.isinf(loss):
                    print(f"\nWARN: NaN/Inf loss ({loss}) Epoch {epoch} Batch {batch_num}. Skipping step."); sys.stdout.flush() # DEBUG
                else:
                    optimizer.step() # Update weights
                print(f"Epoch {epoch}, Batch {batch_num}: Optim Step End"); sys.stdout.flush() # DEBUG
                epoch_train_loss += float(loss) * bs_curr # Accumulate loss
            except Exception as e:
                print(f"\n!!! ERROR Training Epoch {epoch} Batch {batch_num} !!!"); import traceback; traceback.print_exc(); print("!!! Skipping epoch !!!"); sys.stdout.flush(); break # Skip rest of epoch
            finally:
                # print(f"Epoch {epoch}, Batch {batch_num}: Free Output Start"); sys.stdout.flush() # DEBUG: Free start
                if output_tensor: output_tensor.free_memory(); output_tensor = None
                # print(f"Epoch {epoch}, Batch {batch_num}: Free Output End / Batch End"); sys.stdout.flush() # DEBUG: Free end / Batch end
            batch_times.append(time.time() - t_batch_start)
            # Optional: Add a small break/print after a few batches to see if it progresses
            # if batch_num % 10 == 0:
            #    print(f"   ... processed batch {batch_num}")
            #    sys.stdout.flush()
        # --- End Training Loop ---

        avg_train_loss = epoch_train_loss / num_train_samples if num_train_samples > 0 else 0.0
        avg_batch_time = np.mean(batch_times) if batch_times else 0; epoch_val_loss = 0.0; val_output_tensor = None

        # --- Validation Loop ---
        print(f"--- Epoch {epoch} Validation Start ---"); sys.stdout.flush() # DEBUG: Val Start
        if num_val_samples > 0:
            model.eval(); OclTensor._enable_grad = False
            model.embedding.update_host_weight_from_gpu() # Sync host weight for validation lookup
            for i in range(0, num_val_samples, config['batch_size']):
                val_batch_num = i // config['batch_size'] + 1
                # print(f"  Val Batch {val_batch_num}: Start"); sys.stdout.flush() # DEBUG: Val Batch Start
                batch_idx = np.arange(i, min(i + config['batch_size'], num_val_samples)); bs_curr = len(batch_idx)
                if bs_curr == 0: continue
                input_b, target_b = input_val[batch_idx], target_val[batch_idx]
                try:
                    # print(f"  Val Batch {val_batch_num}: Fwd Start"); sys.stdout.flush() # DEBUG
                    val_output_tensor = model(input_b)
                    # print(f"  Val Batch {val_batch_num}: Fwd End / Loss Start"); sys.stdout.flush() # DEBUG
                    loss = cross_entropy_loss_and_backward(val_output_tensor, target_b) # No backward
                    # print(f"  Val Batch {val_batch_num}: Loss End (Loss={loss:.4f})"); sys.stdout.flush() # DEBUG
                    if np.isnan(loss) or np.isinf(loss):
                        print(f"WARN: NaN/Inf validation loss ({loss}). Setting Val Loss to Inf."); sys.stdout.flush()
                        epoch_val_loss = float('inf'); break
                    epoch_val_loss += float(loss) * bs_curr
                except Exception as e:
                    print(f"\n!!! ERROR Validation Epoch {epoch} Batch {val_batch_num} !!!"); import traceback; traceback.print_exc()
                    epoch_val_loss = float('inf'); break
                finally:
                    # print(f"  Val Batch {val_batch_num}: Free Output Start"); sys.stdout.flush() # DEBUG
                    if val_output_tensor: val_output_tensor.free_memory(); val_output_tensor = None
                    # print(f"  Val Batch {val_batch_num}: Free Output End / Batch End"); sys.stdout.flush() # DEBUG
            OclTensor._enable_grad = True # Re-enable grad
            avg_val_loss = epoch_val_loss / num_val_samples if num_val_samples > 0 and np.isfinite(epoch_val_loss) else float('inf')
            print(f"--- Epoch {epoch} Validation End (AvgLoss={avg_val_loss:.5f}) ---"); sys.stdout.flush() # DEBUG: Val End

            # --- Checkpoint Speichern ---
            if avg_val_loss < best_val_loss:
                print(f"  Val loss improved ({best_val_loss:.5f} -> {avg_val_loss:.5f}). Saving best model...")
                best_val_loss = avg_val_loss
                if save_checkpoint_dir: # Nur speichern, wenn ein Verzeichnis angegeben ist
                     save_checkpoint(best_checkpoint_filename, model, optimizer, tokenizer, epoch, best_val_loss, config)
            # ----------------------------
        else:
            avg_val_loss = 0.0 # No validation data
            print(f"--- Epoch {epoch} Validation Skipped (No Val Data) ---"); sys.stdout.flush() # DEBUG

        dur_epoch = time.time() - t_epoch_start
        print(f"Epoch {epoch:>{len(str(config['num_epochs']))}}/{config['num_epochs']} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | AvgBatch: {avg_batch_time:.4f}s | EpochTime: {dur_epoch:.2f}s"); sys.stdout.flush()

        # Inference Periodisch oder am Ende
        if epoch == config['num_epochs'] or (epoch % 1 == 0 and epoch != 0): # Aktuell jede Epoche
             run_inference(model, tokenizer, config['max_len'])
    # --- Ende Epoch Loop ---

    t_total = time.time() - t_total_start; print(f"--- Training Finished --- (Total: {t_total:.2f}s)"); print(f"Best Val Loss: {best_val_loss:.5f}"); sys.stdout.flush()
    print("\n--- Running Final Inference with Last Model ---")
    run_inference(model, tokenizer, config['max_len'])
    if os.path.exists(best_checkpoint_filename):
        print(f"\n--- Running Final Inference with Best Saved Model ({best_checkpoint_filename}) ---")
        best_model_load_result = load_checkpoint(best_checkpoint_filename)
        if best_model_load_result:
            best_model, _, best_tokenizer, _, _, _ = best_model_load_result
            run_inference(best_model, best_tokenizer, config['max_len'])
            best_model.free_memory() # Explicitly free loaded best model
        else:
            print("  Failed load best model for final inference.")
# --------------------------------------------------------------------------
# 15. Main Guard & Ausführung (Angepasst für GPU ID)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OCL LLM Training Framework")
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint file to load and resume training.')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--gpu_id', type=int, default=0, help='Index of the OpenCL GPU device to use.')
    args = parser.parse_args()

    GPU_ID = args.gpu_id
    print(f"[Main] Attempting to use GPU Device Index: {GPU_ID}")

    main_start_time = time.time()
    initialization_success = False
    try:
        initialization_success = ocl_initialize(GPU_ID)
        if initialization_success:
            train(load_checkpoint_path=args.load_checkpoint, save_checkpoint_dir=args.save_dir)
        else:
            print("\nSkipping training: OpenCL initialization failure.")
    except Exception as e:
        print("\n--- UNHANDLED EXCEPTION ---")
        import traceback
        traceback.print_exc()
        print(f"-------------------------")
    finally:
        print("\n--- Final Shutdown ---")
        ocl_shutdown()
        print("----------------------")
        total_time = time.time() - main_start_time
        print(f"--- OCL LLM Finished (Total: {total_time:.2f}s) ---")
