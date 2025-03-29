# OpenCL Neural Network Kernels Library

## Übersicht

Diese Bibliothek stellt eine Sammlung von hardwarebeschleunigten neuronalen Netzwerkoperationen bereit, die OpenCL zur Ausführung auf kompatiblen GPUs und anderen Beschleunigern nutzen. Sie ist als C-Bibliothek (.dll unter Windows, .so unter Linux) konzipiert, die von anderen Sprachen (z. B. Python über `ctypes` oder `cffi`) aufgerufen werden kann, um rechenintensive Aufgaben in Deep-Learning-Workflows zu beschleunigen.

Die Bibliothek umfasst grundlegende Operationen wie Matrixmultiplikation, Aktivierungsfunktionen, Normalisierung, Optimierer und spezifischere Layer wie Embedding-Lookups, die sowohl für Inferenz als auch für das Training von Modellen notwendig sind.

## Hauptmerkmale

*   **OpenCL-Backend:** Nutzt OpenCL für breite Hardwarekompatibilität über verschiedene Anbieter (AMD, NVIDIA, Intel, etc.) und Plattformen hinweg.
*   **Umfangreiche Kernel-Sammlung:** Bietet eine Vielzahl gängiger NN-Operationen:
    *   Matrixmultiplikation (Standard, Batched)
    *   Softmax (Row-wise, Numerisch Stabil)
    *   LogSoftmax (Row-wise, Numerisch Stabil)
    *   GELU Aktivierung (Elementweise)
    *   Elementweise Addition & Multiplikation
    *   Layer Normalization (Row-wise)
    *   Transpose (2D, Batched letzte zwei Dimensionen, Batched Dimensionen 1&2 für 4D)
    *   Embedding Lookup
    *   Adam Optimizer Update
    *   Cross-Entropy Loss & Gradient (mit LogSoftmax kombiniert)
    *   Positional Encoding Addition (Broadcast)
    *   Hilfsfunktionen (Clone/Copy, Reduce Sum)
*   **Forward- und Backward-Pässe:** Enthält Kernel für den Backward-Pass vieler Operationen, was das Training von Modellen ermöglicht (z. B. `matmul_backward`, `gelu_backward`, `layernorm_backward`, `embedding_backward`).
*   **Atomare Operationen:** Nutzt (falls vom Gerät unterstützt) atomare Operationen für korrekte Gradientenakkumulation im Embedding-Backward-Pass.
*   **Dynamische Präzision:** Konfigurierbar für verschiedene Fließkommatypen (aktuell `float` via `KERNEL_FP_TYPE`).
*   **Plattformübergreifend:** Enthält bedingte Kompilierung für Linux und Windows (`DLLEXPORT`, Header).
*   **Simulationsschicht:** Bietet Dummy-Implementierungen (`simulated_*` Funktionen), um die Kompilierung und grundlegende Tests auch ohne installierte OpenCL-Treiber oder GPU zu ermöglichen.
*   **Fehlerbehandlung:** Gibt Fehler über `stderr` aus und verwendet Rückgabewerte zur Signalisierung von Erfolg/Misserfolg.

## Voraussetzungen

*   **C-Compiler:** Ein moderner C-Compiler (GCC, Clang, MSVC).
*   **OpenCL SDK & ICD Loader:**
    *   Eine funktionierende OpenCL-Implementierung für Ihre Zielhardware (GPU-Treiber von AMD, NVIDIA, Intel oder alternative Implementierungen wie POCL).
    *   Die OpenCL-Header (`CL/cl.h` oder `OpenCL/cl.h`) und die Laufzeitbibliothek zum Linken (`OpenCL.lib` unter Windows, `-lOpenCL` unter Linux).
*   **Aufrufende Umgebung (optional):** Z. B. Python mit `ctypes` oder `cffi`, um die kompilierte Bibliothek zu laden und die Funktionen aufzurufen.

## Kompilierung & Installation

Diese Bibliothek muss als Shared Library (.so unter Linux, .dll unter Windows) kompiliert werden. Es wird kein spezifisches Build-System (wie CMake) bereitgestellt, daher hier Beispielbefehle für gängige Compiler:

**Linux (GCC/Clang):**

```bash
# Kompilieren Sie den C-Code zu einer Shared Object (.so) Datei
# Stellen Sie sicher, dass der OpenCL-Include-Pfad korrekt ist (-I)
# Linken Sie gegen die OpenCL-Bibliothek (-lOpenCL)
gcc -shared -o libnnkernels.so -fPIC your_source_file.c -I/path/to/opencl/include -lOpenCL -O3 -Wall -Wextra

# Oder mit Clang:
# clang -shared -o libnnkernels.so -fPIC your_source_file.c -I/path/to/opencl/include -lOpenCL -O3 -Wall -Wextra
```

*   Ersetzen Sie `your_source_file.c` durch den Namen Ihrer C-Quelldatei.
*   Passen Sie `/path/to/opencl/include` an den Speicherort Ihrer `CL/cl.h` an (oft `/usr/include` oder spezifische SDK-Pfade).
*   `-fPIC` ist für Shared Libraries notwendig.
*   `-O3` aktiviert Optimierungen.
*   `-Wall -Wextra` aktivieren nützliche Warnungen.

**Windows (MSVC):**

Verwenden Sie die Developer Command Prompt für Visual Studio.

```bat
REM Kompilieren Sie den C-Code zu einer Dynamic Link Library (.dll)
REM Stellen Sie sicher, dass der OpenCL-Include-Pfad (-I) und der Lib-Pfad (-LIBPATH) korrekt sind
cl /LD your_source_file.c /I"C:\Path\To\OpenCL\include" /link /LIBPATH:"C:\Path\To\OpenCL\lib" OpenCL.lib /OUT:nnkernels.dll /W4 /O2
```

*   Ersetzen Sie `your_source_file.c` durch den Namen Ihrer C-Quelldatei.
*   Passen Sie die Pfade für Include (`CL/cl.h`) und Lib (`OpenCL.lib`) an Ihr OpenCL SDK an.
*   `/LD` erstellt eine DLL.
*   `/W4` aktiviert Warnstufe 4.
*   `/O2` aktiviert Optimierungen.
*   Das `_CRT_SECURE_NO_WARNINGS` Define am Anfang der C-Datei ist für die Kompatibilität mit einigen MSVC-Funktionen wie `sprintf` vorhanden.

Nach der Kompilierung haben Sie eine `libnnkernels.so` (Linux) oder `nnkernels.dll` (Windows), die Sie in Ihrer Anwendung verwenden können.

## Benutzung (API-Überblick)

Die Bibliothek folgt einem typischen Workflow für die GPU-Beschleunigung:

1.  **Initialisierung:** Rufen Sie `initialize_gpu(gpu_index)` auf, um eine Verbindung zum gewünschten OpenCL-Gerät herzustellen, den Kontext und die Command Queue zu erstellen und alle Kernel zu kompilieren. `gpu_index` ist normalerweise `0`. Gibt `1` bei Erfolg, `0` bei Fehler zurück.
2.  **Speicherallokation:** Verwenden Sie `allocate_gpu_memory(gpu_index, size)` um Speicher auf dem GPU-Gerät zu reservieren. Gibt ein opakes Handle (Pointer) zurück oder `NULL` bei Fehler.
3.  **Datentransfer (Host zu GPU):** Kopieren Sie Daten vom Host (CPU-RAM) auf die GPU mit `write_host_to_gpu_blocking(gpu_index, gpu_buffer_handle, offset, size, host_source_ptr)`.
4.  **Kernel-Ausführung:** Rufen Sie die entsprechende `execute_*_on_gpu` Funktion auf (z.B. `execute_matmul_on_gpu`), um den OpenCL-Kernel auf der GPU auszuführen. Übergeben Sie die Buffer-Handles und die notwendigen Dimensionen/Parameter. Diese Funktionen sind blockierend und warten auf die Fertigstellung des Kernels. Sie geben `1` bei Erfolg, `0` bei Fehler zurück.
5.  **Datentransfer (GPU zu Host):** Kopieren Sie Ergebnisse von der GPU zurück zum Host mit `read_gpu_to_host_blocking(gpu_index, gpu_buffer_handle, offset, size, host_destination_ptr)`.
6.  **Speicherfreigabe:** Geben Sie den auf der GPU allokierten Speicher mit `free_gpu_memory(gpu_index, buffer_handle)` wieder frei, wenn er nicht mehr benötigt wird.
7.  **Aufräumen:** Rufen Sie `shutdown_driver()` am Ende Ihres Programms auf, um alle OpenCL-Ressourcen (Kernel, Programme, Queue, Context) freizugeben.

**Wichtig:**
*   Alle exportierten Funktionen (`DLLEXPORT`) geben `1` für Erfolg und `0` für Fehler zurück (außer `allocate_gpu_memory`, das ein Handle oder `NULL` zurückgibt, und `free_gpu_memory`/`shutdown_driver`, die `void` sind). **Überprüfen Sie immer die Rückgabewerte!** Dies ist entscheidend für die Fehlersuche und robuste Anwendungen.
*   Fehlermeldungen werden nach `stderr` geschrieben. Überwachen Sie diesen Stream bei Problemen.
*   Die zurückgegebenen Buffer-Handles von `allocate_gpu_memory` sind opake Pointer und sollten nicht direkt dereferenziert werden. Behandeln Sie sie als undurchsichtige Ressourcen-IDs.

**Beispiel (Python mit ctypes - Pseudocode):**

```python
import ctypes
import numpy as np
import os

# Laden der kompilierten Bibliothek
lib_path = './libnnkernels.so' # oder 'nnkernels.dll' auf Windows
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Shared library not found at {lib_path}. Please compile it first.")
try:
    lib = ctypes.CDLL(lib_path)
except OSError as e:
    raise OSError(f"Error loading shared library: {e}. Ensure dependencies (OpenCL runtime) are available.")

# --- Funktionssignaturen definieren (Beispiel MatMul) ---
# Argumenttypen
lib.initialize_gpu.argtypes = [ctypes.c_int]
lib.initialize_gpu.restype = ctypes.c_int

lib.allocate_gpu_memory.argtypes = [ctypes.c_int, ctypes.c_size_t]
lib.allocate_gpu_memory.restype = ctypes.c_void_p # Buffer handle

lib.write_host_to_gpu_blocking.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
lib.write_host_to_gpu_blocking.restype = ctypes.c_int

lib.read_gpu_to_host_blocking.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
lib.read_gpu_to_host_blocking.restype = ctypes.c_int

lib.execute_matmul_on_gpu.argtypes = [
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.execute_matmul_on_gpu.restype = ctypes.c_int

lib.free_gpu_memory.argtypes = [ctypes.c_int, ctypes.c_void_p]
lib.free_gpu_memory.restype = None

lib.shutdown_driver.argtypes = []
lib.shutdown_driver.restype = None

# --- Beispiel: Matrixmultiplikation ---
gpu_id = 0
FP_TYPE = np.float32 # Muss mit KERNEL_FP_TYPE übereinstimmen!
itemsize = np.dtype(FP_TYPE).itemsize

# Dimensionen
B, M, N, K = 1, 128, 256, 64 # Beispiel: Standard MatMul (B=1)

# Host-Daten vorbereiten
A_host = np.random.rand(B, M, K).astype(FP_TYPE)
B_host = np.random.rand(K, N).astype(FP_TYPE) # Standard MatMul B ist 2D
C_host = np.zeros((B, M, N), dtype=FP_TYPE)

# 1. Initialisieren
print("Initializing GPU...")
if lib.initialize_gpu(gpu_id) == 0:
    print("GPU initialization failed. Check OpenCL setup and stderr logs.")
    exit(1)
print("GPU Initialized.")

# 2. Speicher allokieren
print("Allocating GPU memory...")
a_gpu = lib.allocate_gpu_memory(gpu_id, A_host.nbytes)
b_gpu = lib.allocate_gpu_memory(gpu_id, B_host.nbytes)
c_gpu = lib.allocate_gpu_memory(gpu_id, C_host.nbytes)

if not all([a_gpu, b_gpu, c_gpu]):
    print("GPU memory allocation failed.")
    # Wichtig: Auch im Fehlerfall Ressourcen freigeben, die bereits allokiert wurden
    if a_gpu: lib.free_gpu_memory(gpu_id, a_gpu)
    if b_gpu: lib.free_gpu_memory(gpu_id, b_gpu)
    if c_gpu: lib.free_gpu_memory(gpu_id, c_gpu)
    lib.shutdown_driver()
    exit(1)
print(f"GPU Buffers allocated: A={a_gpu}, B={b_gpu}, C={c_gpu}")

try:
    # 3. Daten auf GPU schreiben
    print("Writing data to GPU...")
    if lib.write_host_to_gpu_blocking(gpu_id, a_gpu, 0, A_host.nbytes, A_host.ctypes.data) == 0:
        raise RuntimeError("Failed to write A to GPU")
    if lib.write_host_to_gpu_blocking(gpu_id, b_gpu, 0, B_host.nbytes, B_host.ctypes.data) == 0:
        raise RuntimeError("Failed to write B to GPU")
    print("Data written.")

    # 4. MatMul-Kernel ausführen
    print("Executing MatMul kernel...")
    if lib.execute_matmul_on_gpu(gpu_id, a_gpu, b_gpu, c_gpu, B, M, N, K) == 0:
        raise RuntimeError("MatMul kernel execution failed.")
    print("Kernel executed.")

    # 5. Ergebnis zurücklesen
    print("Reading result from GPU...")
    if lib.read_gpu_to_host_blocking(gpu_id, c_gpu, 0, C_host.nbytes, C_host.ctypes.data) == 0:
        raise RuntimeError("Failed to read C from GPU")
    print("Result read.")

    # (Optional) Überprüfung
    # C_expected = A_host @ B_host # Funktioniert nur, wenn A 2D ist oder NumPy Broadcasting korrekt passt
    # print("Numpy result shape:", C_expected.shape)
    print("GPU result (first 5 elements):", C_host.flatten()[:5])
    # np.testing.assert_allclose(C_host, C_expected, rtol=1e-5, atol=1e-5)
    # print("Verification successful!")

except RuntimeError as e:
    print(f"An error occurred: {e}")
    print("Check stderr logs for more details from the C library.")

finally:
    # 6. Speicher freigeben (Immer im finally-Block!)
    print("Freeing GPU memory...")
    if a_gpu: lib.free_gpu_memory(gpu_id, a_gpu)
    if b_gpu: lib.free_gpu_memory(gpu_id, b_gpu)
    if c_gpu: lib.free_gpu_memory(gpu_id, c_gpu)
    print("GPU Memory freed.")

    # 7. Aufräumen (Immer im finally-Block!)
    print("Shutting down driver...")
    lib.shutdown_driver()
    print("Driver shut down.")

```

## Verfügbare Operationen

Die folgenden Operationen werden über die exportierten `execute_*` Funktionen bereitgestellt:

| GPUCommand Enum                             | C Export Funktion                             | Beschreibung                                                                                                 |
| :------------------------------------------ | :-------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| `COMMAND_MATRIX_MULTIPLY`                   | `execute_matmul_on_gpu`                       | Standard-Matrixmultiplikation (C[B,M,N] = A[B,M,K] @ B[K,N])                                                 |
| `COMMAND_SOFTMAX_ROWWISE`                   | `execute_softmax_on_gpu`                      | Zeilenweise Softmax-Normalisierung (numerisch stabil)                                                       |
| `COMMAND_GELU_ELEMENTWISE`                  | `execute_gelu_on_gpu`                         | Elementweise GELU-Aktivierungsfunktion                                                                      |
| `COMMAND_ADD_ELEMENTWISE`                   | `execute_add_on_gpu`                          | Elementweise Addition (C = A + B)                                                                           |
| `COMMAND_MUL_ELEMENTWISE`                   | `execute_mul_on_gpu`                          | Elementweise Multiplikation (C = A * B)                                                                     |
| `COMMAND_LAYER_NORM`                        | `execute_layernorm_on_gpu`                    | Zeilenweise Layer Normalization (über die letzte Dimension, ohne affine Parameter)                         |
| `COMMAND_CLONE`                             | `execute_clone_on_gpu`                        | Kopiert den Inhalt eines GPU-Buffers in einen anderen (GPU-zu-GPU-Kopie)                                     |
| `COMMAND_TRANSPOSE`                         | `execute_transpose_on_gpu`                    | Transponiert eine 2D-Matrix (Input: [rows, cols] -> Output: [cols, rows])                                 |
| `COMMAND_GELU_BACKWARD_ELEMENTWISE`         | `execute_gelu_backward_on_gpu`                | Berechnet den Gradienten für die GELU-Aktivierung (dL/dx)                                                    |
| `COMMAND_MATMUL_BACKWARD_DA`                | `execute_matmul_backward_on_gpu`              | Berechnet den Gradienten bzgl. A (dA = dC @ B^T) für Standard-MatMul                                         |
| `COMMAND_MATMUL_BACKWARD_DB`                | `execute_matmul_backward_on_gpu`              | Berechnet den Gradienten bzgl. B (dB = A^T @ dC, summiert über B) für Standard-MatMul                        |
| `COMMAND_LAYER_NORM_BACKWARD`               | `execute_layernorm_backward_on_gpu`           | Berechnet den Gradienten für Layer Normalization (dL/dx)                                                     |
| `COMMAND_ADAM_UPDATE`                       | `execute_adam_update_on_gpu`                  | Führt einen Adam-Optimierungsschritt durch (aktualisiert Parameter, m und v)                                 |
| `COMMAND_SOFTMAX_BACKWARD`                  | `execute_softmax_backward_on_gpu`             | Berechnet den Gradienten für die Softmax-Operation (dL/dx)                                                   |
| `COMMAND_MUL_BACKWARD`                      | `execute_mul_backward_on_gpu`                 | Berechnet Gradienten für elementweise Multiplikation (dL/dA, dL/dB)                                         |
| `COMMAND_TRANSPOSE_BACKWARD`                | `execute_transpose_backward_on_gpu`           | Berechnet den Gradienten für die 2D-Transposition (dA = dC^T)                                               |
| `COMMAND_EMBEDDING_LOOKUP`                  | `execute_embedding_lookup_gpu`                | Führt Embedding-Lookups basierend auf Indizes durch                                                          |
| `COMMAND_EMBEDDING_BACKWARD`                | `execute_embedding_backward_gpu`              | Berechnet den Gradienten für Embedding Weights (dL/dW, erfordert Atomics)                                    |
| `COMMAND_REDUCE_SUM_AXIS01`                 | `execute_reduce_sum_gpu`                      | Reduziert einen 3D-Tensor (B, M, N) zu 1D (N) durch Summation über Achsen 0 & 1 (nützlich für Bias-Gradienten) |
| `COMMAND_BROADCAST_ADD_BIAS`                | `execute_broadcast_add_gpu`                   | Addiert einen Bias-Vektor (N) zu einem 3D-Tensor (B, M, N) durch Broadcasting                               |
| `COMMAND_TRANSPOSE_BATCHED`                 | `execute_transpose_batched_gpu`               | Transponiert die letzten beiden Dimensionen eines Tensors (..., d1, d2) -> (..., d2, d1)                       |
| `COMMAND_MATRIX_MULTIPLY_BATCHED`           | `execute_matmul_batched_on_gpu`               | Batched Matrixmultiplikation (C[b,:,:] = A[b,:,:] @ B[b,:,:])                                               |
| `COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DA` | `execute_matmul_batched_backward_on_gpu`      | Berechnet den Gradienten bzgl. A für Batched MatMul (dA[b,:,:] = dC[b,:,:] @ B[b,:,:]^T)                       |
| `COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DB` | `execute_matmul_batched_backward_on_gpu`      | Berechnet den Gradienten bzgl. B für Batched MatMul (dB[b,:,:] = A[b,:,:]^T @ dC[b,:,:])                       |
| `COMMAND_TRANSPOSE_12_BATCHED`              | `execute_transpose_12_batched_gpu`            | Transponiert Dimensionen 1 und 2 eines 4D-Tensors (B, D1, D2, D3) -> (B, D2, D1, D3)                       |
| `COMMAND_LOG_SOFTMAX_STABLE`                | `execute_log_softmax_stable_gpu`              | Berechnet Logarithmus von Softmax (numerisch stabil)                                                        |
| `COMMAND_CROSS_ENTROPY_LOSS_GRAD`           | `execute_cross_entropy_loss_grad_gpu`         | Berechnet Cross-Entropy-Loss (NLL von LogSoftmax) und den Gradienten bzgl. der Logits                       |
| `COMMAND_ADD_BROADCAST_PE`                  | `execute_add_broadcast_pe_gpu`                | Addiert Positional Encoding (S, E) zu einem Input-Tensor (B, S, E) durch Broadcasting                     |

## Fehlerbehandlung

*   Die meisten `execute_*`-Funktionen geben `1` bei Erfolg und `0` bei einem Fehler während der Kernel-Einreichung oder -Ausführung zurück.
*   `initialize_gpu` gibt `1` bei Erfolg, `0` bei Fehler zurück.
*   `allocate_gpu_memory` gibt ein gültiges Handle (ungleich `NULL`) bei Erfolg und `NULL` bei Fehler zurück.
*   Detaillierte Fehlermeldungen von OpenCL-Aufrufen oder interne Fehlerprüfungen werden nach `stderr` ausgegeben. Überprüfen Sie diese Ausgaben bei Problemen.
*   Es liegt in der Verantwortung des Aufrufers, die Rückgabewerte zu prüfen und auf Fehler zu reagieren (z. B. durch Freigabe bereits allokierter Ressourcen).

## Mitwirkung

Beiträge sind willkommen! Bitte melden Sie Probleme oder reichen Sie Pull Requests im entsprechenden Repository ein (falls vorhanden). Halten Sie sich an gängige Coding Conventions und dokumentieren Sie Ihren Code.

---

## Lizenz

Dieses Projekt steht unter der Creative Commons Namensnennung - Nicht-kommerziell 4.0 International (CC BY-NC 4.0) Lizenz. Siehe die `LICENSE`-Datei (sofern vorhanden) für Details.

---

## FAQ (Frequently Asked Questions)

**F1: Welche GPUs werden unterstützt?**
A: Jede GPU oder jeder Beschleuniger, der einen konformen OpenCL 1.2 (oder höher) Treiber bereitstellt. Dies umfasst GPUs von AMD, NVIDIA, Intel (integrierte und dedizierte) sowie einige FPGAs und DSPs. Die Leistung und die Unterstützung für optionale Features wie FP64 oder Atomics können jedoch variieren.

**F2: Warum OpenCL und nicht CUDA?**
A: OpenCL wurde gewählt, um eine breitere Hardwarekompatibilität über verschiedene Anbieter und Plattformen hinweg zu gewährleisten. CUDA ist auf NVIDIA-Hardware beschränkt.

**F3: Wie wähle ich den richtigen `gpu_index`?**
A: `0` ist normalerweise das erste verfügbare OpenCL-Gerät (oft die primäre GPU). Wenn Sie mehrere Geräte haben, listet die Initialisierungsfunktion (`initialize_gpu`) die verfügbaren Geräte und das ausgewählte Gerät in der Konsolenausgabe (stdout) auf. Sie können den Index anpassen, um ein bestimmtes Gerät auszuwählen.

**F4: Was ist `KERNEL_FP_TYPE`? Kann ich die Präzision ändern?**
A: `KERNEL_FP_TYPE` ist ein C-Makro, das den primären Fließkommatyp für die Kernel-Berechnungen definiert (aktuell `float`, also FP32). Um dies zu ändern (z. B. auf `double` für FP64), müssten Sie das Makro im C-Code ändern und die Bibliothek neu kompilieren. Beachten Sie, dass FP64-Unterstützung nicht auf allen Geräten verfügbar ist (die Bibliothek prüft dies) und oft langsamer ist. Einige Kernel (wie Adam) verwenden intern möglicherweise weiterhin `float` für Zustandsvariablen.

**F5: Was passiert, wenn ich keine GPU habe oder die OpenCL-Treiber nicht installiert sind?**
A: Die `initialize_gpu`-Funktion wird fehlschlagen und `0` zurückgeben, wobei eine Fehlermeldung nach `stderr` ausgegeben wird. Die `simulated_*`-Funktionen könnten jedoch verwendet werden, wenn die Bibliothek entsprechend kompiliert wird, um eine grundlegende CPU-basierte Ausführung (hauptsächlich für Tests/Debugging) zu ermöglichen, aber ohne jegliche GPU-Beschleunigung.

**F6: Ich erhalte OpenCL-Fehlermeldungen in `stderr`. Was bedeuten sie?**
A: Die Bibliothek versucht, OpenCL-Fehlercodes in lesbare Strings zu übersetzen. Häufige Fehler sind:
*   `CL_DEVICE_NOT_FOUND` / `CL_DEVICE_NOT_AVAILABLE`: OpenCL-Treiber nicht installiert oder Gerät nicht erkannt.
*   `CL_OUT_OF_RESOURCES` / `CL_OUT_OF_HOST_MEMORY`: Nicht genügend GPU- oder Host-Speicher.
*   `CL_INVALID_WORK_GROUP_SIZE`: Die für einen Kernel gewählte lokale Arbeitsgruppengröße (LWS) ist für das Gerät ungültig.
*   `CL_BUILD_PROGRAM_FAILURE`: Der OpenCL-Kernel-Code konnte nicht kompiliert werden. Der Build-Log in `stderr` enthält Details.
*   `CL_INVALID_MEM_OBJECT`: Oft beim Versuch, einen bereits freigegebenen Buffer zu verwenden oder freizugeben.

**F7: `execute_embedding_backward_gpu` schlägt fehl oder liefert falsche Ergebnisse. Warum?**
A: Diese Operation erfordert atomare Operationen auf dem globalen Speicher der GPU, um die Gradienten korrekt zu akkumulieren, wenn mehrere Threads auf denselben Index im Gewichtsgradienten-Tensor schreiben. Die Bibliothek prüft bei der Initialisierung, ob ausreichende Atomics-Unterstützung (`global int cmpxchg`) vorhanden ist. Wenn nicht, wird eine Warnung ausgegeben, und die Ausführung dieser Funktion schlägt entweder fehl oder (schlimmer) führt zu Race Conditions und falschen Gradienten. Stellen Sie sicher, dass Ihr Gerät und Treiber dies unterstützen.

**F8: Wie kann ich die Leistung verbessern?**
A: Die Leistung hängt stark von der Hardware, den Treiberversionen und den spezifischen Kernel-Parametern ab. Einige Ansatzpunkte (fortgeschritten):
*   **Kernel-Tuning:** Optimieren Sie die OpenCL-Kernel selbst (Speicherzugriffsmuster, Nutzung von lokalem Speicher, Vectorization).
*   **Arbeitsgruppengrößen (LWS):** Experimentieren Sie mit verschiedenen lokalen Arbeitsgruppengrößen (`lws`) beim Aufruf von `clEnqueueNDRangeKernel`. Dies erfordert Änderungen in der C-Bibliothek.
*   **Asynchrone Ausführung:** Modifizieren Sie die Bibliothek, um nicht-blockierende Aufrufe (`clEnqueue*` mit `blocking=CL_FALSE`) und Ereignisse (`cl_event`) zu verwenden, um Berechnungen und Datenübertragungen zu überlappen.
*   **FP16/Mixed Precision:** Falls von der Hardware unterstützt, könnte die Verwendung von 16-Bit-Floats die Leistung steigern und den Speicherbedarf reduzieren (erfordert erhebliche Kernel-Anpassungen).

---

## Glossar

*   **OpenCL (Open Computing Language):** Ein offener Standard zur Programmierung heterogener Plattformen (CPUs, GPUs, DSPs, FPGAs).
*   **Kernel:** Eine Funktion, die auf einem OpenCL-Gerät (z. B. GPU) ausgeführt wird.
*   **Context:** Eine Umgebung, innerhalb derer OpenCL-Objekte (wie Kernel, Buffer, Command Queues) erstellt und verwaltet werden.
*   **Command Queue:** Eine Warteschlange, in die Befehle (Kernel-Ausführungen, Speicherübertragungen) für ein bestimmtes Gerät eingereiht werden. Befehle können synchron (blockierend) oder asynchron ausgeführt werden.
*   **Platform:** Repräsentiert eine spezifische OpenCL-Implementierung eines Anbieters (z. B. AMD APP SDK, NVIDIA CUDA OpenCL).
*   **Device:** Ein spezifisches Rechengerät innerhalb einer Plattform (z. B. eine bestimmte GPU).
*   **Work-Item:** Eine einzelne Instanz eines Kernels, die auf einem Rechenelement der GPU ausgeführt wird. Analog zu einem Thread.
*   **Work-Group:** Eine Sammlung von Work-Items, die zusammenarbeiten und auf gemeinsamen lokalen Speicher zugreifen können.
*   **GWS (Global Work Size):** Die Gesamtzahl der Work-Items, die für eine Kernel-Ausführung gestartet werden sollen, oft über mehrere Dimensionen definiert.
*   **LWS (Local Work Size):** Die Anzahl der Work-Items in einer einzelnen Work-Group, ebenfalls oft über mehrere Dimensionen definiert. GWS muss ein Vielfaches von LWS sein.
*   **Buffer Handle:** Ein opaker Pointer (`cl_mem` intern), der einen Speicherbereich auf dem OpenCL-Gerät repräsentiert.
*   **DLLEXPORT:** Ein Makro (typischerweise `__declspec(dllexport)` unter Windows oder `__attribute__((visibility("default")))` unter Linux/GCC/Clang), das eine Funktion für den Export aus einer Shared Library markiert.
*   **Host:** Bezieht sich auf die CPU und den Hauptspeicher (RAM) des Systems, das die OpenCL-Anwendung steuert.
*   **GPU (Graphics Processing Unit):** Das primäre Zielgerät für die Beschleunigung in dieser Bibliothek.
*   **MatMul:** Matrixmultiplikation.
*   **Softmax:** Eine Funktion, die einen Vektor von reellen Zahlen in einen Wahrscheinlichkeitsvektor umwandelt.
*   **LogSoftmax:** Berechnet den Logarithmus der Softmax-Ausgabe, oft numerisch stabiler und nützlich für Verlustfunktionen wie NLLLoss.
*   **GELU (Gaussian Error Linear Unit):** Eine gängige Aktivierungsfunktion in Transformer-Modellen.
*   **LayerNorm (Layer Normalization):** Eine Normalisierungstechnik, die über die Features (letzte Dimension) eines Layers normalisiert.
*   **Adam:** Ein adaptiver Optimierungsalgorithmus, der häufig zum Trainieren von neuronalen Netzen verwendet wird.
*   **Embedding:** Eine Technik, bei der diskrete Variablen (wie Wörter oder Tokens) in kontinuierliche Vektoren niedrigerer Dimension abgebildet werden.
*   **Cross-Entropy Loss:** Eine häufig verwendete Verlustfunktion für Klassifikationsaufgaben, oft in Verbindung mit Softmax oder LogSoftmax.
*   **Atomics:** Operationen (wie Addieren, Vergleichen-und-Tauschen), die auf gemeinsam genutztem Speicher garantiert unteilbar (atomar) ausgeführt werden, um Race Conditions in parallelen Umgebungen zu vermeiden.
*   **FP32 (Single Precision):** 32-Bit-Fließkommazahlen (`float`).
*   **FP64 (Double Precision):** 64-Bit-Fließkommazahlen (`double`).
*   **PE (Positional Encoding):** Vektoren, die zu Eingabe-Embeddings hinzugefügt werden, um Informationen über die Position von Elementen in einer Sequenz zu kodieren (häufig in Transformern).
*   **SDK (Software Development Kit):** Eine Sammlung von Werkzeugen, Bibliotheken und Dokumentationen zur Softwareentwicklung für eine bestimmte Plattform oder Technologie (z. B. OpenCL SDK).
*   **ICD (Installable Client Driver):** Ein Mechanismus, der es mehreren OpenCL-Implementierungen (von verschiedenen Anbietern) ermöglicht, auf demselben System zu koexistieren. Der ICD Loader leitet API-Aufrufe an die richtige Implementierung weiter.
```
