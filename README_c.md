# OpenCL-Beschleunigungsbibliothek für Neuronale Netzwerkoperationen

Diese Bibliothek stellt eine C-Schnittstelle mit OpenCL-Implementierungen für gängige Operationen bereit, die in neuronalen Netzen verwendet werden. Sie ist darauf ausgelegt, diese rechenintensiven Aufgaben auf kompatiblen GPUs zu beschleunigen. Die Bibliothek enthält sowohl Forward- als auch Backward-Pass-Operationen (für das Training) und kann als dynamische Bibliothek (.so, .dylib, .dll) kompiliert werden, um sie aus anderen Sprachen wie Python (z.B. über `ctypes` oder `cffi`) zu verwenden.

## Features

*   **GPU-Beschleunigung:** Nutzt OpenCL zur Ausführung von Berechnungen auf GPUs verschiedener Hersteller (AMD, NVIDIA, Intel, etc.).
*   **Umfangreiche Operationen:**
    *   Matrixmultiplikation (Standard und gebatched)
    *   Elementweise Operationen (Addition, Multiplikation, GELU-Aktivierung)
    *   Softmax (Zeilenweise)
    *   Layer Normalization
    *   Transponierungsoperationen:
        *   Standard 2D-Transponierung
        *   Gebatchte Transponierung der letzten beiden Dimensionen (`(..., D1, D2) -> (..., D2, D1)`)
        *   Gebatchte Transponierung der Dimensionen 1 und 2 eines 4D-Tensors (`(B, D1, D2, D3) -> (B, D2, D1, D3)`)
    *   Embedding Lookup
    *   Adam Optimizer Update-Schritt
    *   Summenreduktion über Achsen
    *   Broadcast-Addition (z.B. für Bias-Terme)
*   **Training-Unterstützung:** Enthält Backward-Pass-Kernel für die meisten Operationen, was das Training von Modellen ermöglicht.
*   **Konfigurierbare Präzision:** Unterstützt standardmäßig `float` (`FP_TYPE`), kann aber potenziell für andere Typen angepasst werden (erfordert Code-Änderungen). FP64-Unterstützung wird zur Laufzeit erkannt und für bestimmte Kernel (falls implementiert und hardwareseitig verfügbar) genutzt.
*   **Atomare Operationen:** Nutzt atomare Operationen für korrekte Gradientenakkumulation im Embedding-Backward-Pass (Hardware-Unterstützung wird geprüft).
*   **Cross-Platform:** Entwickelt für Linux, macOS und Windows (mit geringfügigen Unterschieden bei Hilfsfunktionen).
*   **Simulationsmodus:** Bietet grundlegende Dummy-Funktionen (`simulated_*`) für Tests in Umgebungen ohne GPU oder OpenCL-Installation.

## Anforderungen

*   **C-Compiler:** GCC, Clang oder MSVC (mit C99-Unterstützung oder höher).
*   **OpenCL SDK:** Ein installiertes OpenCL SDK und ein ICD Loader (Installable Client Driver) für die Zielplattform.
*   **GPU mit OpenCL-Treiber:** Eine GPU (oder eine andere OpenCL-fähige Hardware wie eine CPU), für die aktuelle OpenCL 1.2 (oder höher) Treiber installiert sind.

## Kompilierung (Building)

Die Bibliothek muss als dynamische/geteilte Bibliothek kompiliert werden (`.so` auf Linux, `.dylib` auf macOS, `.dll` auf Windows).

**Beispiel (Linux/macOS mit GCC/Clang):**

```bash
# Erstelle ein Objektfile (empfohlen)
gcc -c -fPIC -Wall -Wextra -O2 \
    -DKERNEL_FP_TYPE=float \
    -DKERNEL_FP_TYPE_STR="float" \
    $( [ "$(uname)" == "Darwin" ] && echo "-framework OpenCL" || echo "-I/pfad/zum/opencl/include -lOpenCL" ) \
    driver.c -o driver.o

# Erstelle die Shared Library
gcc -shared driver.o -o libgpudriver.so \
    $( [ "$(uname)" == "Darwin" ] && echo "-framework OpenCL" || echo "-lOpenCL" ) -lm
```

*   Ersetzen Sie `driver.c` durch den tatsächlichen Dateinamen Ihres C-Codes.
*   Passen Sie `-I/pfad/zum/opencl/include` ggf. an den Installationsort Ihres OpenCL-Headers an (oft nicht nötig auf Linux).
*   `-fPIC` ist für Shared Libraries notwendig.
*   `-lm` linkt die Mathe-Bibliothek (für `pow`, `sqrt` etc. im Host-Code).
*   `-DKERNEL_FP_TYPE` und `-DKERNEL_FP_TYPE_STR` können angepasst werden, falls eine andere Präzision (z.B. `double`) verwendet werden soll (erfordert aber auch Anpassungen in den Kernels und potenziell in den Host-Code-Typen).

**Beispiel (Windows mit MSVC):**

Öffnen Sie eine Developer Command Prompt für VS.

```bat
cl /c /O2 /W3 /D_CRT_SECURE_NO_WARNINGS /DFP_TYPE=float /DFP_TYPE_STR="float" /I"C:\Pfad\Zum\OpenCL\include" driver.c /Fodriver.obj
link /DLL driver.obj "C:\Pfad\Zum\OpenCL\lib\x64\OpenCL.lib" /OUT:gpudriver.dll
```

*   Passen Sie die Pfade zum OpenCL SDK an.
*   `/LD` ist veraltet, `link /DLL` ist der moderne Weg.
*   `_CRT_SECURE_NO_WARNINGS` wird bereits im Code definiert, schadet aber hier nicht.

## Verwendung (API-Überblick)

Die Bibliothek stellt eine C-API bereit, die über die `DLLEXPORT`-Makros exportiert wird. Der typische Workflow sieht wie folgt aus:

1.  **Initialisierung:** Rufen Sie `initialize_gpu(gpu_index)` auf, um eine bestimmte GPU zu initialisieren, die OpenCL-Umgebung einzurichten und alle Kernel zu kompilieren. `gpu_index` ist der 0-basierte Index des zu verwendenden Geräts.
2.  **Speicherallokation:** Verwenden Sie `allocate_gpu_memory(gpu_index, size)` um Speicher auf dem initialisierten GPU-Gerät zu reservieren. Gibt einen Handle (`void*`) zurück.
3.  **Datentransfer (Host -> GPU):** Kopieren Sie Daten vom Host-Speicher zum GPU-Speicher mit `write_host_to_gpu_blocking(gpu_index, gpu_buffer_handle, offset, size, host_source_ptr)`.
4.  **Kernel-Ausführung:** Rufen Sie eine der `execute_*_on_gpu`-Funktionen auf (z.B. `execute_matmul_on_gpu`), um eine Berechnung auf der GPU durchzuführen. Übergeben Sie die GPU-Speicherhandles und relevante Dimensionen/Parameter. Diese Funktionen sind blockierend und kehren erst zurück, wenn die Operation abgeschlossen ist.
5.  **Datentransfer (GPU -> Host):** Lesen Sie Ergebnisse vom GPU-Speicher zurück in den Host-Speicher mit `read_gpu_to_host_blocking(gpu_index, gpu_buffer_handle, offset, size, host_destination_ptr)`.
6.  **Speicherfreigabe:** Geben Sie GPU-Speicher frei, der nicht mehr benötigt wird, mit `free_gpu_memory(gpu_index, gpu_buffer_handle)`.
7.  **Aufräumen:** Wenn die Bibliothek nicht mehr benötigt wird, rufen Sie `shutdown_driver()` auf, um alle OpenCL-Ressourcen (Kontext, Warteschlange, Kernel, Programme) ordnungsgemäß freizugeben.

### Haupt-API-Funktionen

*   **Initialisierung/Aufräumen:**
    *   `int initialize_gpu(int gpu_index)`
    *   `void shutdown_driver()`
*   **Speicherverwaltung:**
    *   `void *allocate_gpu_memory(int gpu_index, size_t size)`
    *   `void free_gpu_memory(int gpu_index, void* buffer_handle)`
*   **Datentransfer (Blockierend):**
    *   `int write_host_to_gpu_blocking(int gpu_index, void* gpu_buffer_handle, size_t offset, size_t size, const void* host_source_ptr)`
    *   `int read_gpu_to_host_blocking(int gpu_index, void* gpu_buffer_handle, size_t offset, size_t size, void* host_destination_ptr)`
*   **Kernel-Ausführung (Auswahl):**
    *   `int execute_matmul_on_gpu(...)`
    *   `int execute_softmax_on_gpu(...)`
    *   `int execute_gelu_on_gpu(...)`
    *   `int execute_add_on_gpu(...)`
    *   `int execute_mul_on_gpu(...)`
    *   `int execute_layernorm_on_gpu(...)`
    *   `int execute_clone_on_gpu(...)`
    *   `int execute_transpose_on_gpu(...)` // 2D
    *   `int execute_gelu_backward_on_gpu(...)`
    *   `int execute_matmul_backward_on_gpu(...)`
    *   `int execute_layernorm_backward_on_gpu(...)`
    *   `int execute_adam_update_on_gpu(...)`
    *   `int execute_softmax_backward_on_gpu(...)`
    *   `int execute_mul_backward_on_gpu(...)`
    *   `int execute_transpose_backward_on_gpu(...)` // 2D
    *   `int execute_embedding_lookup_gpu(...)`
    *   `int execute_embedding_backward_gpu(...)`
    *   `int execute_reduce_sum_gpu(...)`
    *   `int execute_broadcast_add_gpu(...)`
    *   `int execute_transpose_batched_gpu(...)` // Letzte zwei Dims
    *   `int execute_transpose_12_batched_gpu(...)` // Dims 1 & 2 (4D Tensor)
    *   `int execute_matmul_batched_on_gpu(...)`
    *   `int execute_matmul_batched_backward_on_gpu(...)`
*   **Simulations-API:**
    *   `unsigned long long simulated_kernel_allocate(int gpu_index, size_t size)`
    *   `void simulated_kernel_free(int gpu_index, unsigned long long address, size_t size)`
    *   `void simulated_kernel_write(int gpu_index, unsigned long long address, size_t size, const void *source)`
    *   `unsigned int simulated_get_compute_unit_count(int gpu_index)`

Eine vollständige Liste der Funktionen und ihrer Parameter finden Sie in den `DLLEXPORT`-Deklarationen im Quellcode.

## Unterstützte GPU-Operationen (Kernel)

Die folgende Tabelle listet die implementierten GPU-Kernel und die zugehörigen Wrapper-Funktionen auf:

| Operation                                       | Wrapper-Funktion                           | GPUCommand Enum                        | Beschreibung                                                                       |
| :---------------------------------------------- | :----------------------------------------- | :------------------------------------- | :--------------------------------------------------------------------------------- |
| Matrix Multiply (Standard)                      | `execute_matmul_on_gpu`                    | `COMMAND_MATRIX_MULTIPLY`              | C = A @ B (A: (B, M, K), B: (K, N), C: (B, M, N)) - B wird geteilt.               |
| Softmax (Row-wise)                              | `execute_softmax_on_gpu`                   | `COMMAND_SOFTMAX_ROWWISE`              | Wendet Softmax auf jede Zeile einer Matrix an.                                     |
| GELU Activation (Element-wise)                  | `execute_gelu_on_gpu`                      | `COMMAND_GELU_ELEMENTWISE`             | Wendet die Gaussian Error Linear Unit an.                                        |
| Addition (Element-wise)                         | `execute_add_on_gpu`                       | `COMMAND_ADD_ELEMENTWISE`              | C = A + B                                                                          |
| Multiplication (Element-wise)                   | `execute_mul_on_gpu`                       | `COMMAND_MUL_ELEMENTWISE`              | C = A * B                                                                          |
| Layer Normalization                             | `execute_layernorm_on_gpu`                 | `COMMAND_LAYER_NORM`                   | Normalisiert Features über die letzte Dimension.                                   |
| Clone / Copy Buffer                             | `execute_clone_on_gpu`                     | `COMMAND_CLONE`                        | Kopiert den Inhalt eines GPU-Buffers in einen anderen.                           |
| Transpose (2D)                                  | `execute_transpose_on_gpu`                 | `COMMAND_TRANSPOSE`                    | Transponiert eine 2D-Matrix.                                                       |
| GELU Backward (Element-wise)                    | `execute_gelu_backward_on_gpu`             | `COMMAND_GELU_BACKWARD_ELEMENTWISE`    | Berechnet den Gradienten für GELU.                                                 |
| Matrix Multiply Backward (Standard)             | `execute_matmul_backward_on_gpu`           | `DA: 10, DB: 11`                       | Berechnet Gradienten dA und dB für Standard-Matmul.                                |
| Layer Normalization Backward                    | `execute_layernorm_backward_on_gpu`        | `COMMAND_LAYER_NORM_BACKWARD`          | Berechnet den Gradienten für Layer Normalization.                                  |
| Adam Optimizer Update                           | `execute_adam_update_on_gpu`               | `COMMAND_ADAM_UPDATE`                  | Führt einen Optimierungsschritt mit Adam durch.                                    |
| Softmax Backward                                | `execute_softmax_backward_on_gpu`          | `COMMAND_SOFTMAX_BACKWARD`             | Berechnet den Gradienten für Softmax.                                              |
| Multiplication Backward (Element-wise)          | `execute_mul_backward_on_gpu`              | `COMMAND_MUL_BACKWARD`                 | Berechnet Gradienten dA und dB für elementweise Multiplikation.                  |
| Transpose Backward (2D)                         | `execute_transpose_backward_on_gpu`        | `COMMAND_TRANSPOSE_BACKWARD`           | Berechnet den Gradienten für 2D-Transponierung (ist selbst eine Transponierung). |
| Embedding Lookup                                | `execute_embedding_lookup_gpu`             | `COMMAND_EMBEDDING_LOOKUP`             | Sammelt Embedding-Vektoren basierend auf Indizes.                                  |
| Embedding Backward                              | `execute_embedding_backward_gpu`           | `COMMAND_EMBEDDING_BACKWARD`           | Berechnet Gradienten für Embedding Weights (nutzt atomare Operationen).            |
| Reduce Sum (Axis 0, 1)                          | `execute_reduce_sum_gpu`                   | `COMMAND_REDUCE_SUM_AXIS01`            | Reduziert einen Tensor (B, M, N) zu (N) durch Summation über Achsen 0 und 1.     |
| Broadcast Add Bias                              | `execute_broadcast_add_gpu`                | `COMMAND_BROADCAST_ADD_BIAS`           | Addiert einen Bias-Vektor (N) zu einem Tensor (B, M, N).                           |
| Transpose (Batched, Last Two Dims)              | `execute_transpose_batched_gpu`            | `COMMAND_TRANSPOSE_BATCHED`            | Transponiert die letzten beiden Dimensionen eines Tensors `(..., D1, D2) -> (..., D2, D1)`. |
| Transpose (Batched, Dims 1 & 2)                 | `execute_transpose_12_batched_gpu`         | `COMMAND_TRANSPOSE_12_BATCHED`         | Transponiert Dim 1 und 2 eines 4D Tensors `(B, D1, D2, D3) -> (B, D2, D1, D3)`.    |
| Matrix Multiply (Batched)                       | `execute_matmul_batched_on_gpu`            | `COMMAND_MATRIX_MULTIPLY_BATCHED`      | C = A @ B (A: (B, M, K), B: (B, K, N), C: (B, M, N)).                              |
| Matrix Multiply Backward (Batched)              | `execute_matmul_batched_backward_on_gpu`   | `DA: 23, DB: 24`                       | Berechnet Gradienten dA und dB für gebatchtes Matmul.                              |

## Architektur und Design

*   **OpenCL Kontext:** Die Bibliothek erstellt einen einzigen OpenCL-Kontext und eine Befehlswarteschlange (`cl_command_queue`) pro initialisiertem GPU-Gerät.
*   **Kernel-Kompilierung:** Alle OpenCL-Kernel werden während des Aufrufs von `initialize_gpu` aus den eingebetteten Quellcode-Strings kompiliert. Dies kann einige Sekunden dauern, stellt aber sicher, dass die Kernel für das spezifische Gerät optimiert sind. Build-Optionen wie `-cl-std=CL1.2` und Flags für FP64/Atomics werden automatisch gesetzt.
*   **Befehlseinreichung:** Jede `execute_*`-Funktion erstellt eine Datenstruktur mit den notwendigen Parametern (Buffer-Handles, Dimensionen) und ruft die interne Funktion `submit_kernel_command` auf. Diese Funktion wählt den richtigen Kernel aus, setzt die Argumente (`clSetKernelArg`) und stellt den Kernel zur Ausführung in die Warteschlange (`clEnqueueNDRangeKernel`).
*   **Synchronisation:** Die öffentlichen `execute_*`-Funktionen sind blockierend, d.h. sie warten mittels `clFinish` auf den Abschluss der Operation, bevor sie zurückkehren.

## Gleitkommapräzision (FP_TYPE)

*   Die Standard-Gleitkommapräzision für die Kernel ist `float`, definiert durch `KERNEL_FP_TYPE` und `KERNEL_FP_TYPE_STR`.
*   Einige Kernel (z.B. LayerNorm, Softmax Backward, ReduceSum) verwenden intern einen Akkumulationstyp (`ACCUM_TYPE`), der auf `double` gesetzt wird, wenn die Hardware FP64 unterstützt (`CL_HAS_FP64`), um die Genauigkeit bei Summierungen zu verbessern. Ansonsten wird `float` verwendet.
*   Die Unterstützung für FP64 wird zur Laufzeit in `initialize_gpu` geprüft.

## Atomare Operationen (Atomics)

*   Der Kernel `embedding_backward_scatter_add` erfordert atomare Operationen (`atomic_add` für Gleitkommazahlen), um Gradienten korrekt zu akkumulieren, wenn mehrere Threads auf denselben Index im Gewichtsgradienten-Tensor schreiben.
*   `initialize_gpu` prüft, ob das ausgewählte OpenCL-Gerät atomare Operationen im Geräte-Scope unterstützt (`has_atomics_support`).
*   Obwohl der Kernel mit der Build-Option `-D CL_HAS_ATOMICS` kompiliert wird (um die Syntax zu ermöglichen), wird der `embedding_backward`-Kernel zur Laufzeit nur dann erfolgreich eingereicht, wenn `has_atomics_support` wahr ist. Andernfalls wird ein Fehler ausgegeben.
*   **Warnung:** Wenn Atomics nicht unterstützt werden, kann die Verwendung von `execute_embedding_backward_gpu` zu falschen Gradienten führen (Race Conditions).

## Simulationsmodus

Die `simulated_*`-Funktionen (`simulated_kernel_allocate`, `simulated_kernel_free`, `simulated_kernel_write`, `simulated_get_compute_unit_count`) bieten eine minimale CPU-basierte Simulation der Speicherverwaltung und Informationsabfrage. Sie verwenden `malloc`/`free`/`memcpy` und geben einen festen Wert für Compute Units zurück. **Sie führen keine tatsächlichen Berechnungen durch.** Diese Schicht ist nützlich für grundlegende Tests der Aufruflogik oder in Umgebungen, in denen keine GPU verfügbar ist.

## Lizenz

[TODO: Fügen Sie hier Ihre Lizenzinformationen ein. Z.B. MIT, Apache 2.0, etc.]

Wenn keine Lizenz angegeben ist, ist die Software standardmäßig urheberrechtlich geschützt und darf nicht ohne Erlaubnis verwendet oder verbreitet werden. Es wird dringend empfohlen, eine Open-Source-Lizenz hinzuzufügen.

## Beitragende / Kontakt

[TODO: Fügen Sie Informationen hinzu, wie andere beitragen können oder wer der Ansprechpartner ist.]
```

**Wichtige Punkte und Anpassungen:**

1.  **Dateiname:** Ersetzen Sie `driver.c` in den Build-Beispielen durch den tatsächlichen Namen Ihrer C-Quelldatei.
2.  **Bibliotheksname:** Passen Sie `libgpudriver.so` / `gpudriver.dll` an, wenn Sie einen anderen Namen wünschen.
3.  **Pfade:** Stellen Sie sicher, dass die Pfade zum OpenCL SDK in den Build-Beispielen korrekt sind.
4.  **FP_TYPE:** Wenn Sie die Präzision ändern möchten, müssen Sie die `-D`-Flags beim Kompilieren *und* potenziell den C-Code und die Kernel anpassen. Die aktuelle Implementierung ist stark auf `float` ausgerichtet.
5.  **Lizenz:** Fügen Sie unbedingt eine Lizenz hinzu! MIT ist eine gängige Wahl für solche Bibliotheken.
6.  **Kontakt/Beitragende:** Füllen Sie diese Abschnitte aus, falls relevant.
7.  **Fehlerbehandlung:** Das README geht davon aus, dass die Fehlerprüfungen im Code (`CHECK_CL_ERR` und Rückgabewerte) robust sind.
8.  **Sprache:** Die README ist auf Deutsch verfasst, wie gewünscht. Fachbegriffe wie "Kernel", "Context", "Command Queue", "Shared Library", "Floating-Point Precision", "Atomics" wurden teilweise beibehalten oder mit deutschen Entsprechungen versehen, um die Verständlichkeit für technisch versierte Leser zu gewährleisten.

