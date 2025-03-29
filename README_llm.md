# OCL LLM Framework - OpenCL Accelerated Language Model Training

**Version:** Ultimate Build Attempt 7 - PE GPU Add

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Beispiel-Badge, ggf. anpassen -->

## Übersicht

OCL LLM Framework ist ein Python-basiertes Framework zur Beschleunigung des Trainings von Transformer-basierten Sprachmodellen mithilfe von OpenCL. Es nutzt eine benutzerdefinierte C/C++-Bibliothek (`driver.dll`), die über `ctypes` angebunden wird, um rechenintensive Operationen auf kompatiblen GPUs (AMD, Intel, NVIDIA über OpenCL-Treiber) auszuführen.

Das Framework implementiert Kernkomponenten, die für das Training moderner LLMs erforderlich sind, darunter:

*   Eine `OclTensor`-Klasse mit automatischer Differenzierung (Autograd).
*   GPU-beschleunigte Operationen (MatMul, Add, GeLU, LayerNorm, Softmax, etc.).
*   Standard-Transformer-Layer (Embedding, Positional Encoding, Multi-Head Attention, FFN).
*   Den Adam-Optimizer mit GPU-basierten Zuständen.
*   Checkpointing-Funktionalität zum Speichern und Laden des Trainingsfortschritts.
*   Dynamische Erkennung optionaler, optimierter OpenCL-Kernels.

Ziel ist es, eine Alternative zu CUDA/ROCm-basierten Frameworks für Benutzer zu bieten, die OpenCL-kompatible Hardware einsetzen oder die Flexibilität von OpenCL bevorzugen.

## Features

*   **OpenCL-Backend:** Nutzt OpenCL für GPU-Beschleunigung über eine externe C/C++ DLL.
*   **Tensor-Klasse (`OclTensor`):**
    *   GPU-Speicherverwaltung (`GPUBuffer`).
    *   PyTorch-ähnliche API für Tensor-Operationen.
    *   Integriertes **Autograd**-System für automatische Gradientenberechnung.
*   **GPU-Kernel:**
    *   Implementierung wichtiger NN-Operationen als OpenCL-Kernel (siehe `driver.dll`).
    *   Unterstützung für **Standard- und Batched-MatMul**.
    *   Spezialisierte Kernel für **Transpose**, **LayerNorm**, **Softmax**, **GeLU**, **Adam-Update**.
    *   **Optional optimierte Kernel** für Positional Encoding Addition (`HAS_ADD_BROADCAST_PE`) und Bias-Gradienten-Reduktion (`HAS_REDUCE_SUM`), die zur Laufzeit erkannt werden.
    *   **Optional GPU-beschleunigtes Embedding** (Lookup & Backward) (`HAS_EMBEDDING_LOOKUP`, `HAS_EMBEDDING_BACKWARD`).
*   **Transformer-Architektur:**
    *   Implementierung von `Linear`, `Embedding`, `PositionalEncoding`, `MultiHeadAttention`, `LayerNorm`, `GeLUActivation` und `TransformerBlock`.
    *   Ein einfaches `SimpleModel` als Beispiel-Encoder-Modell.
*   **Optimierung:**
    *   GPU-beschleunigter `AdamOptimizer`.
*   **Hilfsfunktionen:**
    *   `TinyTokenizer` für einfache Textverarbeitung.
    *   Cross-Entropy-Loss-Berechnung (CPU-basiert).
    *   Speichern und Laden von Checkpoints (`.npz`-Format) inkl. Modell, Optimizer, Tokenizer und Trainingszustand.
    *   Automatische Speicherbereinigung für GPU-Ressourcen.
*   **Flexibilität:**
    *   Wahl des GPU-Geräts über Kommandozeilenargument (`--gpu_id`).
    *   Fallback auf CPU-Implementierungen, wenn optimierte Kernel fehlen.

## Voraussetzungen

1.  **Python:** Version 3.x.
2.  **NumPy:** `pip install numpy`
3.  **OpenCL-fähige GPU:** Eine GPU von AMD, Intel oder NVIDIA mit installierten OpenCL-Treibern und Laufzeitumgebung (ICD Loader).
4.  **C/C++ Compiler & OpenCL SDK:** Notwendig, um die `driver.dll` (oder `.so` unter Linux) zu kompilieren. Dies ist *nicht* Teil des Python-Codes, sondern eine externe Abhängigkeit.
    *   Ein C++-Compiler (wie GCC, Clang, MSVC).
    *   Das OpenCL SDK (Header-Dateien wie `CL/cl.h` und die OpenCL-Linker-Bibliothek).
5.  **Kompilierte `driver.dll`:** Die kompilierte C++/OpenCL-Bibliothek muss entweder im selben Verzeichnis wie das Python-Skript (`ocl_framework.py`) oder im aktuellen Arbeitsverzeichnis liegen.

## Installation & Setup

1.  **Repository klonen:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Python-Abhängigkeiten installieren:**
    ```bash
    pip install numpy
    ```
3.  **`driver.dll` kompilieren:**
    *   Dieser Schritt erfordert eine separate C++/OpenCL-Entwicklungsumgebung.
    *   Kompilieren Sie den C++/OpenCL-Quellcode (der nicht in diesem Python-Code enthalten ist) zu einer dynamischen Bibliothek (`driver.dll` unter Windows, `driver.so` unter Linux).
    *   Stellen Sie sicher, dass die Kompilierung gegen die OpenCL-Header erfolgt und mit der OpenCL-Bibliothek gelinkt wird.
    *   **WICHTIG:** Platzieren Sie die resultierende `driver.dll` (oder `.so`) im selben Verzeichnis wie `ocl_framework.py` oder in dem Verzeichnis, von dem aus Sie das Skript starten.
4.  **OpenCL-Treiber überprüfen:** Stellen Sie sicher, dass die OpenCL-Treiber für Ihre GPU korrekt installiert und funktionsfähig sind. Tools wie `clinfo` können dabei helfen.

## Verwendung

### Training

Das Training wird über das Hauptskript `ocl_framework.py` (oder wie auch immer Ihre Hauptdatei heißt) gestartet.

**Grundlegender Start:**

```bash
python ocl_framework.py
```

Dies startet ein neues Training mit der Standardkonfiguration und verwendet GPU-Gerät 0. Es wird eine `input.txt`-Datei im selben Verzeichnis erwartet, die die Trainingsdaten enthält.

**Optionale Argumente:**

*   `--gpu_id <index>`: Wählt das zu verwendende OpenCL-GPU-Gerät aus (Standard: 0).
    ```bash
    python ocl_framework.py --gpu_id 1
    ```
*   `--load_checkpoint <pfad/zur/datei.npz>`: Lädt einen Checkpoint und setzt das Training fort.
    ```bash
    python ocl_framework.py --load_checkpoint checkpoints/best_model.npz
    ```
*   `--save_dir <verzeichnis>`: Gibt das Verzeichnis an, in dem Checkpoints gespeichert werden (Standard: `checkpoints`). Der beste Checkpoint wird als `best_model.npz` in diesem Verzeichnis gespeichert.
    ```bash
    python ocl_framework.py --save_dir training_run_1
    ```

**Trainingsdaten:**

*   Erstellen Sie eine Textdatei namens `input.txt` im selben Verzeichnis wie das Skript. Diese Datei enthält den Rohtext für das Training.

### Inference

Inferenz wird automatisch periodisch während des Trainings und am Ende des Trainingslaufs durchgeführt, um die Modellleistung zu demonstrieren. Es wird sowohl das zuletzt trainierte Modell als auch das Modell aus dem besten gespeicherten Checkpoint (falls vorhanden) verwendet.

## Architektur

Das Framework ist modular aufgebaut:

1.  **CTypes Interface:** Bindet die Funktionen aus der `driver.dll` an Python. Definiert Argument- und Rückgabetypen für die C-Funktionen.
2.  **`GPUBuffer`:** Eine Wrapper-Klasse für OpenCL-Speicherpuffer. Kümmert sich um Allokation, Freigabe sowie Datenübertragungen zwischen Host (CPU) und Device (GPU).
3.  **`OclTensor`:** Die zentrale Tensor-Klasse. Sie enthält einen Verweis auf einen `GPUBuffer` für die Daten, speichert Metadaten wie Shape und implementiert die mathematischen Operationen durch Aufruf der entsprechenden CTypes-Funktionen. Sie enthält auch die Logik für das Autograd-System (`_ctx`, `requires_grad`, `backward`, etc.).
4.  **Backward Contexts:** Klassen wie `MatMulBackwardContext`, `AddBackwardContext`, etc. speichern die für den Backward-Pass benötigten Informationen (Referenzen auf Eingabe-Tensoren, Shapes, etc.) und definieren die `backward`-Methode, um die Gradienten zu berechnen und weiterzuleiten.
5.  **Layers:** Standard-NN-Layer (`Linear`, `Embedding`, `LayerNorm`, `MultiHeadAttention`, etc.), die `OclTensor`-Operationen verwenden und `Parameter` (trainierbare `OclTensor`) verwalten.
6.  **`SimpleModel`:** Ein Beispielmodell, das die Layer zu einer Transformer-Architektur zusammensetzt.
7.  **`AdamOptimizer`:** Implementiert den Adam-Algorithmus, wobei die Momentum-Zustände (`m` und `v`) ebenfalls in `GPUBuffer`n gehalten und über einen OpenCL-Kernel aktualisiert werden.
8.  **Initialisierung & Cleanup:** `ocl_initialize` initialisiert die OpenCL-Umgebung über die DLL und prüft auf optionale Kernel. `ocl_shutdown` gibt alle GPU-Ressourcen (Tensoren, Optimizer-Zustände, PE-Buffer) frei und ruft die Shutdown-Funktion der DLL auf.

## Status Optionaler Kernel

Bei der Initialisierung (`ocl_initialize`) prüft das Framework, ob bestimmte optimierte Kernel in der `driver.dll` vorhanden sind. Dies beeinflusst die Performance und welche Operationen auf der GPU ausgeführt werden:

*   `HAS_BMM_BATCHED`:       ✅ Schnelles Batched Matrix Multiply für MHA. ❌ Fallback auf sequenzielle MatMul-Aufrufe (langsamer) oder CPU-Loop.
*   `HAS_TRANSPOSE_LAST_TWO`:✅ Schnelles Transponieren der letzten beiden Dimensionen (oft in MHA benötigt). ❌ Fallback auf CPU-Transpose.
*   `HAS_TRANSPOSE_12_BATCHED`:✅ Schnelles Transponieren der Dimensionen 1 und 2 in 4D-Tensoren (MHA). ❌ Fallback auf CPU-Transpose.
*   `HAS_REDUCE_SUM`:        ✅ Effiziente GPU-Reduktion für Bias-Gradienten bei Broadcasting-Addition. ❌ Fallback auf CPU-Reduktion (potenziell langsam).
*   `HAS_ADD_BROADCAST_PE`:  ✅ Effiziente GPU-Addition von Positional Encoding (broadcasted). ❌ Fallback auf CPU-Addition mit Broadcasting.
*   `HAS_EMBEDDING_LOOKUP`:  ✅ Schnelles Embedding-Lookup auf der GPU. ❌ Fallback auf CPU-Lookup.
*   `HAS_EMBEDDING_BACKWARD`:✅ Schnelle (atomare) Gradientenakkumulation für Embeddings auf der GPU. ❌ Fallback auf CPU-Gradientenberechnung.

Die Verfügbarkeit dieser Kernel hängt von der Implementierung in der `driver.dll` ab.

## Limitationen & Bekannte Probleme

*   **Cross-Entropy Loss:** Die Berechnung des Loss und des initialen Gradienten für den Backward-Pass erfolgt auf der **CPU**.
*   **CPU Fallbacks:** Wie oben beschrieben, können bestimmte Operationen auf die CPU zurückfallen, wenn die entsprechenden optimierten OpenCL-Kernel in der `driver.dll` nicht vorhanden sind. Dies kann die Leistung erheblich beeinträchtigen.
*   **Embedding Layer:** Standardmäßig wird der Embedding-Lookup auf der CPU durchgeführt, es sei denn, die optionalen GPU-Embedding-Kernel sind vorhanden und werden erkannt. Die Synchronisation zwischen GPU-Parametern und CPU-Host-Gewichten (`update_host_weight_from_gpu`) ist notwendig.
*   **Kein Dropout:** Dropout-Layer sind derzeit nicht implementiert.
*   **Error Handling:** Das Error-Handling bei OpenCL-Kernel-Fehlern ist grundlegend. Detailliertere Fehlermeldungen aus der DLL wären hilfreich.
*   **Kompilierung der DLL:** Die Notwendigkeit, die `driver.dll` separat zu kompilieren, stellt eine zusätzliche Hürde dar.

## FAQ (Häufig gestellte Fragen)

**F: Warum OpenCL statt CUDA oder ROCm?**
**A:** OpenCL ist ein offener Standard, der auf einer breiteren Palette von Hardware läuft, einschließlich GPUs von AMD, Intel und NVIDIA (über deren OpenCL-Treiber). Dies bietet potenziell mehr Flexibilität als proprietäre APIs wie CUDA.

**F: Wie kompiliere ich die `driver.dll`?**
**A:** Sie benötigen eine C/C++-Entwicklungsumgebung (z. B. g++, Clang, MSVC) und das OpenCL SDK für Ihr Betriebssystem. Der Kompilierungsprozess beinhaltet typischerweise:
    1.  Einbinden der OpenCL-Header (z. B. `-I/pfad/zum/opencl/include`).
    2.  Linken gegen die OpenCL-Bibliothek (z. B. `-lOpenCL`).
    3.  Kompilieren des C++/OpenCL-Quellcodes zu einer Shared Library/DLL (z. B. `g++ ... -shared -o driver.so ...`).
    Der genaue Befehl hängt von Ihrem Compiler und Betriebssystem ab. Stellen Sie sicher, dass die exportierten Funktionen mit `extern "C"` deklariert sind, um Namens-Mangling zu vermeiden.

**F: Ich erhalte einen `FileNotFoundError` oder `OSError` beim Laden der DLL.**
**A:** Überprüfen Sie Folgendes:
    1.  Existiert die `driver.dll` (oder `.so`)?
    2.  Liegt sie im richtigen Verzeichnis (neben dem Skript oder im Arbeitsverzeichnis)?
    3.  Sind alle Laufzeitabhängigkeiten der DLL erfüllt? Insbesondere muss die OpenCL-Laufzeitumgebung (ICD Loader und Treiber für Ihre GPU) korrekt installiert und für das System sichtbar sein.
    4.  Stimmt die Architektur (32-Bit vs. 64-Bit) der DLL mit der Ihres Python-Interpreters überein?

**F: Das Training ist sehr langsam.**
**A:** Mögliche Gründe:
    1.  **Fehlende optimierte Kernel:** Überprüfen Sie die Ausgabe von `ocl_initialize`. Wenn viele `HAS_...`-Flags auf `NO` stehen, werden CPU-Fallbacks verwendet, die langsam sind. Stellen Sie sicher, dass Ihre `driver.dll` die optimierten Kernel enthält.
    2.  **Schwache GPU:** Die Leistung hängt stark von der verwendeten GPU ab.
    3.  **Kleine Batch-Größe:** Eine zu kleine Batch-Größe nutzt die GPU möglicherweise nicht effizient aus.
    4.  **Datenübertragungs-Overhead:** Häufige CPU-Fallbacks (z.B. für Loss, Embedding) erfordern Datenübertragungen zwischen CPU und GPU, die teuer sind.

**F: Ich bekomme NaN/Inf (Not a Number / Infinity) Verluste.**
**A:** Dies kann auf numerische Instabilität hinweisen. Versuchen Sie:
    1.  **Lernrate reduzieren:** Eine zu hohe Lernrate ist eine häufige Ursache.
    2.  **Initialisierung überprüfen:** Stellen Sie sicher, dass die Gewichte sinnvoll initialisiert werden.
    3.  **Gradient Clipping:** (Nicht implementiert) Könnte helfen, explodierende Gradienten zu verhindern.
    4.  **Epsilon-Werte:** Überprüfen Sie Epsilon-Werte in LayerNorm oder Adam auf Angemessenheit.

**F: Wie füge ich neue OpenCL-Kernel hinzu?**
**A:**
    1.  Implementieren Sie den Kernel im C++/OpenCL-Quellcode der `driver.dll`.
    2.  Exportieren Sie eine C-Funktion aus der DLL, die diesen Kernel aufruft.
    3.  Fügen Sie in Python (`ocl_framework.py`) eine CTypes-Signatur für die neue C-Funktion hinzu.
    4.  Erstellen Sie eine neue Methode in `OclTensor` oder einer Layer-Klasse, die die CTypes-Funktion aufruft.
    5.  Implementieren Sie ggf. eine entsprechende Backward-Context-Klasse für Autograd.
    6.  Optional: Fügen Sie eine `HAS_...`-Flag-Prüfung in `ocl_initialize` hinzu.

## Glossar

*   **OpenCL (Open Computing Language):** Ein offener Standard für plattformübergreifende, parallele Programmierung von heterogenen Systemen (CPUs, GPUs, DSPs, FPGAs).
*   **Kernel:** Eine Funktion, die auf einem OpenCL-Gerät (typischerweise einer GPU) parallel ausgeführt wird.
*   **DLL (Dynamic Link Library) / Shared Library (.so):** Eine kompilierte Bibliothek mit Code (hier: die OpenCL-Kernel-Aufrufe), die zur Laufzeit von einem anderen Programm (hier: Python) geladen und verwendet werden kann.
*   **ctypes:** Eine Python-Standardbibliothek, die das Aufrufen von Funktionen in C-kompilierten Bibliotheken (DLLs/.so) ermöglicht.
*   **OclTensor:** Die zentrale Tensor-Datenstruktur in diesem Framework, die Daten auf der GPU hält und Operationen darauf ermöglicht.
*   **GPUBuffer:** Ein Wrapper für einen OpenCL-Speicherpuffer auf der GPU.
*   **Autograd (Automatic Differentiation):** Ein System, das automatisch Gradienten von Skalarausgaben (z. B. Loss) bezüglich der Eingaben (z. B. Modellparameter) berechnet, indem es den Berechnungsfluss rückwärts verfolgt.
*   **Backward Context:** Ein Objekt, das während des Forward-Passes erstellt wird und Informationen speichert, die für die Berechnung der Gradienten im Backward-Pass benötigt werden (z. B. Referenzen auf Eingabe-Tensoren, Zwischenwerte).
*   **Parameter:** Ein `OclTensor`, der als trainierbarer Gewichtungs- oder Bias-Parameter eines Modells markiert ist (`requires_grad=True`).
*   **Adam (Adaptive Moment Estimation):** Ein weit verbreiteter Optimierungsalgorithmus für das Training von Deep-Learning-Modellen.
*   **Positional Encoding (PE):** Eine Technik, um Informationen über die Position von Tokens in einer Sequenz in deren Vektorrepräsentationen einzubetten.
*   **Multi-Head Attention (MHA):** Ein Kernmechanismus in Transformer-Modellen, der es dem Modell ermöglicht, die Wichtigkeit verschiedener Teile der Eingabesequenz für jeden Teil der Ausgabesequenz zu gewichten.
*   **Layer Normalization:** Eine Normalisierungstechnik, die dazu beiträgt, das Training von tiefen Netzwerken zu stabilisieren, indem sie die Aktivierungen innerhalb eines Layers normalisiert.
*   **GeLU (Gaussian Error Linear Unit):** Eine Aktivierungsfunktion, die häufig in Transformer-Modellen verwendet wird.
*   **Checkpoint:** Eine Datei, die den Zustand eines Trainingsprozesses speichert (Modellgewichte, Optimizer-Zustand, Epoche, etc.), um das Training später fortsetzen zu können.
*   **Tokenizer:** Ein Werkzeug, das Rohtext in eine Sequenz von Token-IDs (Zahlen) umwandelt, die vom Modell verarbeitet werden können, und umgekehrt.
*   **Vocab Size (Vokabulargröße):** Die Anzahl der einzigartigen Tokens, die der Tokenizer kennt.
*   **Embedding Dimension (Einbettungsdimension):** Die Größe der Vektorrepräsentationen, die verwendet werden, um jedes Token darzustellen.
*   **FP_TYPE (Floating Point Type):** Der Datentyp (z. B. `numpy.float32`), der für die meisten Berechnungen und die Speicherung von Tensor-Daten verwendet wird.
*   **Batch Size (Stapelgröße):** Die Anzahl der Trainingsbeispiele, die gleichzeitig in einem Forward- und Backward-Pass verarbeitet werden.
*   **Epoch (Epoche):** Ein vollständiger Durchlauf durch den gesamten Trainingsdatensatz.

## Lizenz

[MIT License](LICENSE) <!-- Fügen Sie eine LICENSE-Datei hinzu und verlinken Sie sie -->

*Dieses Projekt wird unter der MIT-Lizenz veröffentlicht. Weitere Informationen finden Sie in der LICENSE-Datei.*

## Beitragende

<!-- Optional: Fügen Sie Informationen hinzu, wie andere beitragen können -->
Beiträge sind willkommen! Bitte erstellen Sie einen Pull Request oder ein Issue.

---
