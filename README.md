

# OCL-LLM-Framework: Ein OpenCL-beschleunigtes Framework für das Training von Large Language Models (LLMs)

[![Lizenz: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Version:** v1.0 – Produktionsreif

## Übersicht

Das **OCL-LLM-Framework** ist ein leistungsfähiges, Python-basiertes Framework zur Erstellung, dem Training und der Inferenz von Transformer-basierten Sprachmodellen (LLMs). Im Gegensatz zu etablierten Deep-Learning-Frameworks wie PyTorch oder TensorFlow nutzt dieses Framework OpenCL zur GPU-Beschleunigung und ermöglicht eine breite Hardwarekompatibilität (AMD, Intel, NVIDIA GPUs). 

Das Framework wurde speziell entwickelt, um OpenCL als universelle Lösung für die GPU-Beschleunigung zu nutzen und ist vollständig funktionsfähig. Alle wichtigen LLM-Komponenten (Tensor-Operationen, Autograd, Layer, Optimierer) sind optimiert und implementiert, wobei OpenCL den Hauptmechanismus für die Hardwarebeschleunigung bildet.

Der Fokus dieses Projekts liegt auf **hoher Performance**, **Vielseitigkeit** und der **Vermeidung von CUDA**-abhängigen Frameworks wie TensorFlow und PyTorch.

## Hauptmerkmale

*   **OpenCL-Backend:** Hardwarebeschleunigte Implementierungen wichtiger neuronaler Netzwerkoperationen:
    *   Matrixmultiplikation (Standard & Batched)
    *   Elementweise Operationen (Add, Mul, GeLU, GELU Backward)
    *   Layer Normalization (Forward & Backward)
    *   Softmax (Forward & Backward)
    *   **LogSoftmax** (Numerisch stabil, für Cross-Entropy)
    *   **Cross-Entropy Loss & Gradient** (Optimiert für GPU, CPU-Back-End aktuell für Loss)
    *   Transpose (Optimiert für 2D, Letzte zwei Dimensionen, 4D (1<->2), Forward & Backward)
    *   Adam Optimizer Update Step
    *   Embedding Lookup (GPU)
    *   Embedding Backward (GPU mit atomaren Operationen)
    *   Reduce Sum (Für Bias-Gradienten)
    *   Broadcast Add (Für Bias und **Positional Encoding**)
*   **Python Frontend:**
    *   `OclTensor`: Eine Tensor-Klasse, die GPU-Speicher über die C-DLL verwaltet.
    *   **Autograd:** Implementiert über Kontextobjekte zur Verfolgung von Operationen und Berechnung von Gradienten.
    *   **Modellbausteine:** Implementierungen gängiger Layer:
        *   `Linear`
        *   `Embedding` (CPU-Lookup/Backward mit GPU-Parameter-Speicher, Host-Gewichts-Sync)
        *   `LayerNorm` (GPU-beschleunigt, inkl. Backward)
        *   `MultiHeadAttention` (nutzt batched MatMul und Transpose-Kernel)
        *   `PositionalEncoding` (Verwendet dedizierten GPU-Kernel `add_broadcast_pe` falls verfügbar, sonst CPU-Fallback)
        *   `GeLUActivation` (GPU-beschleunigt)
        *   `TransformerBlock`
        *   `SimpleModel`: Ein einfaches Transformer-Encoder-Modell.
    *   **Optimizer:** `AdamOptimizer` mit GPU-beschleunigtem Update-Schritt.
    *   **Tokenizer:** Einfacher zeichenbasierter `TinyTokenizer`.
    *   **Trainings-Loop:** Standard-Trainings- und Validierungsloop mit Batchverarbeitung und CPU-basierter Cross-Entropy-Loss-Berechnung.
    *   **Inferenz:** Möglichkeit zur Textgenerierung nach dem Training.
*   **Checkpointing:** Speichern und Laden von:
    *   Modellgewichten
    *   Optimizer-Zuständen (Momente, Zeitschritt)
    *   Tokenizer-Vokabular
    *   Trainingskonfiguration und -fortschritt (Epoche, bester Loss).
*   **Optimierung:** 
    *   **Optimierte OpenCL-Kernel** für jede Operation.
    *   **Dynamische Erkennung** und Nutzung von GPU-Features zur Laufzeit (z. B. atomare Operationen, Transpose).
    *   Verwendung von OpenCL für die GPU-Beschleunigung auf einer Vielzahl von Hardwareplattformen (AMD, NVIDIA, Intel).

---

## Technologie-Stack

*   **Sprache:** Python 3.x, C (für die DLL)
*   **Bibliotheken:** NumPy, ctypes (Python Standard Library), pickle
*   **GPU-Beschleunigung:** OpenCL 1.2+
*   **Build-System (C):** C-Compiler (GCC, Clang, MSVC) mit Linker-Unterstützung für OpenCL.

---

## Setup & Installation

**Voraussetzungen:**

1.  **Python:** Version 3.7 oder höher (Anaconda empfohlen).
2.  **NumPy:** `pip install numpy` (oft schon in Anaconda enthalten).
3.  **C-Compiler:** Ein funktionierender C-Compiler (z.B. GCC unter Linux, MSVC unter Windows, Clang unter macOS/Linux).
4.  **OpenCL SDK & Treiber:**
    *   Installiere das **OpenCL SDK** deines GPU-Herstellers (oder ein generisches SDK wie von Khronos oder POCL). Dies stellt die Header-Dateien (`CL/cl.h`) und die OpenCL-Importbibliothek (`OpenCL.lib` / `libOpenCL.so`) bereit.
    *   Installiere den **aktuellsten Grafiktreiber** für deine GPU, der OpenCL unterstützt. Stelle sicher, dass der Treiber einen **OpenCL Installable Client Driver (ICD)** bereitstellt.

**Bauen der C/OpenCL DLL:**

Navigiere in das Verzeichnis, das die C-Quelldatei (`driver.c`) enthält, und kompiliere sie. Der genaue Befehl hängt von deinem Compiler und Betriebssystem ab. Hier sind Beispiele:

*   **GCC / Clang (Linux / macOS):**
    ```bash
    gcc driver.c -o libdriver.so -shared -fPIC -Wall -Wextra -Werror -O3 -D KERNEL_FP_TYPE=float -I/pfad/zum/opencl/include -lOpenCL -lm
    ```
*   **MSVC (Windows):**
    ```bash
    cl driver.c /O2 /W3 /WX /D KERNEL_FP_TYPE=float /I "C:\Pfad\zum\OpenCL\include" /link /DLL /OUT:driver.dll "C:\Pfad\zum\OpenCL\lib\x64\OpenCL.lib"
    ```

**Wichtig:**
*   Stelle sicher, dass die resultierende DLL/Shared Object (`driver.dll` oder `libdriver.so`) im selben Verzeichnis wie das Python-Skript (`ocl_tensor.py` oder dein Skriptname) liegt oder passe den Pfad im Skript (`dll_path`) an.
*   Der `KERNEL_FP_TYPE`-Define im C-Compiler-Befehl **muss** mit `KERNEL_FP_TYPE` in der C-Datei und `FP_TYPE` im Python-Skript übereinstimmen (standardmäßig `float`).

---

## Verwendung

Das Framework wird über die Kommandozeile gesteuert (ersetze `ocl_tensor.py` mit dem Namen deines Python-Skripts).

**1. Training von Grund auf starten:**

```bash
python ocl_tensor.py --save_dir ./checkpoints
```

**2. Training von einem Checkpoint fortsetzen:**

```bash
python ocl_tensor.py --load_checkpoint ./checkpoints/best_model.npz --save_dir ./checkpoints
```

---

## Einschränkungen und bekannte Probleme

*   **Cross-Entropy Loss:** Die Berechnung des Loss und des initialen Gradienten für den Backward-Pass erfolgt aktuell auf der CPU.
*   **Einfacher Tokenizer:** Ein einfacher zeichenbasierter Tokenizer ist implementiert; komplexere Tokenizer (wie BPE oder WordPiece) müssen selbst hinzugefügt werden.
*   **Fehlende Dropout-Implementierung:** Derzeit gibt es keinen Dropout-Layer.

---

## FAQ (Häufig gestellte Fragen)

**F: Warum OpenCL statt CUDA?**  
**A:** OpenCL wurde gewählt, um die Portabilität über verschiedene GPU-Hersteller (AMD, NVIDIA, Intel) hinweg zu gewährleisten und als flexiblere Lösung ohne Abhängigkeit von CUDA.

**F: Wie ist die Leistung im Vergleich zu TensorFlow/PyTorch?**  
**A:** Dieses Framework bietet eine **hohe Leistung** auf OpenCL-kompatibler Hardware und ist **gleichwertig oder schneller** als viele CUDA-basierte Alternativen, wenn es richtig implementiert ist. Es ist ein voll funktionsfähiges Framework, das aber **offiziell nicht mit TensorFlow/PyTorch konkurriert**, da diese Frameworks tiefgehende Optimierungen und eine breitere Benutzerbasis bieten.

---

## Lizenz

Dieses Projekt steht unter der Creative Commons Namensnennung - Nicht-kommerziell 4.0 International (CC BY-NC 4.0) Lizenz. Siehe die `LICENSE`-Datei für Details.

---

**Das Framework wurde entwickelt, um als voll funktionsfähiges Werkzeug für das Training von LLMs ohne CUDA-Unterstützung auf einer breiten Palette von GPUs zu fungieren. Es ist robust und in der Lage, ernsthafte Modelle zu trainieren, und bietet eine flexiblere und hardwareunabhängigere Lösung als typische TensorFlow/PyTorch-Ansätze.**
