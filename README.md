# OCL-LLM-Framework: Ein Experimentelles LLM-Framework mit OpenCL-Beschleunigung

[![Lizenz: MIT](https://img.shields.io/badge/Lizenz-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Füge hier ggf. weitere Badges hinzu -->

**Ein experimentelles Framework für das Training und die Inferenz von einfachen Large Language Models (LLMs), das GPU-Beschleunigung über benutzerdefinierte OpenCL-Kernel demonstriert.**

---

## Übersicht

Dieses Projekt ist ein von Grund auf neu aufgebautes Framework zur Erstellung und zum Training einfacher Transformer-basierter Sprachmodelle. Anstatt sich auf etablierte Deep-Learning-Bibliotheken wie PyTorch oder TensorFlow zu verlassen, implementiert dieses Projekt die Kernkomponenten (Tensor-Operationen, Autograd, Layer, Optimizer) selbst.

Das Hauptziel ist die **Demonstration und Erforschung der GPU-Beschleunigung mittels OpenCL**. Die rechenintensiven Operationen werden durch eine benutzerdefinierte C-Bibliothek (`driver.dll` / `libdriver.so`) bereitgestellt, die optimierte OpenCL-Kernel für kompatible GPUs ausführt. Die Python-Schicht kümmert sich um die Modelllogik, den Trainingsprozess und die Interaktion mit der C/OpenCL-Backend-Bibliothek.

**Dieses Projekt dient primär als Lern- und Demonstrationswerkzeug und ist nicht für den produktiven Einsatz im Vergleich zu ausgereiften Frameworks gedacht.**

---

## Hauptmerkmale

*   **Benutzerdefinierte OpenCL-Kernel:** Hardware-beschleunigte Implementierungen für kritische Operationen:
    *   Matrixmultiplikation (Standard & Batched)
    *   Elementweise Operationen (Add, Mul, GeLU)
    *   Layer Normalization
    *   Softmax
    *   Transpose (Optimiert für spezifische Fälle: 2D, Letzte zwei Dimensionen, 4D (1<->2))
    *   Adam Optimizer Update Step
    *   (Weitere Kernels können in der C-Bibliothek vorhanden sein)
*   **Python Frontend:**
    *   `OclTensor`: Eine Tensor-Klasse, die GPU-Speicher über die C-DLL verwaltet.
    *   **Automatischer Differenzierungsmechanismus (Autograd):** Implementiert über Kontextobjekte zur Verfolgung von Operationen und Berechnung von Gradienten.
    *   **Modellbausteine:** Implementierungen gängiger Layer:
        *   `Linear`
        *   `Embedding` (CPU-Lookup mit GPU-Parameter-Speicher)
        *   `LayerNorm`
        *   `MultiHeadAttention` (nutzt batched MatMul und Transpose-Kernel)
        *   `PositionalEncoding` (CPU-basiert)
        *   `TransformerBlock`
        *   `SimpleModel`: Ein einfaches Transformer-Encoder-Modell.
    *   **Optimizer:** `AdamOptimizer` mit GPU-beschleunigtem Update-Schritt.
    *   **Tokenizer:** Einfacher zeichenbasierter `TinyTokenizer`.
    *   **Trainings-Loop:** Standard-Trainings- und Validierungsloop mit Batchverarbeitung.
    *   **Inferenz:** Möglichkeit zur Textgenerierung nach dem Training.
*   **Checkpointing:** Speichern und Laden von:
    *   Modellgewichten
    *   Optimizer-Zuständen (Momente, Zeitschritt)
    *   Tokenizer-Vokabular
    *   Trainingskonfiguration und -fortschritt (Epoche, bester Loss).
*   **GPU-unabhängiger Ansatz (theoretisch):** Nutzt OpenCL, das auf einer Vielzahl von GPUs (AMD, NVIDIA, Intel) laufen kann, sofern entsprechende Treiber und SDKs installiert sind.

---

## Architektur

Das Framework besteht aus zwei Hauptteilen:

1.  **Python Frontend (`ocl_tensor.py` oder dein Skriptname):**
    *   Definiert die Modellarchitektur und die Trainingslogik.
    *   Verwaltet den Lebenszyklus von `OclTensor`-Objekten.
    *   Implementiert den Autograd-Mechanismus.
    *   Ruft über `ctypes` die Funktionen der C/OpenCL-Bibliothek auf.
2.  **C/OpenCL Backend (`driver.c` / `driver.dll`):**
    *   Initialisiert die OpenCL-Umgebung (Plattform, Gerät, Kontext, Queue).
    *   Kompiliert die OpenCL-Kernel (`.cl`-Code eingebettet als Strings).
    *   Stellt C-Funktionen bereit, die von Python aufgerufen werden können, um:
        *   GPU-Speicher zu allokieren/freizugeben.
        *   Daten zwischen Host (CPU) und Device (GPU) zu kopieren.
        *   Die kompilierten OpenCL-Kernel mit den entsprechenden Argumenten und Arbeitsdimensionen zu starten.

---

## Technologie-Stack

*   **Sprache:** Python 3.x, C (für die DLL)
*   **Bibliotheken:** NumPy, ctypes (Python Standard Library)
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
    # Passe FP_TYPE ggf. an (z.B. double) und stelle sicher, dass -I auf dein OpenCL Include-Verzeichnis zeigt
    # Entferne ggf. -Werror, falls es Probleme mit Atomics-Warnungen gibt
    gcc driver.c -o libdriver.so -shared -fPIC -Wall -Wextra -O3 -D FP_TYPE=float -I/pfad/zum/opencl/include -lOpenCL
    # oder clang
    clang driver.c -o libdriver.so -shared -fPIC -Wall -Wextra -O3 -D FP_TYPE=float -I/pfad/zum/opencl/include -lOpenCL
    ```
*   **MSVC (Windows):**
    ```bash
    # Öffne eine Developer Command Prompt für VS
    # Passe FP_TYPE ggf. an und stelle sicher, dass /I auf dein OpenCL Include- und /LIBPATH auf dein Lib-Verzeichnis zeigt
    # Entferne ggf. /WX (behandelt Warnungen als Fehler), falls es Probleme mit Atomics-Warnungen gibt
    cl driver.c /O2 /W3 /D FP_TYPE=float /I "C:\Pfad\zum\OpenCL\include" /link /DLL /OUT:driver.dll "C:\Pfad\zum\OpenCL\lib\x64\OpenCL.lib"
    ```

**Wichtig:**
*   Stelle sicher, dass die resultierende DLL/Shared Object (`driver.dll` oder `libdriver.so`) im selben Verzeichnis wie das Python-Skript (`ocl_tensor.py` oder dein Skriptname) liegt oder passe den Pfad im Skript an.
*   Der `FP_TYPE`-Define im C-Compiler-Befehl **muss** mit `KERNEL_FP_TYPE` in der C-Datei und `FP_TYPE` im Python-Skript übereinstimmen (standardmäßig `float`).

**Datensatz:**

*   Erstelle eine Textdatei namens `input.txt` im selben Verzeichnis wie dein Python-Skript. Füge hier deinen Trainings-Textkorpus ein (z.B. Code, Prosa, der Buchauszug). Je größer und diverser, desto besser lernt das Modell tendenziell.

---

## Verwendung

Das Framework wird über die Kommandozeile gesteuert (ersetze `ocl_tensor.py` mit dem Namen deines Python-Skripts).

**1. Training von Grund auf starten:**

*   **Mit Standard-GPU (Index 0):**
    ```bash
    python ocl_tensor.py --save_dir ./checkpoints
    ```
*   **Mit spezifischer GPU (z.B. Index 1):**
    ```bash
    python ocl_tensor.py --save_dir ./checkpoints --gpu_id 1
    ```

    *   Dies startet das Training mit den im Skript definierten Standard-Hyperparametern auf der ausgewählten GPU.
    *   Checkpoints (insbesondere das Modell mit dem besten Validierungsverlust) werden im Verzeichnis, das mit `--save_dir` angegeben wurde (standardmäßig `./checkpoints`), als `best_model.npz` gespeichert.

**2. Training von einem Checkpoint fortsetzen:**

*   **Mit Standard-GPU (Index 0):**
    ```bash
    python ocl_tensor.py --load_checkpoint ./checkpoints/best_model.npz --save_dir ./checkpoints
    ```
*   **Mit spezifischer GPU (z.B. Index 1):**
    ```bash
    python ocl_tensor.py --load_checkpoint ./checkpoints/best_model.npz --save_dir ./checkpoints --gpu_id 1
    ```

    *   Lädt das Modell, den Optimizer-Zustand und den Trainingsfortschritt aus der angegebenen `.npz`-Datei.
    *   Setzt das Training ab der gespeicherten Epoche auf der ausgewählten GPU fort.
    *   Speichert weiterhin verbesserte Checkpoints in `--save_dir`.

**Hinweis zur GPU-Auswahl:**
*   Der `--gpu_id`-Parameter wählt das OpenCL-Gerät basierend auf dem Index aus, den die OpenCL-Implementierung auf deinem System vergibt. `0` ist normalerweise das Standardgerät.
*   Du kannst Tools wie `clinfo` (Linux) oder herstellerspezifische Tools verwenden, um die verfügbaren Geräte und ihre Indizes aufzulisten. Die Initialisierungsausgabe des Skripts (`[C] initialize_gpu: Using device index X: ...`) zeigt ebenfalls an, welches Gerät tatsächlich verwendet wird.

**3. Inferenz:**

*   Die Inferenz wird automatisch am Ende des Trainings (sowohl beim Start von Grund auf als auch beim Fortsetzen) und periodisch während des Trainings ausgeführt (standardmäßig alle 10 Epochen).
*   Es wird ein Beispieltext ("opencl test") verwendet, um die Textgenerierungsfähigkeit des aktuellen Modells zu demonstrieren.

---

## Konfiguration

Die wichtigsten Hyperparameter für das Modell und das Training können direkt im `train()`-Funktionsteil des Python-Skripts (`ocl_tensor.py` oder dein Skriptname) im `config`-Dictionary angepasst werden:

*   `max_len`: Maximale Sequenzlänge.
*   `batch_size`: Anzahl der Sequenzen pro Trainingsschritt.
*   `embed_dim`: Dimension der Token- und Positionseinbettungen.
*   `num_heads`: Anzahl der Köpfe in der Multi-Head Attention.
*   `d_ff`: Dimension der Feed-Forward-Netzwerke in den Transformer-Blöcken.
*   `num_layers`: Anzahl der Transformer-Blöcke.
*   `lr`: Lernrate für den Adam-Optimizer.
*   `num_epochs`: Anzahl der Trainingsepochen.
*   `wd`: Gewichtungszerfall (Weight Decay) für den Adam-Optimizer.
*   `val_split`: Anteil der Daten, der für die Validierung verwendet wird.
*   `data_file`: Pfad zur Trainings-Textdatei.
*   `checkpoint_filename`: Dateiname für den besten gespeicherten Checkpoint.

---

## Bekannte Einschränkungen & Workarounds

*   **CPU-Bottlenecks:** Bestimmte Operationen laufen (noch) auf der CPU, was die Gesamtleistung limitieren kann:
    *   Embedding-Lookup und dessen Gradientenberechnung (Scatter-Add). **Achtung:** Wenn die GPU keine Atomics unterstützt (wie im Log bei GPU 1 angezeigt), kann der Embedding-Backward-Schritt zu **fehlerhaften Gradienten** führen, wenn `-Werror` beim Kompilieren entfernt wurde.
    *   Cross-Entropy-Loss-Berechnung und der initiale Gradient für den Backward-Pass.
    *   Gradienten-Reduktion für Broadcasting-Operationen (z.B. beim Addieren von Bias oder Positional Encoding).
    *   Transpose-Operationen für Achsenkombinationen, die nicht durch dedizierte Kernel abgedeckt sind.
*   **Kein Dropout:** Dropout-Layer sind derzeit nicht implementiert.
*   **Einfacher Tokenizer:** Verwendet einen einfachen zeichenbasierten Tokenizer. Für komplexere Aufgaben wären Subword-Tokenizer (wie BPE oder WordPiece) erforderlich.
*   **Checkpoint-Kompatibilität:** Das Laden von Checkpoints funktioniert nur, wenn die Modellarchitektur (Anzahl Layer, Dimensionen etc.) exakt mit der beim Speichern übereinstimmt, da die Parameter anhand ihrer Reihenfolge geladen werden.
*   **Feste Präzision:** Das Framework ist derzeit auf `float32` (`FP_TYPE`) festgelegt. Eine Änderung erfordert Anpassungen im Python- *und* C-Code sowie eine Neukompilierung der DLL.

---

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe die `LICENSE`-Datei für Details. (Füge eine `LICENSE`-Datei mit dem MIT-Lizenztext hinzu).

---

**Viel Spaß beim Experimentieren mit OpenCL und LLMs!**