# OCL-LLM-Framework: Ein Experimentelles LLM-Framework mit OpenCL-Beschleunigung

[![Lizenz: MIT](https://img.shields.io/badge/Lizenz-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
    *   Elementweise Operationen (Add, Mul, GeLU, GELU Backward)
    *   Layer Normalization (Forward & Backward)
    *   Softmax (Forward & Backward)
    *   **NEU:** LogSoftmax (Numerisch stabil, für Cross-Entropy)
    *   **NEU:** Cross-Entropy Loss & Gradient (Kernel vorhanden, *aber Python verwendet derzeit CPU-Version*)
    *   Transpose (Optimiert für 2D, Letzte zwei Dimensionen, 4D (1<->2), jew. Forward & Backward)
    *   Adam Optimizer Update Step
    *   Embedding Lookup (GPU)
    *   Embedding Backward (GPU mit atomaren Operationen)
    *   Reduce Sum (Für Bias-Gradienten)
    *   Broadcast Add (Für Bias und **NEU: Positional Encoding**)
*   **Python Frontend:**
    *   `OclTensor`: Eine Tensor-Klasse, die GPU-Speicher über die C-DLL verwaltet.
    *   **Automatischer Differenzierungsmechanismus (Autograd):** Implementiert über Kontextobjekte zur Verfolgung von Operationen und Berechnung von Gradienten.
    *   **Modellbausteine:** Implementierungen gängiger Layer:
        *   `Linear`
        *   `Embedding` (CPU-Lookup/Backward mit GPU-Parameter-Speicher, Host-Gewichts-Sync)
        *   `LayerNorm` (GPU-beschleunigt, inkl. Backward)
        *   `MultiHeadAttention` (nutzt batched MatMul und Transpose-Kernel)
        *   `PositionalEncoding` (**NEU:** Verwendet dedizierten GPU-Kernel `add_broadcast_pe` falls verfügbar, sonst CPU-Fallback)
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
    *   Kompiliert die OpenCL-Kernel (Code als C-Strings eingebettet).
    *   Stellt C-Funktionen bereit, die von Python aufgerufen werden können, um:
        *   GPU-Speicher zu allokieren/freizugeben.
        *   Daten zwischen Host (CPU) und Device (GPU) zu kopieren.
        *   Die kompilierten OpenCL-Kernel mit den entsprechenden Argumenten und Arbeitsdimensionen zu starten (`submit_kernel_command`).

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
    # Passe FP_TYPE ggf. an (z.B. double) und stelle sicher, dass -I auf dein OpenCL Include-Verzeichnis zeigt
    # -Werror behandelt Warnungen als Fehler (kann bei Atomics-Warnungen problematisch sein)
    # -D CL_HAS_ATOMICS=1 (oder 0) könnte explizit gesetzt werden, aber die C-Datei versucht es selbst zu erkennen.
    gcc driver.c -o libdriver.so -shared -fPIC -Wall -Wextra -Werror -O3 -D KERNEL_FP_TYPE=float -I/pfad/zum/opencl/include -lOpenCL -lm
    # oder clang
    clang driver.c -o libdriver.so -shared -fPIC -Wall -Wextra -Werror -O3 -D KERNEL_FP_TYPE=float -I/pfad/zum/opencl/include -lOpenCL -lm
    ```
*   **MSVC (Windows):**
    ```bash
    # Öffne eine Developer Command Prompt für VS
    # Passe FP_TYPE ggf. an und stelle sicher, dass /I auf dein OpenCL Include- und /LIBPATH auf dein Lib-Verzeichnis zeigt
    # /WX behandelt Warnungen als Fehler (kann bei Atomics-Warnungen problematisch sein)
    cl driver.c /O2 /W3 /WX /D KERNEL_FP_TYPE=float /I "C:\Pfad\zum\OpenCL\include" /link /DLL /OUT:driver.dll "C:\Pfad\zum\OpenCL\lib\x64\OpenCL.lib"
    ```

**Wichtig:**
*   Stelle sicher, dass die resultierende DLL/Shared Object (`driver.dll` oder `libdriver.so`) im selben Verzeichnis wie das Python-Skript (`ocl_tensor.py` oder dein Skriptname) liegt oder passe den Pfad im Skript (`dll_path`) an.
*   Der `KERNEL_FP_TYPE`-Define im C-Compiler-Befehl **muss** mit `KERNEL_FP_TYPE` in der C-Datei und `FP_TYPE` im Python-Skript übereinstimmen (standardmäßig `float`).
*   Die C-Datei versucht, FP64- und Atomics-Unterstützung zur Laufzeit zu erkennen und entsprechende `-D`-Flags (`CL_HAS_FP64`, `CL_HAS_ATOMICS`) für die Kernel-Kompilierung zu setzen.

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

*   Die Inferenz wird automatisch am Ende des Trainings (sowohl beim Start von Grund auf als auch beim Fortsetzen) und periodisch während des Trainings ausgeführt (standardmäßig jede Epoche).
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
    *   Embedding-Lookup und dessen Gradientenberechnung (Scatter-Add). **Achtung:** Wenn die GPU keine ausreichende Atomics-Unterstützung für `FP_TYPE` hat (siehe `[C] WARN: ... atomics support ... potentially UNSUPPORTED`), kann der Embedding-Backward-Schritt zu **fehlerhaften Gradienten** führen (Race Conditions), selbst wenn der Kernel kompiliert. Die Laufzeitprüfung in Python versucht dies zu verhindern, aber es bleibt eine Einschränkung.
    *   Cross-Entropy-Loss-Berechnung und der initiale Gradient für den Backward-Pass (`cross_entropy_loss_and_backward` in Python). Der OpenCL-Kernel `cross_entropy_loss_grad` wird derzeit **nicht** vom Python-Code genutzt.
    *   Gradienten-Reduktion für Broadcasting-Operationen (z.B. beim Addieren von Bias), falls der `ReduceSum`-Kernel nicht verfügbar ist (`HAS_REDUCE_SUM == False`).
    *   Transpose-Operationen für Achsenkombinationen, die nicht durch dedizierte Kernel (`basic`, `last_two`, `transpose12`) abgedeckt sind.
*   **Kein Dropout:** Dropout-Layer sind derzeit nicht implementiert.
*   **Einfacher Tokenizer:** Verwendet einen einfachen zeichenbasierten Tokenizer. Für komplexere Aufgaben wären Subword-Tokenizer (wie BPE oder WordPiece) erforderlich.
*   **Checkpoint-Kompatibilität:** Das Laden von Checkpoints funktioniert nur, wenn die Modellarchitektur (Anzahl Layer, Dimensionen etc.) exakt mit der beim Speichern übereinstimmt, da die Parameter anhand ihrer Reihenfolge geladen werden.
*   **Feste Präzision:** Das Framework ist derzeit auf `float32` (`FP_TYPE`) festgelegt. Eine Änderung erfordert Anpassungen im Python- *und* C-Code sowie eine Neukompilierung der DLL mit dem passenden `-D KERNEL_FP_TYPE=...`-Flag.

---

## FAQ (Häufig gestellte Fragen)

**F: Warum OpenCL statt CUDA?**
A: OpenCL wurde gewählt, um die Portabilität über verschiedene GPU-Hersteller (AMD, NVIDIA, Intel) hinweg zu demonstrieren und als Lernübung für Low-Level-GPU-Programmierung zu dienen. CUDA ist oft performanter, aber auf NVIDIA-Hardware beschränkt.

**F: Wie ist die Leistung im Vergleich zu PyTorch/TensorFlow mit CUDA?**
A: Dieses Framework ist **experimentell** und wird voraussichtlich **deutlich langsamer** sein als optimierte Frameworks wie PyTorch/TensorFlow, die auf hochoptimierten Bibliotheken (cuDNN, cuBLAS) aufbauen. Der Fokus liegt auf der Demonstration des Konzepts.

**F: Welche GPUs werden unterstützt?**
A: Theoretisch jede GPU, für die ein funktionierender OpenCL 1.2 (oder neuer) Treiber und ein entsprechendes SDK verfügbar sind. Die tatsächliche Leistung und Kompatibilität (insbesondere bei Atomics für Embedding Backward) kann variieren.

**F: Wie überprüfe ich meine OpenCL-Installation?**
A: Unter Linux kann das Tool `clinfo` detaillierte Informationen liefern. GPU-Hersteller bieten oft eigene Diagnose-Tools an. Die Startausgabe des Skripts zeigt die erkannte Plattform und das verwendete Gerät an. Fehlermeldungen beim Start deuten auf Probleme mit Treibern oder dem SDK hin.

**F: Warum laufen manche Operationen (Embedding, Loss) noch auf der CPU?**
A: Die Implementierung aller Operationen als effiziente GPU-Kernel ist aufwändig. Embedding Backward erfordert atomare Operationen, die nicht auf allen GPUs gleich gut unterstützt werden. Die Loss-Berechnung wurde der Einfachheit halber auf der CPU belassen.

**F: Kann ich FP16 (halbe Präzision) oder FP64 (doppelte Präzision) verwenden?**
A: Derzeit nicht ohne Weiteres. Es müssten der `FP_TYPE` in Python, der `KERNEL_FP_TYPE` in C und das entsprechende Compiler-Flag (`-D KERNEL_FP_TYPE=...`) angepasst werden. Zudem müssten die Kernel und die C-Wrapper ggf. für die andere Präzision überarbeitet werden (insbesondere bei FP64-Atomics).

**F: Wie kann ich GPU-Fehler debuggen?**
A: Achte auf Fehlermeldungen in der Konsolenausgabe, die mit `[C]` beginnen. Die C-Bibliothek gibt OpenCL-Fehlercodes und deren Bedeutung aus (`clGetErrorString`). Bei Kernel-Kompilierungsfehlern wird der Build-Log ausgegeben.

**F: Ist dieses Framework für den produktiven Einsatz geeignet?**
A: **Nein.** Es ist ein Proof-of-Concept und ein Lernprojekt. Für ernsthafte Anwendungen sollten etablierte Frameworks wie PyTorch oder TensorFlow verwendet werden.

---

## Glossar

*   **OpenCL (Open Computing Language):** Ein offener Standard für plattformübergreifende parallele Programmierung, insbesondere für GPUs.
*   **Kernel:** Eine Funktion, die auf der GPU ausgeführt wird. Im Code sind dies die C-Strings (`*_kernel_src`).
*   **Context:** Ein OpenCL-Objekt, das den Zustand der OpenCL-Umgebung (Geräte, Speicher etc.) verwaltet.
*   **Command Queue:** Eine Warteschlange, in die Befehle (Kernel-Ausführungen, Speicher-Transfers) für ein bestimmtes OpenCL-Gerät eingereiht werden. OpenCL führt diese Befehle normalerweise sequentiell (pro Queue) aus.
*   **Platform / Device:** Hierarchie in OpenCL. Eine Platform repräsentiert eine OpenCL-Implementierung (z.B. von AMD, NVIDIA), ein Device ist die tatsächliche Hardware (z.B. eine spezifische GPU).
*   **Work-Item / Work-Group:** Die grundlegenden Einheiten der parallelen Ausführung in OpenCL. Viele Work-Items bilden eine Work-Group.
*   **Autograd:** Automatischer Differenzierungsmechanismus, der Gradienten für die Optimierung berechnet.
*   **Tensor (`OclTensor`):** Eine mehrdimensionale Datenstruktur (Array), deren Daten im GPU-Speicher liegen.
*   **DLL (Dynamic Link Library) / Shared Object (.so):** Eine kompilierte Bibliothek (hier in C geschrieben), die Funktionen enthält, welche von anderen Programmen (hier Python) zur Laufzeit geladen und aufgerufen werden können.
*   **ctypes:** Eine Python-Standardbibliothek, die das Aufrufen von Funktionen in C-kompilierten Bibliotheken (DLLs/.so) ermöglicht.
*   **Checkpoint:** Ein gespeicherter Zustand des Modells und des Trainingsfortschritts, um das Training später fortsetzen zu können.
*   **Atomics:** Operationen (z.B. Addieren, Compare-and-Swap), die garantiert unteilbar (atomar) ausgeführt werden, auch wenn mehrere Threads gleichzeitig darauf zugreifen. Wichtig für korrekte Gradientenakkumulation bei Operationen wie Embedding Backward.
*   **FP_TYPE:** Der im Framework verwendete Fließkomma-Datentyp (z.B. `numpy.float32`).
*   **Logits:** Die rohen, unnormalisierten Ausgaben eines Modells vor der Anwendung einer Aktivierungsfunktion wie Softmax.
*   **Cross-Entropy Loss:** Eine gängige Verlustfunktion für Klassifikationsaufgaben, die misst, wie gut die vorhergesagten Wahrscheinlichkeiten mit den tatsächlichen Zielklassen übereinstimmen.
*   **Positional Encoding (PE):** Informationen über die Position von Tokens in einer Sequenz, die den Eingabe-Embeddings hinzugefügt werden, da Transformer keine inhärente Vorstellung von Reihenfolge haben.
*   **Reduce Sum:** Eine Operation, die Elemente entlang bestimmter Achsen eines Tensors aufsummiert. Hier z.B. verwendet, um Gradienten für Bias-Terme zu aggregieren.
*   **Broadcast:** Eine Technik, bei der Tensoren mit unterschiedlichen, aber kompatiblen Formen für elementweise Operationen behandelt werden, indem kleinere Tensoren implizit erweitert werden.

---

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe die `LICENSE`-Datei für Details. (Füge eine `LICENSE`-Datei mit dem MIT-Lizenztext hinzu, falls noch nicht geschehen).

