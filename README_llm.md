
# Python OpenCL LLM Framework

Dieses Projekt implementiert ein einfaches Framework für das Training und die Ausführung von kleinen Transformer-basierten Sprachmodellen (LLMs) in Python. Es nutzt eine benutzerdefinierte C-Bibliothek mit OpenCL-Kerneln zur Beschleunigung rechenintensiver Operationen auf GPUs. Das Framework beinhaltet automatische Differentiation (Autograd), gängige neuronale Netzwerk-Layer, einen Adam-Optimizer und Funktionen zum Speichern/Laden von Checkpoints.

## Features

*   **GPU-Beschleunigung:** Nutzt eine C/OpenCL-Backend-DLL (`driver.dll`/`.so`/`.dylib`) für Operationen wie Matrixmultiplikation (Standard & Batched), Layer Normalization, Softmax, GELU, elementweise Operationen, spezialisierte Transponierungen und den Adam-Optimierungsschritt.
*   **Autograd Engine:** Eine einfache, PyTorch-ähnliche Engine für automatische Differentiation, die das Training von Modellen mittels Backpropagation ermöglicht.
*   **Neuronale Netzwerk-Layer:** Enthält Bausteine wie `Linear`, `Embedding` (mit CPU-Fallback), `LayerNorm`, `MultiHeadAttention` (mit spezialisierten Transpose-Kerneln), `TransformerBlock`, `PositionalEncoding` (CPU).
*   **Adam Optimizer:** Eine Implementierung des Adam-Optimizers, dessen Zustände (Momentum, Varianz) auf der GPU gehalten und aktualisiert werden.
*   **Checkpointing:** Ermöglicht das Speichern und Laden des vollständigen Trainingszustands:
    *   Modellkonfiguration
    *   Modellgewichte (`Parameter`)
    *   Optimizer-Zustand (`m`, `v`, `t`)
    *   Tokenizer-Vokabular
    *   Trainings-Metadaten (Epoche, bester Validierungsverlust)
    *   Verwendet `numpy.savez_compressed` und `pickle`.
*   **CPU-Fallbacks:** Implementiert Fallbacks auf NumPy/CPU für Operationen, die (noch) nicht auf der GPU implementiert sind (z.B. Embedding Lookup/Backward, einige Broadcasting-Szenarien, Cross-Entropy-Loss). *Dies hat Auswirkungen auf die Performance.*
*   **Flexibilität:** Die Modellarchitektur und Trainingsparameter werden über ein Konfigurations-Dictionary gesteuert.
*   **Kommandozeilen-Interface:** Ermöglicht das Starten des Trainings, das Fortsetzen von Checkpoints, das Angeben des Speicherorts und die Auswahl des GPU-Geräts über Argumente.
*   **Speicherverwaltung:** Nutzt `weakref` und explizite `free_memory`-Aufrufe zur Verwaltung von GPU-Ressourcen.

## Anforderungen

1.  **Python:** Python 3.x empfohlen.
2.  **NumPy:** `pip install numpy`
3.  **Kompilierte C/OpenCL DLL:** Die Backend-Bibliothek (`driver.dll`, `libgpudriver.so`, `libgpudriver.dylib`) muss kompiliert und für das Python-Skript auffindbar sein (z.B. im selben Verzeichnis oder im Systempfad). Siehe die `README.md` des C/OpenCL-Codes für Kompilierungsanweisungen.
4.  **OpenCL Runtime/Treiber:** Eine funktionierende OpenCL 1.2 (oder höher) Installation und Treiber für die Ziel-GPU (oder CPU).

## Installation / Setup

1.  **Kompilieren Sie die C/OpenCL DLL:** Folgen Sie den Anweisungen in der `README.md` des C-Codes, um die `driver.dll` (oder `.so`/`.dylib` für Linux/macOS) zu erstellen.
2.  **Platzieren Sie die DLL:** Kopieren Sie die kompilierte DLL-Datei in dasselbe Verzeichnis wie das Python-Skript (`ocl_framework.py`) oder an einen Ort, der im Systempfad enthalten ist.
3.  **Installieren Sie NumPy:**
    ```bash
    pip install numpy
    ```
4.  **Bereiten Sie Trainingsdaten vor:** Erstellen Sie eine Textdatei (standardmäßig `input.txt`) mit Ihrem Trainingskorpus.

## Verwendung

Das Training wird über die Kommandozeile gestartet.

```bash
python ocl_framework.py [OPTIONEN]
```

**Wichtige Optionen:**

*   `--gpu_id INT`: Index des zu verwendenden OpenCL-GPU-Geräts (Standard: `0`).
*   `--load_checkpoint PFAD`: Pfad zur `.npz`-Checkpoint-Datei, von der das Training fortgesetzt werden soll.
*   `--save_dir PFAD`: Verzeichnis, in dem Checkpoints gespeichert werden sollen (Standard: `./checkpoints`).

**Beispiele:**

1.  **Training von Grund auf starten (GPU 0, speichert in `./checkpoints`):**
    ```bash
    python ocl_framework.py
    ```

2.  **Training auf GPU 1 starten:**
    ```bash
    python ocl_framework.py --gpu_id 1
    ```

3.  **Training fortsetzen von einem Checkpoint:**
    ```bash
    python ocl_framework.py --load_checkpoint ./checkpoints/best_model.npz
    ```

4.  **Training fortsetzen und Checkpoints in einem anderen Verzeichnis speichern:**
    ```bash
    python ocl_framework.py --load_checkpoint ./old_run/best_model.npz --save_dir ./new_run_saves
    ```

## Code-Struktur (Übersicht)

*   **`ocl_framework.py`:** Hauptskript.
    *   **DLL Binding & Signaturen:** Lädt die C-DLL und definiert Typen für die C-Funktionen.
    *   **`GPUBuffer` Klasse:** Wrapper für OpenCL-Speicherpuffer (`cl_mem`). Kümmert sich um Allokation, Freigabe, Datenübertragung (Host <-> GPU) und Klonen.
    *   **`OclTensor` Klasse:** Kernklasse für Tensoren auf der GPU.
        *   Verwaltet `GPUBuffer` für Daten und Gradienten.
        *   Implementiert Operationen (`matmul`, `add`, `softmax`, etc.) durch Aufruf der C-Funktionen.
        *   Enthält die Logik für die automatische Differentiation (`backward`, `_ctx`).
        *   Implementiert CPU-Fallbacks für nicht beschleunigte Operationen.
    *   **Autograd Context Klassen (`FunctionContext`, ...):** Speichern Informationen aus dem Forward Pass für den Backward Pass und definieren die Gradientenberechnung für jede Operation.
    *   **Layer Klassen (`Linear`, `Embedding`, ...):** Definieren Bausteine für neuronale Netze. `Parameter` ist eine Unterklasse von `OclTensor` mit `requires_grad=True`.
    *   **`SimpleModel` Klasse:** Beispielhafte Transformer-Architektur, die die Layer kombiniert. Nimmt eine `config` entgegen und implementiert `load_state_dict`.
    *   **`AdamOptimizer` Klasse:** Implementiert den Adam-Optimizer mit GPU-beschleunigtem Update-Schritt und GPU-Speicher für Zustände (`m`, `v`). Implementiert `state_dict` und `load_state_dict`.
    *   **`TinyTokenizer` Klasse:** Einfacher zeichenbasierter Tokenizer mit Speicher-/Ladefunktionen für das Vokabular.
    *   **`cross_entropy_loss_and_backward`:** Verlustfunktion (CPU-basiert), die auch den Backward-Pass initiiert.
    *   **Init/Shutdown/Cleanup:** Funktionen zur Verwaltung des OpenCL-Kontexts und des Speichers.
    *   **Checkpointing Funktionen (`save_checkpoint`, `load_checkpoint`):** Implementieren das Speichern und Laden des Trainingszustands.
    *   **`train` Funktion:** Orchestriert den Trainingsprozess, inkl. Datenladen, Trainingsepochen, Validierung und Checkpointing.
    *   **`run_inference` Funktion:** Führt Inferenz auf einem Beispieltext durch.
    *   **Main Guard (`if __name__ == "__main__":`):** Parst Kommandozeilenargumente, initialisiert OpenCL, startet das Training und kümmert sich um das Herunterfahren.

## GPU vs. CPU Operationen

Dieses Framework beschleunigt viele, aber nicht alle Operationen auf der GPU.

**GPU-Beschleunigt (über C/OpenCL DLL):**

*   Matrixmultiplikation (Standard & Batched)
*   Addition, Multiplikation (elementweise)
*   GELU-Aktivierung (Forward & Backward)
*   Layer Normalization (Forward & Backward)
*   Softmax (Forward & Backward)
*   Transponierung (2D, letzte zwei Dimensionen gebatched, Dimensionen 1&2 gebatched in 4D)
*   Adam Optimizer Update Schritt
*   Speicheroperationen (Allokation, Transfer, Klonen)

**CPU-basiert (Python/NumPy):**

*   Embedding Lookup (Forward Pass)
*   Embedding Gradientenberechnung (Backward Pass, Scatter-Add)
*   Positional Encoding Addition (Forward Pass)
*   Gradientenreduktion für Broadcasting in `add` (Backward Pass)
*   Cross-Entropy-Loss Berechnung
*   Gradientenberechnung für Cross-Entropy (Backward Pass Start)
*   Transpose für nicht speziell implementierte Dimensionspermutationen
*   Tokenizer-Operationen (`encode`/`decode`)
*   Datenvorverarbeitung und Batching

## Checkpointing

*   Checkpoints werden als `.npz`-Dateien mithilfe von `numpy.savez_compressed` gespeichert.
*   Sie enthalten Python-Objekte (Dictionaries für Konfiguration, Zustände, Vokabular), die mittels `pickle` serialisiert und dann in NumPy-Arrays vom Typ `object` verpackt werden.
*   **Gespeicherte Daten:**
    *   `config`: Das Konfigurations-Dictionary des Modells/Trainings.
    *   `model_state`: Dictionary, das die Gewichte (`Parameter`) des Modells als NumPy-Arrays enthält (von der GPU gelesen). Die Schlüssel sind `param_0`, `param_1`, etc., basierend auf der Reihenfolge in `model.parameters()`.
    *   `optimizer_state`: Dictionary mit dem Optimizer-Zustand (`t`, `m`, `v`). `m` und `v` sind wiederum Dictionaries, die die Zustands-Arrays (von der GPU gelesen) enthalten, mit Schlüsseln `state_0`, `state_1`, etc., basierend auf der Reihenfolge der Parameter im Optimizer.
    *   `tokenizer_vocab`: Das Vokabular-Dictionary (`char -> id`).
    *   `tokenizer_inv_vocab`: Das inverse Vokabular-Dictionary (`id -> char`).
    *   `epoch`: Die Epoche, *nach* der der Checkpoint gespeichert wurde.
    *   `best_val_loss`: Der bisher beste erreichte Validierungsverlust.
*   Beim Laden (`load_checkpoint`) werden neue Instanzen von Modell, Optimizer und Tokenizer erstellt und deren Zustände mit den geladenen Daten initialisiert. Das Training wird ab der `start_epoch` (geladene Epoche + 1) fortgesetzt.

## Bekannte Einschränkungen

*   **CPU-Flaschenhälse:** Operationen, die auf der CPU ausgeführt werden (insbesondere Embedding, Loss, einige Gradientenberechnungen), können die Gesamtleistung limitieren.
*   **Einfacher Tokenizer:** Der zeichenbasierte Tokenizer ist sehr grundlegend.
*   **Kein Dropout:** Dropout-Layer sind nicht implementiert.
*   **Begrenzte Fehlerprüfung:** Obwohl grundlegende Prüfungen vorhanden sind, könnte die Fehlerbehandlung in Randfällen verbessert werden.
*   **Speicherverwaltung:** Benötigt sorgfältige manuelle Speicherfreigabe (`free_memory`), obwohl `weakref` und `__del__` helfen, Lecks zu reduzieren.
*   **Abhängigkeit von DLL:** Das Python-Skript funktioniert nur mit der korrekt kompilierten und auffindbaren C/OpenCL-Backend-DLL.

## Lizenz

[TODO: Fügen Sie hier Ihre Lizenzinformationen ein. Z.B. MIT, Apache 2.0, etc.]
