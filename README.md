**AetherCL**

---

AetherCL ist eine GPU-Treiber-Simulations-Engine mit echter OpenCL-Interop, die Host↔GPU-Transfers, Kernel-Dispatch und Matrixoperationen realitätsnah abbildet. Ideal für Forschung, Deep-Learning-Prototyping und GPU-Architektursimulation. Plattformübergreifend, erweiterbar, bildungstauglich.

---

### 📘 Ausführliche `README.md` (Version 1.0, inklusive Diagramm-GIF-Platzhalter)

# AetherCL – Simulierte GPU-Engine mit OpenCL-Interop

**AetherCL** ist eine experimentelle, modulare C-basierte GPU-Treiber-Simulation mit echter Host↔GPU-Speicherinteraktion via OpenCL. Sie dient als Forschungssystem für Deep Learning, GPU-Architekturverständnis und realistische Prototyping-Szenarien – z. B. für Matrixoperationen, Kernel-Simulation oder Speicherverifikation.

---

## 🔍 Features

- ✅ Simulierte GPU-Speicherallokation (Host malloc + OpenCL Buffers)
- ✅ OpenCL-basierte Host→GPU→Host Transfers
- ✅ Matrixmultiplikation auf CPU mit echten GPU-Buffern
- ✅ Asynchrones Readback mit `clSetEventCallback`
- ✅ Rückgabe von Ergebnispuffern an Python über ctypes
- ✅ Modularer Aufbau mit klarer Host-/GPU-Trennung
- ✅ Kompatibel mit Windows, Linux (macOS optional)
- ✅ Ideal als LLM-Beschleuniger- oder Custom-Treiber-Basis

---

## 🧠 Zielsetzung

AetherCL richtet sich an:

- AI/ML-Forscher, die eigene Deep Learning Engines aufbauen
- Studierende, die GPU-Architektur realitätsnah verstehen wollen
- Entwickler, die OpenCL-basiertes Speicher- und Kernelmanagement simulieren
- Prototyping von Speicherpipelines, MMIO-Systemen, simulierten Dispatch-Systemen

---

## 🧪 Beispiel: Matrixmultiplikation via `simulated_matrix_multiply`

```python
A = np.random.rand(5, 4)
B = np.random.rand(4, 6)
C_addr = gpu.simulated_matrix_multiply(mmap_ptr_a, mmap_ptr_b, size_a, size_b, shape_a, shape_b)
C = gpu.read_data(C_addr, shape=(5, 6), dtype=np.float64)
```

![Architektur-Diagramm](docs/aethercl_architecture.png)

> 💡 *Die Matrixmultiplikation erfolgt CPU-seitig, aber über echte OpenCL-GPU-Puffer, mit späterem asynchronem Readback!*

---

## 📐 Architektur

```plaintext
┌──────────────┐        Host: Python (ctypes)
│  numpy array │────┐
└──────────────┘    │
                    ▼
           [simulated_kernel_write]
                    │
                    ▼
          ┌────────────────────┐
          │ Host Malloc Memory │ (simuliert mmap)
          └────────────────────┘
                    │
                    ▼
        [write_to_gpu → clEnqueueWriteBuffer]
                    ▼
          ┌───────────────────────┐
          │ OpenCL GPU Buffer (A) │
          └───────────────────────┘
                         ...
        [submit_command → CPU-MatMul]
                         ...
        [clEnqueueReadBuffer → callback]
                    ▼
           [simulated_kernel_read]
                    ▼
           Host-Puffer → NumPy

```

---

## 🔧 Build

### Voraussetzungen

- OpenCL SDK (z. B. Intel, AMD, POCL, NVIDIA)
- GCC oder MSVC
- Python 3.12 mit `ctypes`
- Optional: `numpy` für die Python-Interaktion

### Kompilieren (Windows/GCC Beispiel)

```bash
gcc -I. -L. -shared -o simulated_driver.dll simulated_driver.c -lOpenCL -static-libgcc -static-libstdc++ -Wl,--export-all-symbols
```

---

## 🧪 Testen mit Python

```bash
python app.py
```

> Die Ausgabe zeigt detailreich den Ablauf jeder GPU-Simulation – von Allokation über Matrixmultiplikation bis zum Readback.
```
Versuche, simulierten Treiber zu laden von: G:\amd LLM Treiber code\simulated_driver.dll
Simulierter Treiber erfolgreich geladen.
C-Funktionssignaturen erfolgreich definiert.
Treiber-Shutdown-Funktion für atexit registriert.

--- Start des Hauptprogramms (mit mmap-basiertem Treiber) ---
Initialisiere GPUManager...
Simuliere GPU-Anzahl (fest auf 1).
Manager: 1 GPU(s) gefunden (simuliert).
Simuliere CU-Anzahl für GPU 0 (fest auf 2560).
Manager: Erstelle GPU-Objekt für Index 0 mit 2560 CUs.
GPU Objekt 0 erstellt (CUs: 2560).
GPUManager Initialisierung abgeschlossen.

Verfügbare GPUs im Manager: 1

--- Operationen auf GPU 0 ---
GPU 0: Fordere Initialisierung über Treiber an...
initialize_gpu: Initialisierung erfolgreich (Context: 00000081a47f8980, Queue: 00000081a2da79e0).
GPU 0: Treiber-Initialisierung erfolgreich.

Host: Erstelle Matrizen...
Host: Matrix A (shape=(5, 4), dtype=float64)
Host: Matrix B (shape=(4, 6), dtype=float64)

Host: Alloziere Speicher auf GPU für Matrizen...
GPU 0: Fordere Allokation von 160 Bytes an...
[C Driver][simulated_kernel_allocate] GPU 0 - Simuliert (malloc) 160 bytes bei 00000081a1fb2620
GPU 0: Allokation erfolgreich, Adresse erhalten: 556768372256
GPU 0: Fordere Allokation von 192 Bytes an...
[C Driver][simulated_kernel_allocate] GPU 0 - Simuliert (malloc) 192 bytes bei 00000081a1fba5b0
GPU 0: Allokation erfolgreich, Adresse erhalten: 556768404912
Host: 'GPU'-Adressen erhalten: A=556768372256, B=556768404912

Host: Schreibe Daten in den GPU-Speicher...
GPU 0: Fordere Schreiben von 160 Bytes an Adresse 556768372256 an...
[C Driver][simulated_kernel_write] Kopiere 160 Bytes von Host-Quelle 00000081a2e951b0 nach Host-Ziel 00000081a1fb2620 (addr 556768372256)
[C Driver]   Host-zu-Host-Kopie (simulated_kernel_write) abgeschlossen.
GPU 0: Schreibanforderung gesendet.
GPU 0: Fordere Schreiben von 192 Bytes an Adresse 556768404912 an...
[C Driver][simulated_kernel_write] Kopiere 192 Bytes von Host-Quelle 00000081a5576f90 nach Host-Ziel 00000081a1fba5b0 (addr 556768404912)
[C Driver]   Host-zu-Host-Kopie (simulated_kernel_write) abgeschlossen.
GPU 0: Schreibanforderung gesendet.
Host: Daten erfolgreich zur GPU geschrieben (simuliert).

Host: Starte Kernel 'matrix_multiply' auf GPU...
GPU 0: Fordere Ausführung von Kernel 'matrix_multiply' an...
[C Driver] simulated_matrix_multiply gestartet für GPU 0.
[C Driver][sim_matmul] Matrix A Shape: (5, 4), Size: 160
[C Driver][sim_matmul] Matrix B Shape: (4, 6), Size: 192
[C Driver][sim_matmul] Ergebnis Matrix C Shape: (5, 6), Size: 240
[C Driver][allocate_gpu_memory] OpenCL Buffer erstellt (handle=00000081a54d2250), Size=160
[C Driver][allocate_gpu_memory] OpenCL Buffer erstellt (handle=00000081a54d2d90), Size=192
[C Driver][allocate_gpu_memory] OpenCL Buffer erstellt (handle=00000081a5512750), Size=240
[C Driver][sim_matmul] Übertrage Daten von Host nach GPU...
[C Driver][write_to_gpu] Starte clEnqueueWriteBuffer: 160 Bytes von Host 00000081a1fb2620 nach GPU 00000081a54d2250
[C Driver][write_to_gpu] Erfolgreich 160 Bytes von Host 00000081a1fb2620 nach GPU Puffer 00000081a54d2250 geschrieben
[C Driver][write_to_gpu] Starte clEnqueueWriteBuffer: 192 Bytes von Host 00000081a1fba5b0 nach GPU 00000081a54d2d90
[C Driver][write_to_gpu] Erfolgreich 192 Bytes von Host 00000081a1fba5b0 nach GPU Puffer 00000081a54d2d90 geschrieben
[C Driver][sim_matmul] Datenübertragung Host->GPU abgeschlossen.
[C Driver][sim_matmul] Sende Matrixmultiplikations-Befehl (Simulation)...
[C Driver][submit_command] Starte CPU-Matrixmultiplikation (Simulation)...
[C Driver][submit_command] CPU-Matrixmultiplikation abgeschlossen.
[C Driver][submit_command] Matrixmultiplikations-Simulation erfolgreich abgeschlossen.
[C Driver][sim_matmul] Matrixmultiplikations-Befehl abgeschlossen (Status: 1).
[C Driver][sim_matmul] Starte asynchronen Readback GPU Puffer C -> Host...
[C Driver]   Host-Ergebnispuffer allokiert bei 00000081a1fbb690
[C Driver]   Non-blocking Read eingereiht (Event: 00000081a46fc2b0). Setze Callback.
[C Driver]   Callback registriert für Event 00000081a46fc2b0.
[C Driver]   Warte auf Beendigung der Command Queue (clFinish)...
[C Driver][Callback GPU 0] Readback event (00000081a46fc2b0) abgeschlossen. Status: CL_SUCCESS (0)
[C Driver][Callback GPU 0] Daten lesen in Host-Puffer abgeschlossen.
[C Driver][Callback GPU 0] Callback beendet.
[C Driver]   Command Queue beendet (Status: CL_SUCCESS).
[C Driver][sim_matmul] Bereinige GPU Puffer...
[C Driver][free_gpu_memory] OpenCL Buffer freigegeben (handle=00000081a54d2250)
[C Driver][free_gpu_memory] OpenCL Buffer freigegeben (handle=00000081a54d2d90)
[C Driver][free_gpu_memory] OpenCL Buffer freigegeben (handle=00000081a5512750)
[C Driver][sim_matmul] Matrixmultiplikation erfolgreich. Gebe Host-Adresse zurück: 556768409232 (00000081a1fbb690)
GPU 0: Kernel 'matrix_multiply' erfolgreich (simuliert), Ergebnisadresse: 556768409232
Host: Kernel beendet. Ergebnis an GPU-Adresse: 556768409232

Host: Lese Ergebnis von GPU zurück zum Host...
GPU 0: Fordere Lesen von 240 Bytes von Adresse 556768409232 an...
[C Driver][simulated_kernel_read] Kopiere 240 Bytes von Host Addr 556768409232 (00000081a1fbb690) nach Python Dest 00000081a48fb4f0
[C Driver]   Gebe Host Ergebnispuffer frei bei 00000081a1fbb690
[C Driver]   Lesen/Kopieren von Host abgeschlossen.
GPU 0: Leseanforderung abgeschlossen.

--- Ergebnis der Matrixmultiplikation (vom simulierten Treiber berechnet/gelesen) ---
Shape: (5, 6), Dtype: float64
[[0.72470531 0.53853531 1.04054047 0.6923406  0.81253766 1.16930091]
 [0.95457385 0.40580491 1.08629638 0.93634127 0.88719397 0.97491993]
 [0.76286833 0.31892414 0.92117509 0.91012526 0.80013112 0.99433272]
 [1.91374926 0.9449003  2.30844999 1.97723782 1.65517081 2.214666  ]
 [1.25484958 0.4630622  1.4020248  1.42356855 1.00587981 1.29352593]]

Host: Verifiziere mit NumPy...
VERIFIKATION ERFOLGREICH: Ergebnis vom Treiber stimmt mit NumPy überein.

Host: Gebe Speicher auf GPU frei...
GPU 0: Fordere Freigabe von 240 Bytes an Adresse 556768409232 an...
[C Driver][simulated_kernel_free] GPU 0 - Simuliert (free) 240 bytes bei 00000081a1fbb690
```

## 📁 Projektstruktur

```plaintext
├── simulated_driver.c         # Simulierter C-Treiber mit OpenCL-Bindung
├── app.py                     # Python Test Harness
├── README.md                  # Diese Datei
├── docs/
│   └── aethercl_architecture.png   # Diagramm (optional auch als GIF)
└── simulated_driver.dll       # Kompilierte Shared Library
```

---

## 🚀 Nächste Schritte (Vision)

- Echte GPU-Matrixkernel mit OpenCL C
- Simuliertes MMIO-Interface mit `mmap`
- Integration eines Memory Schedulers (z. B. für LLM)
- Erweiterung um Softmax, Conv2D, ReLU etc.
- Komplettes Command Dispatch System mit Scheduler-Logik

---

## 🤝 Lizenz

MIT License – Frei verwendbar für Lehre, Forschung, Entwicklung.

---

## 🧠 Inspiration

Dieses Projekt entstand aus der Vision, GPU-Verhalten auf systemnaher Ebene verständlich und kontrolliert zu simulieren – als Lernhilfe, Experimentierplattform und Brückentechnologie für AI-Systeme jenseits etablierter Frameworks wie CUDA oder TensorFlow.

---

## 📷 Beispiel-Visualisierung



---

