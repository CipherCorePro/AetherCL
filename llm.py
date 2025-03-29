import ctypes
import os
import platform # Um OS zu erkennen
import numpy as np
import atexit # Für automatischen Cleanup

# --- Laden der simulierten Kerneltreiber-Bibliothek ---
driver_lib_path = ""
if platform.system() == "Windows":
    driver_lib_path = "./simulated_driver.dll"
elif platform.system() == "Linux" or platform.system() == "Darwin": # Darwin ist macOS
    driver_lib_path = "./simulated_driver.so"
else:
    print(f"Unbekanntes Betriebssystem: {platform.system()}. Versuche .so")
    driver_lib_path = "./simulated_driver.so"

try:
    abs_driver_path = os.path.abspath(driver_lib_path)
    print(f"Versuche, simulierten Treiber zu laden von: {abs_driver_path}")
    lib = ctypes.CDLL(abs_driver_path)
    print("Simulierter Treiber erfolgreich geladen.")
except OSError as e:
    print(f"FEHLER: Simulierter Treiber nicht gefunden oder konnte nicht geladen werden.")
    print(f"  Pfad: {abs_driver_path}")
    print(f"  Fehlerdetails: {e}")
    print(f"  Stellen Sie sicher, dass '{os.path.basename(driver_lib_path)}' existiert und korrekt kompiliert wurde.")
    exit()
except Exception as e:
    print(f"Ein unerwarteter Fehler beim Laden des Treibers trat auf: {e}")
    exit()


# --- C-Funktionssignaturen definieren ---
try:
    # int initialize_gpu(int gpu_index) -> int
    lib.initialize_gpu.argtypes = [ctypes.c_int]
    lib.initialize_gpu.restype = ctypes.c_int

    # unsigned long long simulated_kernel_allocate(int gpu_index, size_t size) -> unsigned long long
    lib.simulated_kernel_allocate.argtypes = [ctypes.c_int, ctypes.c_size_t]
    lib.simulated_kernel_allocate.restype = ctypes.c_uint64

    # void simulated_kernel_free(int gpu_index, unsigned long long address, size_t size) -> void
    lib.simulated_kernel_free.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t]
    lib.simulated_kernel_free.restype = None

    # void simulated_kernel_write(int gpu_index, unsigned long long address, size_t size, const void *source) -> void
    lib.simulated_kernel_write.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
    lib.simulated_kernel_write.restype = None

    # void simulated_kernel_read(int gpu_index, unsigned long long address, size_t size, void *destination) -> void
    lib.simulated_kernel_read.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
    lib.simulated_kernel_read.restype = None

    # unsigned long long simulated_matrix_multiply(int gpu_index, unsigned long long address_a, unsigned long long address_b,
    #                                              size_t size_a, size_t size_b, int *shape_a, int *shape_b) -> unsigned long long
    lib.simulated_matrix_multiply.argtypes = [
        ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
    ]
    lib.simulated_matrix_multiply.restype = ctypes.c_uint64

    # void shutdown_driver() -> void
    lib.shutdown_driver.argtypes = None
    lib.shutdown_driver.restype = None

    print("C-Funktionssignaturen erfolgreich definiert.")

    # Registriere die Cleanup-Funktion für das Ende des Skripts
    atexit.register(lib.shutdown_driver)
    print("Treiber-Shutdown-Funktion für atexit registriert.")

except AttributeError as e:
    print(f"FEHLER: Eine erwartete Funktion wurde nicht im Treiber gefunden: {e}")
    print("  Stellen Sie sicher, dass 'simulated_driver.c' korrekt kompiliert wurde und alle Funktionen enthält.")
    exit()
except Exception as e:
    print(f"Ein unerwarteter Fehler beim Definieren der Signaturen trat auf: {e}")
    exit()

# --- Hilfsfunktionen (Platzhalter) ---
def get_num_gpus():
    print("Simuliere GPU-Anzahl (fest auf 1).")
    return 1

def get_num_cus(device_index):
    print(f"Simuliere CU-Anzahl für GPU {device_index} (fest auf 2560).")
    return 2560

# --- Klassen Definitionen ---
class GPU:
    """ Repräsentiert eine einzelne GPU, die über den simulierten Treiber interagiert."""
    def __init__(self, index, num_cus):
        self.index = index
        self.num_cus = num_cus
        self.initialized = False
        print(f"GPU Objekt {self.index} erstellt (CUs: {self.num_cus}).")

    def initialize(self) -> bool:
        """Initialisiert die GPU über den Treiber."""
        if self.initialized:
            print(f"GPU {self.index}: Bereits initialisiert.")
            return True
        print(f"GPU {self.index}: Fordere Initialisierung über Treiber an...")
        try:
            result = lib.initialize_gpu(self.index)
            if result == 1:
                print(f"GPU {self.index}: Treiber-Initialisierung erfolgreich.")
                self.initialized = True
                return True
            else:
                print(f"GPU {self.index}: Treiber-Initialisierung fehlgeschlagen (Code: {result}).")
                self.initialized = False
                return False
        except Exception as e:
            print(f"GPU {self.index}: Fehler beim Aufruf von initialize_gpu: {e}")
            self.initialized = False
            return False

    def allocate(self, size: int) -> int | None:
        """Fordert Speicherallokation über den simulierten Treiber an."""
        if not self.initialized:
            print(f"GPU {self.index}: Fehler: Muss zuerst initialisiert werden (call .initialize()).")
            return None
        print(f"GPU {self.index}: Fordere Allokation von {size} Bytes an...")
        if size <= 0:
             print(f"GPU {self.index}: Fehler: Ungültige Allokationsgröße {size}.")
             return None
        try:
            address = lib.simulated_kernel_allocate(self.index, size)
            if address != 0:
                print(f"GPU {self.index}: Allokation erfolgreich, Adresse erhalten: {address}")
                return address
            else:
                print(f"GPU {self.index}: Allokation fehlgeschlagen (Treiber gab 0 zurück).")
                return None
        except Exception as e:
            print(f"GPU {self.index}: Fehler beim Aufruf von simulated_kernel_allocate: {e}")
            return None

    def free(self, address: int, size: int):
        """Fordert Speicherfreigabe über den simulierten Treiber an."""
        if not self.initialized:
            print(f"GPU {self.index}: Warnung: Freigabe ohne Initialisierung aufgerufen.")
            # Trotzdem versuchen, falls der Treiber es erlaubt
        print(f"GPU {self.index}: Fordere Freigabe von {size} Bytes an Adresse {address} an...")
        if address == 0 or size <= 0:
             print(f"GPU {self.index}: Fehler: Ungültige Adresse ({address}) oder Größe ({size}) für Freigabe.")
             return
        try:
            lib.simulated_kernel_free(self.index, address, size)
            print(f"GPU {self.index}: Freigabeanforderung gesendet (Treiber meldet keine Bestätigung).")
        except Exception as e:
             print(f"GPU {self.index}: Fehler beim Aufruf von simulated_kernel_free: {e}")

    def write_data(self, address: int, source_data: np.ndarray):
        """Schreibt Daten aus einem NumPy Array an eine 'GPU'-Adresse via Treiber."""
        if not self.initialized:
            print(f"GPU {self.index}: Fehler: Muss zuerst initialisiert werden.")
            return False
        print(f"GPU {self.index}: Fordere Schreiben von {source_data.nbytes} Bytes an Adresse {address} an...")
        if address == 0:
             print(f"GPU {self.index}: Fehler: Ungültige Zieladresse 0 für Schreiboperation.")
             return False
        if source_data.nbytes == 0:
             print(f"GPU {self.index}: Warnung: Versuch, 0 Bytes zu schreiben.")
             return True # Nichts zu tun

        # Sicherstellen, dass das Array C-kontinuierlich ist für ctypes
        if not source_data.flags['C_CONTIGUOUS']:
            print(f"GPU {self.index}: Warnung: Quell-Array ist nicht C-kontinuierlich. Erstelle Kopie.")
            source_data = np.ascontiguousarray(source_data)

        try:
            source_ptr = source_data.ctypes.data_as(ctypes.c_void_p)
            size = source_data.nbytes
            lib.simulated_kernel_write(self.index, address, size, source_ptr)
            print(f"GPU {self.index}: Schreibanforderung gesendet.")
            return True
        except Exception as e:
             print(f"GPU {self.index}: Fehler beim Aufruf von simulated_kernel_write: {e}")
             return False

    def read_data(self, address: int, size: int, destination_buffer: np.ndarray):
        """Liest Daten von einer 'GPU'-Adresse in einen NumPy Puffer via Treiber."""
        if not self.initialized:
            print(f"GPU {self.index}: Fehler: Muss zuerst initialisiert werden.")
            return False
        print(f"GPU {self.index}: Fordere Lesen von {size} Bytes von Adresse {address} an...")
        if address == 0 or size <= 0:
            print(f"GPU {self.index}: Fehler: Ungültige Adresse ({address}) oder Größe ({size}) für Leseoperation.")
            return False
        if destination_buffer.nbytes < size:
             print(f"GPU {self.index}: Fehler: Zielpuffer ist zu klein ({destination_buffer.nbytes} Bytes) für angeforderte Daten ({size} Bytes).")
             return False
        # Sicherstellen, dass der Zielpuffer beschreibbar und C-kontinuierlich ist
        if not destination_buffer.flags['WRITEABLE']:
             print(f"GPU {self.index}: Fehler: Zielpuffer ist nicht beschreibbar.")
             return False
        if not destination_buffer.flags['C_CONTIGUOUS']:
             # Für read ist das kritisch, da C direkt hineinschreibt
             print(f"GPU {self.index}: Fehler: Zielpuffer muss C-kontinuierlich sein.")
             return False

        try:
            dest_ptr = destination_buffer.ctypes.data_as(ctypes.c_void_p)
            lib.simulated_kernel_read(self.index, address, size, dest_ptr)
            print(f"GPU {self.index}: Leseanforderung abgeschlossen.")
            return True
        except Exception as e:
            print(f"GPU {self.index}: Fehler beim Aufruf von simulated_kernel_read: {e}")
            return False

    def run_kernel(self, kernel_code: str, *args):
        """Führt einen Kernel über den simulierten Treiber aus."""
        if not self.initialized:
            print(f"GPU {self.index}: Fehler: Muss zuerst initialisiert werden.")
            return None
        print(f"GPU {self.index}: Fordere Ausführung von Kernel '{kernel_code}' an...")
        if kernel_code == "matrix_multiply":
            if len(args) != 6:
                print(f"GPU {self.index}: Fehler: Falsche Argumente für 'matrix_multiply'. Erwartet: addr_a, addr_b, size_a, size_b, shape_a, shape_b")
                return None
            address_a, address_b, size_a, size_b, shape_a, shape_b = args
            if not isinstance(shape_a, (tuple, list)) or len(shape_a) != 2 or \
               not isinstance(shape_b, (tuple, list)) or len(shape_b) != 2:
                 print(f"GPU {self.index}: Fehler: Shapes müssen Tupel/Listen mit 2 Elementen sein.")
                 return None
            try:
                shape_a_arr = (ctypes.c_int * len(shape_a))(*shape_a)
                shape_b_arr = (ctypes.c_int * len(shape_b))(*shape_b)
                shape_a_ptr = ctypes.cast(shape_a_arr, ctypes.POINTER(ctypes.c_int))
                shape_b_ptr = ctypes.cast(shape_b_arr, ctypes.POINTER(ctypes.c_int))

                result_address = lib.simulated_matrix_multiply(
                    self.index, address_a, address_b, size_a, size_b, shape_a_ptr, shape_b_ptr
                )

                if result_address != 0:
                    print(f"GPU {self.index}: Kernel '{kernel_code}' erfolgreich (simuliert), Ergebnisadresse: {result_address}")
                    return result_address
                else:
                    print(f"GPU {self.index}: Kernel '{kernel_code}' fehlgeschlagen (Treiber gab 0 zurück).")
                    return None
            except Exception as e:
                print(f"GPU {self.index}: Fehler beim Aufruf von simulated_matrix_multiply: {e}")
                return None
        else:
            print(f"GPU {self.index}: Fehler: Unbekannter Kernel-Code '{kernel_code}'.")
            return None

class GPUManager:
    """ Verwaltet die verfügbaren GPUs."""
    def __init__(self):
        self.gpus = []
        print("Initialisiere GPUManager...")
        num_gpus = get_num_gpus()
        print(f"Manager: {num_gpus} GPU(s) gefunden (simuliert).")
        for i in range(num_gpus):
            num_cus = get_num_cus(i)
            print(f"Manager: Erstelle GPU-Objekt für Index {i} mit {num_cus} CUs.")
            gpu = GPU(i, num_cus)
            self.gpus.append(gpu)
        print("GPUManager Initialisierung abgeschlossen.")

    def get_gpu(self, index):
        """Gibt das GPU-Objekt für den gegebenen Index zurück."""
        if 0 <= index < len(self.gpus):
            return self.gpus[index]
        else:
            print(f"Fehler: GPU-Index {index} ungültig. Verfügbar: {len(self.gpus)} GPUs.")
            return None

# --- Hauptausführung ---
if __name__ == "__main__":
    print("\n--- Start des Hauptprogramms (mit mmap-basiertem Treiber) ---")
    manager = GPUManager()

    print(f"\nVerfügbare GPUs im Manager: {len(manager.gpus)}")
    if not manager.gpus:
        print("Keine GPUs verfügbar. Programm wird beendet.")
        exit()

    gpu0 = manager.get_gpu(0)

    if gpu0:
        print(f"\n--- Operationen auf GPU {gpu0.index} ---")

        # 1. GPU initialisieren
        if not gpu0.initialize():
             print("GPU-Initialisierung fehlgeschlagen. Programm wird beendet.")
             exit()

        # 2. Host-Daten vorbereiten
        print("\nHost: Erstelle Matrizen...")
        matrix_a = np.random.rand(5, 4).astype(np.float64) # Kleinere Matrizen für übersichtlichere Ausgabe
        matrix_b = np.random.rand(4, 6).astype(np.float64)
        print(f"Host: Matrix A (shape={matrix_a.shape}, dtype={matrix_a.dtype})")
        # print(matrix_a)
        print(f"Host: Matrix B (shape={matrix_b.shape}, dtype={matrix_b.dtype})")
        # print(matrix_b)

        # 3. Speicher auf 'GPU' allozieren
        print("\nHost: Alloziere Speicher auf GPU für Matrizen...")
        gpu_address_a = gpu0.allocate(matrix_a.nbytes)
        gpu_address_b = gpu0.allocate(matrix_b.nbytes)

        if gpu_address_a and gpu_address_b:
            print(f"Host: 'GPU'-Adressen erhalten: A={gpu_address_a}, B={gpu_address_b}")

            # 4. Daten zur 'GPU' schreiben
            print("\nHost: Schreibe Daten in den GPU-Speicher...")
            write_a_ok = gpu0.write_data(gpu_address_a, matrix_a)
            write_b_ok = gpu0.write_data(gpu_address_b, matrix_b)

            if write_a_ok and write_b_ok:
                print("Host: Daten erfolgreich zur GPU geschrieben (simuliert).")

                # 5. Kernel ausführen
                print("\nHost: Starte Kernel 'matrix_multiply' auf GPU...")
                result_address = gpu0.run_kernel(
                    "matrix_multiply",
                    gpu_address_a, gpu_address_b,
                    matrix_a.nbytes, matrix_b.nbytes,
                    matrix_a.shape, matrix_b.shape
                )

                if result_address:
                    print(f"Host: Kernel beendet. Ergebnis an GPU-Adresse: {result_address}")

                    # 6. Ergebnis von 'GPU' lesen
                    print("\nHost: Lese Ergebnis von GPU zurück zum Host...")
                    rows_res, cols_res = matrix_a.shape[0], matrix_b.shape[1]
                    result_shape = (rows_res, cols_res)
                    result_dtype = np.float64 # Muss mit C übereinstimmen
                    result_size = rows_res * cols_res * np.dtype(result_dtype).itemsize

                    # Host-Puffer für das Ergebnis (muss C-kontinuierlich sein)
                    result_matrix_host = np.zeros(result_shape, dtype=result_dtype)

                    read_success = gpu0.read_data(result_address, result_size, result_matrix_host)

                    if read_success:
                        print("\n--- Ergebnis der Matrixmultiplikation (vom simulierten Treiber berechnet/gelesen) ---")
                        print(f"Shape: {result_matrix_host.shape}, Dtype: {result_matrix_host.dtype}")
                        print(result_matrix_host)

                        # --- Verifikation (optional) ---
                        print("\nHost: Verifiziere mit NumPy...")
                        host_result_np = np.matmul(matrix_a, matrix_b)
                        if np.allclose(result_matrix_host, host_result_np):
                             print("VERIFIKATION ERFOLGREICH: Ergebnis vom Treiber stimmt mit NumPy überein.")
                        else:
                             print("VERIFIKATION FEHLGESCHLAGEN: Ergebnis vom Treiber weicht von NumPy ab!")
                             # print("NumPy-Ergebnis:")
                             # print(host_result_np)
                    else:
                        print("Host: Fehler beim Lesen des Ergebnisses von der GPU.")

                    # 7. Speicher freigeben
                    print("\nHost: Gebe Speicher auf GPU frei...")
                    gpu0.free(result_address, result_size)
                    gpu0.free(gpu_address_b, matrix_b.nbytes)
                    gpu0.free(gpu_address_a, matrix_a.nbytes)

                else:
                    print("Host: Kernel-Ausführung fehlgeschlagen.")
            else:
                print("Host: Fehler beim Schreiben der Daten zur GPU.")
        else:
            print("Host: Fehler beim Allozieren des Speichers auf der GPU.")
    else:
        print("Konnte GPU 0 nicht abrufen.")

    # Der Cleanup wird automatisch durch atexit aufgerufen
    print("\n--- Programmende (Treiber-Cleanup via atexit) ---")
