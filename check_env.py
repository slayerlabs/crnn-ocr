"""Sprawdza, czy wszystkie zależności projektu są dostępne i w jakiej wersji.

Uruchomienie:  python check_env.py
Kod wyjścia 0 = wszystko OK, 1 = czegoś brakuje (wygodne np. w CI).
"""
import importlib
import sys

sys.stdout.reconfigure(encoding="utf-8")    # polskie znaki w konsoli Windows

# (nazwa modułu do importu, opis, czy wymagany)
DEPENDENCIES = [
    ("torch",        "PyTorch",                      True),
    ("torchvision", "Torchvision (transformacje)",  True),
    ("PIL",          "Pillow (wczytywanie obrazów)", True),
    ("yaml",         "PyYAML (pliki configs/*.yaml)", True),
    ("rapidfuzz",    "rapidfuzz (metryki CER/WER)",  True),
    ("tensorboard",  "TensorBoard (logi treningu)",  True),
    ("pytest",       "pytest (testy jednostkowe)",   False),
    ("ruff",         "ruff (linter)",                False),
]

# nazwa modułu do importu != nazwa pakietu na PyPI
PYPI_NAMES = {
    "PIL": "pillow",
    "yaml": "pyyaml",
}

def main():
    missing = []

    for module, desc, required in DEPENDENCIES:
        try:
            m = importlib.import_module(module)
            version = getattr(m, "__version__", "?")
            print(f"  [OK]      {desc:36s} {module} {version}")
        except ImportError:
            status = "[BRAK]    " if required else "[OPCJONALNIE]"
            print(f"  {status}  {desc:36s} {module}")
            if required:
                missing.append(module)

    # GPU — tylko jeśli torch jest dostępny
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  [OK]      {'CUDA (GPU)':36s} {name}")
        else:
            print(  "  [UWAGA]   CUDA niedostępne — trening pójdzie na CPU")
    except ImportError:
        pass

    print()
    if missing:
        print(f"Brakuje: {', '.join(missing)}")
        hint = " ".join(PYPI_NAMES.get(m, m) for m in missing)
        print(f"Doinstaluj:  pip install {hint}")
        sys.exit(1)
    print("Wszystkie zależności są dostępne. OK")
    sys.exit(0)

if __name__ == "__main__":
    main()
