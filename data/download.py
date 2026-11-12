"""Auto-pobranie datasetu ocr-pl-lines z HF, gdy data/raw/{train,val} są puste.

Idempotentne: przy istniejących danych nic nie pobiera. Archiwum tar.gz
weryfikowane SHA-256 z configu (wersja v1 przypięta hashem; zmiana datasetu
= zmiana dataset_url + dataset_sha256 w configs/default.yaml).
"""
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

CHUNK = 1 << 20          # 1 MB na odczyt
PROGRESS_EVERY = 32       # co ile MB linia postępu


def has_pairs(directory) -> bool:
    """Czy katalog zawiera choć jedną parę {base}.png + {base}.txt?"""
    directory = Path(directory)
    if not directory.is_dir():
        return False
    for png in directory.glob("*.png"):
        if png.with_suffix(".txt").exists():
            return True
    return False


def _download(url: str, dest: Path, sha256: str):
    """Ściąga url strumieniowo do dest, licząc po drodze SHA-256."""
    h = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as f:
            while chunk := r.read(CHUNK):
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if done // CHUNK % PROGRESS_EVERY == 0:
                    print(f"  ... {done // (1 << 20)} MB", flush=True)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    if h.hexdigest() != sha256:
        dest.unlink(missing_ok=True)
        sys.exit("[ERROR] SHA-256 pobranego archiwum ≠ data.dataset_sha256 "
                 "z configu — dataset na HF zmienił się bez zmiany configu?")


def ensure_dataset(train_dir, val_dir, url=None, sha256=None) -> bool:
    """Gwarantuje pary png+txt w train_dir i val_dir; True = właśnie pobrało.

    url=None przy braku danych = twardy błąd (jak dotąd).
    """
    train_dir, val_dir = Path(train_dir), Path(val_dir)
    if has_pairs(train_dir) and has_pairs(val_dir):
        return False

    if not url:
        sys.exit(f"[ERROR] Brak danych lokalnie ({train_dir} / {val_dir}) "
                 "i brak data.dataset_url w configu — patrz README, sekcja Dane")

    raw_dir = train_dir.parent
    print(f"Dane:      brak w {raw_dir} — pobieram {url}", flush=True)
    if sha256 is None:
        sys.exit("[ERROR] data.dataset_sha256 puste przy ustawionym dataset_url "
                 "(bez weryfikacji integrity nie pobieramy)")
    raw_dir.mkdir(parents=True, exist_ok=True)

    archive = raw_dir / "ocr-pl-lines.tar.gz.part"
    _download(url, archive, sha256)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(raw_dir, filter="data")   # train/ i val/ do data/raw
    finally:
        archive.unlink(missing_ok=True)

    if not (has_pairs(train_dir) and has_pairs(val_dir)):
        sys.exit(f"[ERROR] Archiwum rozpakowane, ale {train_dir} / {val_dir} "
                 "nadal bez par png+txt — sprawdź układ plików w tar")
    print(f"[OK] Dane pobrane do {raw_dir}", flush=True)
    return True
