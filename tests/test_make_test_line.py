"""Testy font_candidates/find_font: ścieżki z WINDIR/LOCALAPPDATA, nie sztywny dysk."""
from pathlib import Path

import pytest

from make_test_line import find_font, font_candidates


def test_candidates_windows_dirs_from_env(monkeypatch):
    """Windows: Fonts z WINDIR (dowolna litera dysku) + per-user z LOCALAPPDATA."""
    monkeypatch.setenv("WINDIR", "X:/Windows")          # celowo inny dysk niż C:
    monkeypatch.setenv("LOCALAPPDATA", "X:/Users/a/AppData/Local")
    c = list(font_candidates())
    assert c[0] == Path("X:/Windows") / "Fonts" / "arial.ttf"
    assert c[1] == (Path("X:/Users/a/AppData/Local") / "Microsoft"
                    / "Windows" / "Fonts" / "arial.ttf")


def test_candidates_no_localappdata_limited_to_windir(monkeypatch):
    """Bez LOCALAPPDATA nie ma ścieżki per-user, WINDIR wystarcza."""
    monkeypatch.setenv("WINDIR", "X:/Windows")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    c = list(font_candidates())
    assert c[0] == Path("X:/Windows") / "Fonts" / "arial.ttf"
    assert len(c) == 4


def test_candidates_without_windir_starts_from_bare_name(monkeypatch):
    """Linux/CI (brak WINDIR): żadnych ścieżek Windows, od razu nazwa + DejaVu."""
    monkeypatch.delenv("WINDIR", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    c = list(font_candidates())
    assert c[0] == "arial.ttf"
    assert "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" in c
    assert "/usr/share/fonts/TTF/DejaVuSans.ttf" in c
    assert not any("Fonts" in str(p) for p in c)


def test_find_font_prefers_windir_fonts(monkeypatch, tmp_path):
    """Pierwszy istniejący plik wygrywa: WINDIR/Fonts/arial.ttf."""
    fonts = tmp_path / "Fonts"
    fonts.mkdir()
    (fonts / "arial.ttf").write_bytes(b"")   # find_font sprawdza tylko is_file
    monkeypatch.setenv("WINDIR", str(tmp_path))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert find_font() == fonts / "arial.ttf"


def test_find_font_falls_back_to_localappdata(monkeypatch, tmp_path):
    """WINDIR pusty → font per-user (Windows 10/11: instalacja przez Ustawienia)."""
    windir = tmp_path / "sys"
    windir.mkdir()
    local = tmp_path / "local" / "Microsoft" / "Windows" / "Fonts"
    local.mkdir(parents=True)
    (local / "arial.ttf").write_bytes(b"")
    monkeypatch.setenv("WINDIR", str(windir))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.chdir(tmp_path)              # "arial.ttf" (CWD) nie istnieje
    assert find_font() == local / "arial.ttf"


def test_find_font_exits_when_nothing_matches(monkeypatch, tmp_path):
    """Zero kandydatów na dysku → SystemExit z podpowiedzią --font."""
    monkeypatch.setenv("WINDIR", str(tmp_path))       # pusty — bez fontów
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        find_font()
    assert "--font" in str(exc.value)
