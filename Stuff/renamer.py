"""
Ordner-Umbenenner mit Pattern-Matching
=======================================
Benennt Ordner nach einem definierten Muster um oder macht die Änderung rückgängig.

Usage:
    python rename_folders.py
"""

from pathlib import Path
import sys
import re

# KONFIGURATION
INPUT_FOLDER = r'/Users/philippadelbrecht/PycharmProjects/Crater/Evaluation+plotting/Volume/Volume2'

# Muster-Definition (verwende {} als Platzhalter für variable Teile)
PATTERN_OLD = '{}_{}_{}'  # z.B. 600-800_400ml_measurement_results_data
PATTERN_NEW = ('300-400_{}_{}_{}')  # z.B. 600-800_400ml_measurement_40cm_results_data

# Modus
MODE = 'RENAME'  # 'RENAME' = Umbenennen, 'REVERT' = Rückgängig machen
DRY_RUN = False   # True = Nur anzeigen, False = Wirklich ausführen


def pattern_to_regex(pattern):
    """
    Wandelt Pattern mit {} Platzhaltern in Regex um.

    Args:
        pattern: String mit {} als Platzhalter

    Returns:
        Regex-Pattern und Anzahl der Platzhalter
    """
    # Escape special regex characters außer {}
    escaped = re.escape(pattern)
    # Ersetze escaped \{ und \} durch capture groups
    regex_pattern = escaped.replace(r'\{', r'(').replace(r'\}', r'[^_]+)')
    # Zähle Platzhalter
    num_placeholders = pattern.count('{}')

    return regex_pattern, num_placeholders


def extract_parts(name, pattern):
    """
    Extrahiert die variablen Teile aus einem Namen basierend auf dem Pattern.

    Args:
        name: Ordnername
        pattern: Pattern-String mit {} Platzhaltern

    Returns:
        Liste der extrahierten Teile oder None wenn nicht matched
    """
    regex_pattern, num_placeholders = pattern_to_regex(pattern)

    match = re.match(f'^{regex_pattern}$', name)
    if match:
        return list(match.groups())
    return None


def build_name_from_pattern(parts, pattern):
    """
    Baut einen Namen aus Teilen und Pattern zusammen.

    Args:
        parts: Liste der variablen Teile
        pattern: Pattern-String mit {} Platzhaltern

    Returns:
        Zusammengebauter Name
    """
    result = pattern
    for part in parts:
        result = result.replace('{}', part, 1)
    return result


def rename_folders(input_path, pattern_old, pattern_new, mode='RENAME', dry_run=True):
    """
    Benennt Ordner nach Pattern um oder macht Umbenennung rückgängig.

    Args:
        input_path: Path-Objekt des Input-Ordners
        pattern_old: Pattern des alten Namens
        pattern_new: Pattern des neuen Namens
        mode: 'RENAME' oder 'REVERT'
        dry_run: Wenn True, nur anzeigen ohne umbenennen
    """
    if not input_path.exists():
        print(f"[ERROR] Pfad nicht gefunden: {input_path}")
        return

    if not input_path.is_dir():
        print(f"[ERROR] Pfad ist kein Ordner: {input_path}")
        return

    # Im REVERT-Modus Pattern vertauschen
    if mode == 'REVERT':
        pattern_old, pattern_new = pattern_new, pattern_old
        mode_text = 'RÜCKGÄNGIG MACHEN'
    else:
        mode_text = 'UMBENENNEN'

    # Finde alle Unterordner
    folders = [f for f in input_path.iterdir() if f.is_dir()]

    if len(folders) == 0:
        print(f"[WARNING] Keine Ordner gefunden in: {input_path}")
        return

    print(f"\n{'=' * 70}")
    print(f"ORDNER-UMBENENNUNG: {mode_text}")
    print(f"{'=' * 70}")
    print(f"Input-Ordner: {input_path}")
    print(f"Pattern Alt:  {pattern_old}")
    print(f"Pattern Neu:  {pattern_new}")
    print(f"Modus: {'DRY RUN (keine Änderungen)' if dry_run else 'LIVE (Ordner werden umbenannt)'}")
    print(f"Gefundene Ordner: {len(folders)}")
    print(f"{'=' * 70}\n")

    renamed_count = 0
    skipped_count = 0
    error_count = 0

    for folder in sorted(folders):
        old_name = folder.name

        # Extrahiere Teile aus altem Namen
        parts = extract_parts(old_name, pattern_old)

        if parts is None:
            print(f"[SKIP] {old_name}")
            print(f"       → Passt nicht zum Pattern: {pattern_old}")
            skipped_count += 1
            print()
            continue

        # Baue neuen Namen
        try:
            new_name = build_name_from_pattern(parts, pattern_new)
        except Exception as e:
            print(f"[ERROR] {old_name}")
            print(f"        → Fehler beim Erstellen des neuen Namens: {e}")
            error_count += 1
            print()
            continue

        # Prüfe ob Name sich ändert
        if old_name == new_name:
            print(f"[SKIP] {old_name}")
            print(f"       → Keine Änderung nötig")
            skipped_count += 1
            print()
            continue

        new_path = folder.parent / new_name

        # Prüfe ob Zielname bereits existiert
        if new_path.exists():
            print(f"[ERROR] {old_name}")
            print(f"        → {new_name}")
            print(f"        → Zielordner existiert bereits!")
            error_count += 1
            print()
            continue

        # Zeige Umbenennung an
        print(f"[{'DRY' if dry_run else 'OK'}] {old_name}")
        print(f"      → {new_name}")

        # Zeige extrahierte Teile (Debug)
        print(f"      Teile: {parts}")

        # Führe Umbenennung durch (wenn nicht dry_run)
        if not dry_run:
            try:
                folder.rename(new_path)
                renamed_count += 1
            except Exception as e:
                print(f"      → FEHLER: {e}")
                error_count += 1
        else:
            renamed_count += 1

        print()

    # Zusammenfassung
    print(f"{'=' * 70}")
    print(f"ZUSAMMENFASSUNG")
    print(f"{'=' * 70}")
    print(f"Gesamt: {len(folders)} Ordner")
    print(f"{'Würden umbenannt werden' if dry_run else 'Umbenannt'}: {renamed_count}")
    print(f"Übersprungen: {skipped_count}")
    print(f"Fehler: {error_count}")
    print(f"{'=' * 70}\n")

    if dry_run and renamed_count > 0:
        print(f"[INFO] Dies war ein DRY RUN - keine Änderungen vorgenommen.")
        print(f"[INFO] Setze DRY_RUN = False um die Umbenennung durchzuführen.\n")


def main():
    input_path = Path(INPUT_FOLDER)

    print("\n" + "=" * 70)
    print("ORDNER-UMBENENNER MIT PATTERN-MATCHING")
    print("=" * 70 + "\n")

    # Validierung
    if MODE not in ['RENAME', 'REVERT']:
        print(f"[ERROR] Ungültiger MODE: {MODE}")
        print(f"        Erlaubt: 'RENAME' oder 'REVERT'")
        sys.exit(1)

    rename_folders(input_path, PATTERN_OLD, PATTERN_NEW, mode=MODE, dry_run=DRY_RUN)

    print("FERTIG!\n")


if __name__ == "__main__":
    main()
