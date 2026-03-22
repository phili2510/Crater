#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Krater-Messdaten Plot-Generator
Erstellt flexible Plots aus CSV-Daten mit kombinierbaren Parametern
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# KONFIGURATION
# ============================================================================

# Dateipfade
CSV_PATH = r'/pfad/zur/Results.csv'
OUTPUT_PATH = r'/pfad/zum/output/'

# X-Achse: Zwei Parameter + Operation
X_PARAM_1 = 'volume_ml'  # Erster Parameter für X-Achse
X_PARAM_2 = '1'  # Zweiter Parameter für X-Achse ('1' = nicht verwendet)
X_OPERATION = '/'  # '/' oder '*'

# Y-Achse: Zwei Parameter + Operation
Y_PARAM_1 = 'd_peak'  # Erster Parameter für Y-Achse
Y_PARAM_2 = 'h'  # Zweiter Parameter für Y-Achse ('1' = nicht verwendet)
Y_OPERATION = '/'  # '/' oder '*'

# Filterung: Welche Parameter fixiert werden sollen (dict, leer = kein Filter)
FIXED_PARAMS = {
    'grain_size': '300-400',
    'fill_height_cm': '40.0'
}

# Trendlinie
SHOW_TRENDLINE = True  # Trendlinie anzeigen?
TRENDLINE_THROUGH_ORIGIN = True  # Trendlinie durch (0,0) erzwingen?

# Plot-Einstellungen
FIGSIZE = (6, 4)
MARKERSIZE = 8
DPI = 300
FONTSIZE_LABELS = 12

# ============================================================================
# PARAMETER-LABELS
# ============================================================================

LABELS = {
    'grain_size': 'Grain size [µm]',
    'volume_ml': 'Volume [ml]',
    'fill_height_cm': 'Falling height [cm]',
    'h': 'h [cm]',
    'd_outer': 'd_outer [cm]',
    'r_inner': 'r_inner [cm]',
    'd_fit': 'd_fit [cm]',
    'd_peak': 'd [cm]',
    'sigma_rim': 'σ [cm]',
    'A_rim': 'A_rim [cm]',
}


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def parse_grain_size(grain_str):
    """
    Berechnet Mittelpunkt und halbe Breite aus grain_size Range-String.

    Args:
        grain_str: String wie "300-400"

    Returns:
        (midpoint, half_width) oder (None, None) bei Fehler
    """
    try:
        parts = grain_str.split('-')
        if len(parts) == 2:
            lower = float(parts[0])
            upper = float(parts[1])
            midpoint = (lower + upper) / 2.0
            half_width = (upper - lower) / 2.0
            return midpoint, half_width
        else:
            return None, None
    except:
        return None, None


def get_parameter_value(row, param_name):
    """
    Extrahiert Parameterwert aus CSV-Zeile.

    Args:
        row: Dictionary einer CSV-Zeile
        param_name: Name des Parameters

    Returns:
        Wert als float, oder None bei Fehler
    """
    # Berechnete Spalte: d_peak
    if param_name == 'd_peak':
        try:
            d_fit = float(row['d_fit'])
            return 2.0 * d_fit
        except:
            return None

    # grain_size: Mittelpunkt berechnen
    if param_name == 'grain_size':
        midpoint, _ = parse_grain_size(row.get('grain_size', ''))
        return midpoint

    # Standard-Spalten
    try:
        value_str = row.get(param_name, '').strip()
        if value_str == '':
            return None
        return float(value_str)
    except:
        return None


def calculate_axis_value(row, param_1, param_2, operation):
    """
    Berechnet Achsenwert aus zwei Parametern und Operation.

    Args:
        row: CSV-Zeile als Dictionary
        param_1: Erster Parameter
        param_2: Zweiter Parameter ('1' = nicht verwendet)
        operation: '/' oder '*'

    Returns:
        Berechneter Wert oder None bei Fehler
    """
    # Einzelner Parameter
    if param_2 == '1':
        return get_parameter_value(row, param_1)

    # Zwei Parameter kombinieren
    val_1 = get_parameter_value(row, param_1)
    val_2 = get_parameter_value(row, param_2)

    if val_1 is None or val_2 is None:
        return None

    if operation == '/':
        if val_2 == 0:
            return None
        return val_1 / val_2
    elif operation == '*':
        return val_1 * val_2
    else:
        return None


def generate_axis_label(param_1, param_2, operation):
    """
    Generiert Achsenbeschriftung aus Parametern und Operation.

    Args:
        param_1: Erster Parameter
        param_2: Zweiter Parameter ('1' = nicht verwendet)
        operation: '/' oder '*'

    Returns:
        Formatiertes Label-String
    """
    label_1 = LABELS.get(param_1, param_1)

    if param_2 == '1':
        return label_1

    label_2 = LABELS.get(param_2, param_2)

    if operation == '/':
        return f"{label_1} / {label_2}"
    elif operation == '*':
        return f"{label_1} · {label_2}"
    else:
        return f"{label_1} {operation} {label_2}"


def generate_filename(y_param_1, y_param_2, y_op, x_param_1, x_param_2, x_op):
    """
    Generiert sinnvollen Dateinamen aus Parametern.

    Returns:
        Dateiname als String (ohne Pfad)
    """
    # Y-Achse
    if y_param_2 == '1':
        y_part = y_param_1
    else:
        op_str = 'over' if y_op == '/' else 'times'
        y_part = f"{y_param_1}_{op_str}_{y_param_2}"

    # X-Achse
    if x_param_2 == '1':
        x_part = x_param_1
    else:
        op_str = 'over' if x_op == '/' else 'times'
        x_part = f"{x_param_1}_{op_str}_{x_param_2}"

    return f"{y_part}_vs_{x_part}.png"


def apply_filters(rows, fixed_params):
    """
    Filtert Zeilen basierend auf FIXED_PARAMS.

    Args:
        rows: Liste von Dictionaries (CSV-Zeilen)
        fixed_params: Dictionary mit zu fixierenden Parametern

    Returns:
        Gefilterte Liste von Zeilen
    """
    if not fixed_params:
        return rows

    filtered = []
    for row in rows:
        match = True
        for param, value in fixed_params.items():
            if row.get(param, '').strip() != str(value).strip():
                match = False
                break
        if match:
            filtered.append(row)

    return filtered


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def main():
    print("=" * 70)
    print("Krater-Messdaten Plot-Generator")
    print("=" * 70)

    # CSV einlesen
    print(f"\n[1] Lade CSV-Datei: {CSV_PATH}")
    csv_path = Path(CSV_PATH)

    if not csv_path.exists():
        print(f"FEHLER: CSV-Datei nicht gefunden: {CSV_PATH}")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"    → {len(all_rows)} Zeilen geladen")

    # Filterung anwenden
    print(f"\n[2] Wende Filter an: {FIXED_PARAMS}")
    filtered_rows = apply_filters(all_rows, FIXED_PARAMS)
    print(f"    → {len(filtered_rows)} Zeilen nach Filterung")

    if len(filtered_rows) == 0:
        print("FEHLER: Keine Zeilen nach Filterung übrig!")
        return

    # Datenpunkte berechnen
    print(f"\n[3] Berechne Achsenwerte")
    print(f"    X-Achse: {X_PARAM_1} {X_OPERATION} {X_PARAM_2}")
    print(f"    Y-Achse: {Y_PARAM_1} {Y_OPERATION} {Y_PARAM_2}")

    x_values = []
    y_values = []
    x_errors = []  # Für grain_size Fehlerbalken
    skipped = 0

    for i, row in enumerate(filtered_rows):
        x_val = calculate_axis_value(row, X_PARAM_1, X_PARAM_2, X_OPERATION)
        y_val = calculate_axis_value(row, Y_PARAM_1, Y_PARAM_2, Y_OPERATION)

        if x_val is None or y_val is None:
            skipped += 1
            continue

        x_values.append(x_val)
        y_values.append(y_val)

        # Fehlerbalken für grain_size (nur wenn allein auf X-Achse)
        if X_PARAM_1 == 'grain_size' and X_PARAM_2 == '1':
            _, half_width = parse_grain_size(row.get('grain_size', ''))
            x_errors.append(half_width if half_width else 0)

    print(f"    → {len(x_values)} gültige Datenpunkte")
    print(f"    → {skipped} Zeilen übersprungen (fehlende/ungültige Werte)")

    if len(x_values) == 0:
        print("FEHLER: Keine gültigen Datenpunkte!")
        return

    # Sortiere nach X-Wert
    sorted_indices = np.argsort(x_values)
    x_values = np.array(x_values)[sorted_indices]
    y_values = np.array(y_values)[sorted_indices]
    if x_errors:
        x_errors = np.array(x_errors)[sorted_indices]

    # Plot erstellen
    print(f"\n[4] Erstelle Plot")

    # Schriftart setzen
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Fehlerbalken oder Scatter-Plot
    use_errorbar = (X_PARAM_1 == 'grain_size' and X_PARAM_2 == '1')

    if use_errorbar:
        print("    → Verwende Fehlerbalken für grain_size")
        ax.errorbar(x_values, y_values, xerr=x_errors, fmt='o',
                    markersize=MARKERSIZE, capsize=5, capthick=2,
                    label='Messdaten')
    else:
        ax.scatter(x_values, y_values, marker='o', s=MARKERSIZE ** 2,
                   label='Messdaten')

    # Trendlinie
    if SHOW_TRENDLINE and len(x_values) > 1:
        print(f"    → Berechne Trendlinie (durch Ursprung: {TRENDLINE_THROUGH_ORIGIN})")

        if TRENDLINE_THROUGH_ORIGIN:
            # Steigung durch Ursprung: m = Σ(xy) / Σ(x²)
            slope = np.sum(x_values * y_values) / np.sum(x_values ** 2)
            ax.axline((0, 0), slope=slope, color='red', linestyle='--',
                      linewidth=1.5, label=f'Trendlinie (m={slope:.3f})')
            print(f"       Steigung: {slope:.4f}")
        else:
            # Lineare Regression
            coeffs = np.polyfit(x_values, y_values, 1)
            slope, intercept = coeffs
            x_trend = np.array([x_values.min(), x_values.max()])
            y_trend = slope * x_trend + intercept
            ax.plot(x_trend, y_trend, color='red', linestyle='--',
                    linewidth=1.5, label=f'Trendlinie (m={slope:.3f}, b={intercept:.3f})')
            print(f"       Steigung: {slope:.4f}, Achsenabschnitt: {intercept:.4f}")

    # Achsenbeschriftungen
    x_label = generate_axis_label(X_PARAM_1, X_PARAM_2, X_OPERATION)
    y_label = generate_axis_label(Y_PARAM_1, Y_PARAM_2, Y_OPERATION)

    ax.set_xlabel(x_label, fontsize=FONTSIZE_LABELS)
    ax.set_ylabel(y_label, fontsize=FONTSIZE_LABELS)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legende (falls Trendlinie vorhanden)
    if SHOW_TRENDLINE and len(x_values) > 1:
        ax.legend(fontsize=10)

    # Layout optimieren
    plt.tight_layout()

    # Speichern
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = generate_filename(Y_PARAM_1, Y_PARAM_2, Y_OPERATION,
                                 X_PARAM_1, X_PARAM_2, X_OPERATION)
    output_file = output_dir / filename

    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"\n[5] Plot gespeichert: {output_file}")

    # Statistik ausgeben
    print(f"\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)
    print(f"Datenpunkte:        {len(x_values)}")
    print(f"X-Bereich:          {x_values.min():.3f} - {x_values.max():.3f}")
    print(f"Y-Bereich:          {y_values.min():.3f} - {y_values.max():.3f}")
    print(f"Ausgabedatei:       {filename}")
    print("=" * 70)

    plt.show()


if __name__ == '__main__':
    main()
