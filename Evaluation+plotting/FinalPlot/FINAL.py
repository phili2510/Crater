#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatisierte Plot-Erstellung aus Krater-Messdaten (CSV)
Variiert einen Versuchsparameter und plottet eine Krater-Kenngröße
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ============================================================================
# KONFIGURATION
# ============================================================================

CSV_PATH = r'/Users/philippadelbrecht/PycharmProjects/Crater/Evaluation+plotting/Results.csv'
VARY_PARAMETER = 'fill_height_cm'  # 'grain_size', 'volume_ml', 'fill_height_cm'
CRATER_METRIC = 'w_over_h'  # 'h', 'd_outer', 'r_inner', 'd_fit', 'sigma_rim', 'A_rim, 'w_over_h'
OUTPUT_PATH = r'/Users/philippadelbrecht/PycharmProjects/Crater/Evaluation+plotting/FinalPlot/Plots'

# ============================================================================
# LABEL-DEFINITIONEN
# ============================================================================

PARAM_LABELS = {
    'grain_size': 'Grain size [µm]',
    'volume_ml': 'Sand Volume [ml]',
    'fill_height_cm': 'Falling height [cm]'
}

METRIC_LABELS = {
    'h': 'Rim-height h [cm]',
    'd_outer': 'Außendurchmesser d [cm]',
    'r_inner': 'r [cm]',
    'd_fit': 'Halbabstand d_fit [cm]',
    'd_peak_distance': 'd [cm]',  # NEU
    'sigma_rim': 'Gauß-Breite σ [cm]',
    'A_rim': 'Gauß-Amplitude A_rim [cm]',
    'w_over_h': 'w/h',  # NEU
}


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def parse_grain_size(grain_str):
    """
    Parst Korngröße-String (z.B. "300-400") in (min, max, center, half_range)
    """
    parts = grain_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Ungültiges Korngröße-Format: {grain_str}")

    min_val = float(parts[0])
    max_val = float(parts[1])
    center = (min_val + max_val) / 2
    half_range = (max_val - min_val) / 2

    return min_val, max_val, center, half_range


def load_csv_data(csv_path):
    """
    Lädt CSV-Datei und gibt Liste von Dictionaries zurück
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    print(f"✓ {len(data)} Zeilen aus CSV geladen")
    return data


def get_fixed_parameters(vary_param):
    """
    Gibt die zwei Parameter zurück, die NICHT variiert werden
    """
    all_params = ['grain_size', 'volume_ml', 'fill_height_cm']
    return [p for p in all_params if p != vary_param]


def filter_by_most_common_group(data, fixed_params):
    """
    Filtert Daten nach der häufigsten Kombination der fixierten Parameter
    Gibt gefilterte Daten und die fixierten Werte zurück
    """
    # Gruppiere nach fixierten Parametern
    groups = defaultdict(list)

    for row in data:
        # Erstelle Gruppenschlüssel
        key = tuple(row[param] for param in fixed_params)
        groups[key].append(row)

    # Finde größte Gruppe
    largest_group_key = max(groups.keys(), key=lambda k: len(groups[k]))
    largest_group = groups[largest_group_key]

    # Erstelle Dictionary mit fixierten Werten
    fixed_values = dict(zip(fixed_params, largest_group_key))

    print(f"\n✓ Fixierte Parameter:")
    for param, value in fixed_values.items():
        label = PARAM_LABELS[param].split('[')[0].strip()
        if param == 'grain_size':
            print(f"  - {label}: {value} µm")
        elif param == 'volume_ml':
            print(f"  - {label}: {value} ml")
        elif param == 'fill_height_cm':
            print(f"  - {label}: {value} cm")

    print(f"✓ {len(largest_group)} Zeilen nach Filterung")

    return largest_group, fixed_values


def extract_plot_data(data, vary_param, metric):
    """
    Extrahiert X- und Y-Werte für den Plot
    Behandelt fehlende Werte und sortiert nach X
    """
    x_values = []
    y_values = []
    x_errors = []  # Nur für grain_size relevant

    skipped = 0

    for row in data:
        # Prüfe ob Y-Wert vorhanden
        # Prüfe ob Y-Wert vorhanden
        # Prüfe ob Y-Wert vorhanden
        if metric == 'd_peak_distance':
            d_fit_str = row['d_fit'].strip()
            if not d_fit_str:
                skipped += 1
                continue
            try:
                y_val = 2 * float(d_fit_str)
            except ValueError:
                skipped += 1
                continue
        elif metric == 'w_over_h':
            r_inner_str = row['r_inner'].strip()
            h_str = row['h'].strip()
            if not r_inner_str or not h_str:
                skipped += 1
                continue
            try:
                h_val = float(h_str)
                if h_val == 0:
                    skipped += 1
                    continue
                y_val = (float(r_inner_str)) / h_val
            except (ValueError, ZeroDivisionError):
                skipped += 1
                continue
        else:
            y_str = row[metric].strip()
            if not y_str:
                skipped += 1
                continue
            try:
                y_val = float(y_str)
            except ValueError:
                skipped += 1
                continue

        # Extrahiere X-Wert
        if vary_param == 'grain_size':
            try:
                _, _, center, half_range = parse_grain_size(row[vary_param])
                x_values.append(center)
                x_errors.append(half_range)
                y_values.append(y_val)
            except ValueError as e:
                print(f"  Warnung: {e}")
                skipped += 1
        else:
            x_str = row[vary_param].strip()
            if not x_str:
                skipped += 1
                continue
            try:
                x_val = float(x_str)
                x_values.append(x_val)
                y_values.append(y_val)
            except ValueError:
                skipped += 1

    if skipped > 0:
        print(f"⚠ {skipped} Zeilen übersprungen (fehlende/ungültige Werte)")

    print(f"✓ {len(x_values)} Datenpunkte für Plot vorbereitet")

    # Sortiere nach X-Wert
    if vary_param == 'grain_size':
        sorted_indices = np.argsort(x_values)
        x_values = [x_values[i] for i in sorted_indices]
        y_values = [y_values[i] for i in sorted_indices]
        x_errors = [x_errors[i] for i in sorted_indices]
        return np.array(x_values), np.array(y_values), np.array(x_errors)
    else:
        sorted_indices = np.argsort(x_values)
        x_values = [x_values[i] for i in sorted_indices]
        y_values = [y_values[i] for i in sorted_indices]
        return np.array(x_values), np.array(y_values), None


def create_subtitle(fixed_values):
    """
    Erstellt Untertitel-String mit fixierten Parametern
    """
    parts = []
    for param, value in fixed_values.items():
        if param == 'grain_size':
            parts.append(f"Korngröße: {value} µm")
        elif param == 'volume_ml':
            parts.append(f"Sandvolumen: {value} ml")
        elif param == 'fill_height_cm':
            parts.append(f"Füllhöhe: {value} cm")

    return ", ".join(parts)


def create_plot(x_data, y_data, x_errors, vary_param, metric, fixed_values, output_path):
    """
    Erstellt und speichert den Plot
    """
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig, ax = plt.subplots(figsize=(6, 4))
    #slope = np.sum(x_data * y_data) / np.sum(x_data ** 2)
    #x_trend = np.linspace(0, x_data.max() * 1.1, 100)
    #trend_line = slope * x_trend
    #ax.plot(x_trend, trend_line, '--', color='red', linewidth=1.5, label = f'Fit ( y = {np.round(slope,2)}x)')
    #plt.ylim([min(y_data)-5, max(y_data)+5])
    #plt.xlim([min(x_data)-2, max(x_data)+2])

    # Plot erstellen
    if vary_param == 'grain_size':
        # Mit horizontalen Fehlerbalken
        ax.errorbar(x_data, y_data, xerr=x_errors,
                    fmt='o', markersize=8, capsize=5, capthick=2)
    else:
        # Einfacher Line-Plot
        ax.scatter(x_data, y_data, marker='o')


    # Achsenbeschriftungen
    ax.set_xlabel(PARAM_LABELS[vary_param], fontsize=12)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    #ax.legend(fontsize=12)

    # Layout optimieren
    plt.tight_layout()

    # Speichern
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{metric}_vs_{vary_param}.png"
    output_file = output_dir / filename

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot gespeichert: {output_file}")

    plt.close()


# ============================================================================
# MULTI-PLOT FUNKTIONEN
# ============================================================================

def create_multi_plot_grid(data, output_path):
    """
    Erstellt ein Grid mit mehreren Subplots:
    - Spalte 1: volume_ml variiert
    - Spalte 2: grain_size variiert
    - Spalte 3: fill_height_cm variiert
    - Zeilen: Alle Krater-Kenngrößen (h, d_outer, r_inner, d_fit, sigma_rim, A_rim)
    """
    # Definiere die Krater-Metriken (Y-Achsen)
    metrics = ['h', 'd_peak_distance', 'r_inner', 'd_fit', 'sigma_rim', 'A_rim']

    # Definiere die zu variierenden Parameter (X-Achsen)
    vary_params = ['volume_ml', 'grain_size', 'fill_height_cm']

    # Erstelle Figure mit Subplots (6 Zeilen x 3 Spalten)
    fig, axes = plt.subplots(len(metrics), len(vary_params),
                             figsize=(24, 20))

    print("\n" + "=" * 70)
    print("MULTI-PLOT-GRID ERSTELLEN")
    print("=" * 70)

    # Iteriere über alle Kombinationen
    for row_idx, metric in enumerate(metrics):
        for col_idx, vary_param in enumerate(vary_params):
            ax = axes[row_idx, col_idx]

            print(f"\nPlot [{row_idx + 1},{col_idx + 1}]: {metric} vs. {vary_param}")

            # Bestimme fixierte Parameter
            fixed_params = get_fixed_parameters(vary_param)

            # Filtere Daten
            filtered_data, fixed_values = filter_by_most_common_group(data, fixed_params)

            # Extrahiere Plot-Daten
            x_data, y_data, x_errors = extract_plot_data(filtered_data, vary_param, metric)

            if len(x_data) == 0:
                ax.text(0.5, 0.5, 'Keine Daten',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='red')
                ax.set_xlabel(PARAM_LABELS[vary_param], fontsize=10)
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
                continue

            # Plotte Daten
            if vary_param == 'grain_size':
                ax.errorbar(x_data, y_data, xerr=x_errors,
                            fmt='o', markersize=6, capsize=4, capthick=1.5,
                            color='C0')
            else:
                ax.scatter(x_data, y_data, marker='o', color='C0')

            # Achsenbeschriftungen
            ax.set_xlabel(PARAM_LABELS[vary_param], fontsize=10)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)

            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # Titel nur für oberste Zeile
            if row_idx == 0:
                ax.set_title(f"{PARAM_LABELS[vary_param]}",
                             fontsize=11, fontweight='bold', pad=10)

            # Tick-Parameter
            ax.tick_params(labelsize=9)

            print(f"  ✓ {len(x_data)} Datenpunkte geplottet")

    # Gesamt-Titel
    fig.suptitle('Krater-Kenngrößen vs. Versuchsparameter',
                 fontsize=16, fontweight='bold', y=0.995)

    # Layout optimieren
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Speichern
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "multi_plot_grid.png"
    output_file = output_dir / filename

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Multi-Plot gespeichert: {output_file}")

    plt.close()


def create_multi_plot_grid_with_fixed_info(data, output_path):
    """
    Erweiterte Version mit Informationen über fixierte Parameter in jedem Subplot
    """
    metrics = ['h', 'd_outer', 'r_inner', 'd_fit', 'sigma_rim', 'A_rim']
    vary_params = ['volume_ml', 'grain_size', 'fill_height_cm']

    fig, axes = plt.subplots(len(metrics), len(vary_params),
                             figsize=(24, 20))

    print("\n" + "=" * 70)
    print("MULTI-PLOT-GRID MIT FIXIERTEN PARAMETERN")
    print("=" * 70)

    # Dictionary zum Speichern der fixierten Werte für jede Spalte
    fixed_info = {}

    for row_idx, metric in enumerate(metrics):
        for col_idx, vary_param in enumerate(vary_params):
            ax = axes[row_idx, col_idx]

            print(f"\nPlot [{row_idx + 1},{col_idx + 1}]: {metric} vs. {vary_param}")

            fixed_params = get_fixed_parameters(vary_param)
            filtered_data, fixed_values = filter_by_most_common_group(data, fixed_params)

            # Speichere fixierte Werte für erste Zeile
            if row_idx == 0:
                fixed_info[col_idx] = fixed_values

            x_data, y_data, x_errors = extract_plot_data(filtered_data, vary_param, metric)

            if len(x_data) == 0:
                ax.text(0.5, 0.5, 'Keine Daten',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='red')
                ax.set_xlabel(PARAM_LABELS[vary_param], fontsize=10)
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
                continue

            # Plotte
            if vary_param == 'grain_size':
                ax.errorbar(x_data, y_data, xerr=x_errors,
                            fmt='o-', markersize=6, capsize=4, capthick=1.5,
                            linewidth=1.5, color='C0')
            else:
                ax.plot(x_data, y_data, 'o-', markersize=6,
                        linewidth=1.5, color='C0')

            ax.set_xlabel(PARAM_LABELS[vary_param], fontsize=10)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # Titel mit fixierten Parametern für oberste Zeile
            if row_idx == 0:
                subtitle = create_subtitle_short(fixed_values)
                title_text = f"{PARAM_LABELS[vary_param]}\n({subtitle})"
                ax.set_title(title_text, fontsize=10, fontweight='bold', pad=10)

            ax.tick_params(labelsize=9)

            print(f"  ✓ {len(x_data)} Datenpunkte geplottet")

    fig.suptitle('Krater-Kenngrößen vs. Versuchsparameter',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "multi_plot_grid_detailed.png"
    output_file = output_dir / filename

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Multi-Plot gespeichert: {output_file}")

    plt.close()



def create_subtitle_short(fixed_values):
    """
    Erstellt kurzen Untertitel für Subplots
    """
    parts = []
    for param, value in fixed_values.items():
        if param == 'grain_size':
            parts.append(f"Korn: {value} µm")
        elif param == 'volume_ml':
            parts.append(f"Vol: {value} ml")
        elif param == 'fill_height_cm':
            parts.append(f"Höhe: {value} cm")

    return ", ".join(parts)


# ============================================================================
# ERWEITERTE HAUPTPROGRAMME
# ============================================================================

def main_single_plot():
    """
    Ursprüngliche Funktion für einzelnen Plot
    """
    print("=" * 70)
    print("KRATER-PLOT-GENERATOR (EINZELPLOT)")
    print("=" * 70)
    print(f"\nKonfiguration:")
    print(f"  CSV-Datei: {CSV_PATH}")
    print(f"  Variierter Parameter: {VARY_PARAMETER}")
    print(f"  Krater-Kenngröße: {CRATER_METRIC}")
    print(f"  Output-Pfad: {OUTPUT_PATH}")
    print()

    data = load_csv_data(CSV_PATH)
    fixed_params = get_fixed_parameters(VARY_PARAMETER)
    filtered_data, fixed_values = filter_by_most_common_group(data, fixed_params)
    x_data, y_data, x_errors = extract_plot_data(filtered_data, VARY_PARAMETER, CRATER_METRIC)

    if len(x_data) == 0:
        print("\n✗ FEHLER: Keine gültigen Datenpunkte gefunden!")
        return

    create_plot(x_data, y_data, x_errors, VARY_PARAMETER, CRATER_METRIC,
                fixed_values, OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("FERTIG!")
    print("=" * 70)


def main_multi_plot():
    """
    Neue Funktion für Multi-Plot-Grid
    """
    print("=" * 70)
    print("KRATER-PLOT-GENERATOR (MULTI-PLOT-GRID)")
    print("=" * 70)
    print(f"\nKonfiguration:")
    print(f"  CSV-Datei: {CSV_PATH}")
    print(f"  Output-Pfad: {OUTPUT_PATH}")
    print(f"  Grid: 6 Zeilen (Metriken) x 2 Spalten (volume_ml, grain_size)")
    print()

    data = load_csv_data(CSV_PATH)

    # Erstelle einfaches Grid
    create_multi_plot_grid(data, OUTPUT_PATH)

    # Optional: Erstelle auch detailliertes Grid
    create_multi_plot_grid_with_fixed_info(data, OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("FERTIG!")
    print("=" * 70)


# ============================================================================
# HAUPTPROGRAMM MIT MODUS-AUSWAHL
# ============================================================================

def main():
    """
    Hauptprogramm mit Auswahl zwischen Einzel- und Multi-Plot
    """
    # MODUS-AUSWAHL
    MODE = 'single'  # 'single' oder 'multi'

    if MODE == 'single':
        main_single_plot()
    elif MODE == 'multi':
        main_multi_plot()
    else:
        print(f"✗ FEHLER: Ungültiger Modus '{MODE}'. Verwende 'single' oder 'multi'.")


if __name__ == "__main__":
    main()
