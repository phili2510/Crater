"""
Plot-Generator für Laser-Messpunkte aus CSV
============================================
Liest measurement_points.csv ein und erstellt alle Visualisierungen.

Usage:
    python plot_from_csv.py path/to/measurement_points.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import csv
from pathlib import Path
import sys
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


INPUT_PATH =r'/Users/philippadelbrecht/PycharmProjects/Crater/Evaluation+plotting/Volume/Volume3'

BATCH_MODE = True
def load_measurement_points(csv_path):
    """
    Lädt X/Y/Z-Koordinaten aus measurement_points.csv.

    Args:
        csv_path: Pfad zur measurement_points.csv

    Returns:
        numpy array (N, 3) mit [x, y, z] Koordinaten
    """
    points = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['x_cm'])
            y = float(row['y_cm'])
            z = float(row['z_cm'])
            points.append([x, y, z])

    return np.array(points)


def load_metadata(csv_path):
    """
    Lädt Metadaten aus metadata.csv (Key-Value-Format: parameter,value).

    Args:
        csv_path: Pfad zur metadata.csv

    Returns:
        Dictionary mit Metadaten {parameter_name: value_string}
    """
    metadata = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get('parameter', '').strip()
            value = row.get('value', '').strip()
            if key:
                metadata[key] = value

    if len(metadata) == 0:
        return None

    return metadata

def print_metadata_info(metadata):
    if not metadata:
        print("[WARNING] Keine Metadaten verfügbar")
        return

    print("\n" + "=" * 70)
    print("METADATEN")
    print("=" * 70)
    print(f"Kalibrierungsbild:     {metadata.get('calibration_image', 'N/A')}")
    print(f"Messbild:              {metadata.get('measurement_image', 'N/A')}")
    print(f"\nBrechungsindizes:")
    print(f"  n_air:               {metadata.get('n_air', 'N/A')}")
    print(f"  n_water:             {metadata.get('n_water', 'N/A')}")
    print(f"\nKameraposition:")
    print(f"  X:                   {metadata.get('camera_x_cm', 'N/A')} cm")
    print(f"  Y:                   {metadata.get('camera_y_cm', 'N/A')} cm")
    print(f"  Z:                   {metadata.get('camera_z_cm', 'N/A')} cm")
    print(f"\nKalibrierungsebene:")
    print(f"  Y-Ebene:             {metadata.get('calibration_y_plane_cm', 'N/A')} cm")
    print(f"\nTank-Geometrie:")
    print(f"  Breite (X):          {metadata.get('tank_width_x_cm', 'N/A')} cm")
    print(f"  Tiefe (Y):           {metadata.get('tank_depth_y_cm', 'N/A')} cm")
    print(f"  Höhe (Z):            {metadata.get('tank_height_z_cm', 'N/A')} cm")
    print(f"\nAnzahl Punkte:         {metadata.get('num_points', 'N/A')}")
    print("=" * 70 + "\n")


def parse_folder_name(folder_name):
    """
    Parst Ordnernamen im Format: {korngröße}_{volumen}ml_{füllhöhe}cm_measurement_results

    Args:
        folder_name: String, z.B. "300-400_1000ml_40cm_measurement_results"

    Returns:
        Dictionary mit Keys: grain_size, volume_ml, fill_height_cm
        oder None falls Parsing fehlschlägt
    """
    try:
        # Entferne "_measurement_results" am Ende
        if not folder_name.endswith('_measurement_results'):
            return None

        name_without_suffix = folder_name.replace('_measurement_results', '')

        # Splitte bei Unterstrichen
        parts = name_without_suffix.split('_')

        if len(parts) != 3:
            return None

        grain_size = parts[0]  # z.B. "300-400"

        # Extrahiere Volumen (entferne "ml")
        volume_str = parts[1]
        if not volume_str.endswith('ml'):
            return None
        volume_ml = float(volume_str.replace('ml', ''))

        # Extrahiere Füllhöhe (entferne "cm")
        fill_height_str = parts[2]
        if not fill_height_str.endswith('cm'):
            return None
        fill_height_cm = float(fill_height_str.replace('cm', ''))

        return {
            'grain_size': grain_size,
            'volume_ml': volume_ml,
            'fill_height_cm': fill_height_cm
        }

    except Exception as e:
        print(f"[WARNING] Fehler beim Parsen des Ordnernamens '{folder_name}': {e}")
        return None


def clean_measurement_points(points, metadata):
    """
    Bereinigt Messpunkte in 5 Schritten:
    1. Ränder außerhalb des Tanks abschneiden
    2. Medianfilter zur Spike-Entfernung
    3. Zentrierung (Kratermitte auf X=0 verschieben)
    4. Tilt-Korrektur (Baseline-Entschieflage)
    5. Negative Z-Werte auf 0 clampen

    Args:
        points: numpy array, Shape (N,3) mit Spalten [x, y, z]
        metadata: dict mit Metadaten oder None

    Returns:
        Bereinigtes numpy array (N',3)
    """
    # Parameter
    MEDIAN_WINDOW = 15  # Halbe Fensterbreite für lokalen Median
    MAD_FACTOR = 3.5  # Ausreißer-Schwelle
    MAD_FLOOR = 0.05  # Mindest-MAD in cm
    BASELINE_PERCENT = 10  # Prozent der äußersten Punkte für Baseline-Fit

    print(f"\n{'=' * 60}")
    print("PREPROCESSING: MESSPUNKTE BEREINIGEN")
    print(f"{'=' * 60}\n")

    # Arbeite auf Kopie
    points = points.copy()
    n_original = len(points)

    # ============================================================
    # SCHRITT 1: RÄNDER ABSCHNEIDEN
    # ============================================================
    print(f"[SCHRITT 1] Ränder abschneiden...")

    if metadata is not None and 'tank_width_x_cm' in metadata:
        tank_width = float(metadata['tank_width_x_cm'])
        print(f"  Tank-Breite: {tank_width:.2f} cm")

        # Filtere Punkte innerhalb [0, tank_width]
        mask = (points[:, 0] >= 0) & (points[:, 0] <= tank_width)
        points = points[mask]

        n_removed = n_original - len(points)
        print(f"  Entfernt: {n_removed} Punkte außerhalb [0, {tank_width:.2f}] cm")
        print(f"  Verbleibend: {len(points)} Punkte")
    else:
        print(f"  Übersprungen (keine Metadaten oder tank_width_x_cm fehlt)")

    if len(points) == 0:
        print("[ERROR] Keine Punkte nach Schritt 1 übrig!")
        return points

    # ============================================================
    # SCHRITT 2: MEDIANFILTER (SPIKE-ENTFERNUNG)
    # ============================================================
    print(f"\n[SCHRITT 2] Medianfilter (Spike-Entfernung)...")
    print(f"  Fenster: ±{MEDIAN_WINDOW} Nachbarn")
    print(f"  MAD-Faktor: {MAD_FACTOR}, MAD-Floor: {MAD_FLOOR} cm")

    # Sortiere nach X
    sort_idx = np.argsort(points[:, 0])
    points = points[sort_idx]

    z_vals = points[:, 2].copy()
    n = len(z_vals)
    outlier_count = 0

    for i in range(n):
        # Definiere Fenster
        window_start = max(0, i - MEDIAN_WINDOW)
        window_end = min(n, i + MEDIAN_WINDOW + 1)

        window_z = z_vals[window_start:window_end]

        # Lokaler Median
        local_median = np.median(window_z)

        # Lokale MAD
        abs_deviations = np.abs(window_z - local_median)
        local_mad = np.median(abs_deviations)

        # Abweichung des aktuellen Punktes
        deviation = abs(z_vals[i] - local_median)

        # Ausreißer-Schwelle
        mad_threshold = max(local_mad, MAD_FLOOR)

        if deviation > MAD_FACTOR * mad_threshold:
            z_vals[i] = local_median
            outlier_count += 1

    points[:, 2] = z_vals

    print(f"  Ausreißer ersetzt: {outlier_count} von {n} Punkten ({100 * outlier_count / n:.1f}%)")

    # ============================================================
    # SCHRITT 3: ZENTRIERUNG (KRATERMITTE AUF X=0)
    # ============================================================
    print(f"\n[SCHRITT 3] Zentrierung (Kratermitte auf X=0 verschieben)...")

    if metadata is not None and 'tank_width_x_cm' in metadata:
        tank_width = float(metadata['tank_width_x_cm'])
        tank_center = tank_width / 2.0
        print(f"  Tank-Mitte: {tank_center:.2f} cm")

        # Punkte sind bereits nach X sortiert
        x_sorted = points[:, 0]
        z_sorted = points[:, 2]

        # Ermittle Kratermitte
        peaks, properties = find_peaks(z_sorted, prominence=0.05)

        if len(peaks) >= 2:
            # Nimm die zwei höchsten Peaks
            peak_heights = z_sorted[peaks]
            sorted_peak_indices = np.argsort(peak_heights)[-2:]
            top_two_peaks = peaks[sorted_peak_indices]

            # Berechne Mittelpunkt der zwei höchsten Peaks
            crater_center = np.mean(x_sorted[top_two_peaks])
            print(f"  Kratermitte (aus 2 höchsten Peaks): {crater_center:.2f} cm")
        else:
            # Fallback: Schwerpunkt der oberen 20% der Z-Werte
            threshold = np.percentile(z_sorted, 80)
            high_points_mask = z_sorted >= threshold
            crater_center = np.mean(x_sorted[high_points_mask])
            print(f"  Kratermitte (Fallback, obere 20%): {crater_center:.2f} cm")

        # Berechne Verschiebung zur Tank-Mitte
        shift = tank_center - crater_center
        print(f"  Verschiebung zur Tank-Mitte: {shift:.2f} cm")

        # Verschiebe alle X-Koordinaten zur Tank-Mitte
        points[:, 0] += shift

        # Verschiebe X-Achse so dass Kratermitte bei 0 liegt
        points[:, 0] -= tank_center
        print(f"  X-Achse verschoben: Kratermitte liegt jetzt bei X=0")
        print(f"  Neuer X-Bereich: [{-tank_width / 2:.2f}, {tank_width / 2:.2f}] cm")

        # Schneide Punkte außerhalb [-tank_width/2, +tank_width/2] ab
        mask = (points[:, 0] >= -tank_width / 2) & (points[:, 0] <= tank_width / 2)
        n_before_trim = len(points)
        points = points[mask]
        n_trimmed = n_before_trim - len(points)

        print(
            f"  Nach Zentrierung entfernt: {n_trimmed} Punkte außerhalb [{-tank_width / 2:.2f}, {tank_width / 2:.2f}] cm")
        print(f"  Verbleibend: {len(points)} Punkte")
    else:
        print(f"  Übersprungen (keine Metadaten oder tank_width_x_cm fehlt)")

    if len(points) == 0:
        print("[ERROR] Keine Punkte nach Schritt 3 übrig!")
        return points

    # ============================================================
    # SCHRITT 4: TILT-KORREKTUR (BASELINE-ENTSCHIEFLAGE)
    # ============================================================
    print(f"\n[SCHRITT 4] Tilt-Korrektur (Baseline-Entschieflage)...")
    print(f"  Baseline-Bereich: äußerste {BASELINE_PERCENT}% links + rechts")

    # Sortiere erneut nach X (falls durch Zentrierung durcheinander)
    sort_idx = np.argsort(points[:, 0])
    points = points[sort_idx]

    n = len(points)

    # Berechne Anzahl der Baseline-Punkte
    n_baseline = int(np.ceil(n * BASELINE_PERCENT / 100.0))

    # Äußerste Punkte links und rechts
    baseline_indices = np.concatenate([
        np.arange(n_baseline),  # Links
        np.arange(n - n_baseline, n)  # Rechts
    ])

    baseline_x = points[baseline_indices, 0]
    baseline_z = points[baseline_indices, 2]

    # Fitte Gerade z = a*x + b
    coeffs = np.polyfit(baseline_x, baseline_z, deg=1)
    a, b = coeffs[0], coeffs[1]

    print(f"  Baseline-Fit: z = {a:.6f}*x + {b:.6f}")
    print(f"  Baseline-Punkte: {len(baseline_indices)} ({2 * BASELINE_PERCENT}% von {n})")

    # Subtrahiere Gerade von ALLEN Z-Werten
    tilt = a * points[:, 0] + b
    points[:, 2] = points[:, 2] - tilt

    print(f"  Tilt-Korrektur angewandt")

    # ============================================================
    # SCHRITT 5: NEGATIVE WERTE AUF 0 CLAMPEN
    # ============================================================
    print(f"\n[SCHRITT 5] Negative Z-Werte auf 0 clampen...")

    negative_mask = points[:, 2] < 0
    n_negative = np.sum(negative_mask)

    points[negative_mask, 2] = 0.0

    print(f"  Geclampt: {n_negative} Punkte ({100 * n_negative / n:.1f}%)")

    # ============================================================
    # ZUSAMMENFASSUNG
    # ============================================================
    print(f"\n{'=' * 60}")
    print("PREPROCESSING ABGESCHLOSSEN")
    print(f"{'=' * 60}")
    print(f"Original: {n_original} Punkte")
    print(f"Nach Bereinigung: {len(points)} Punkte")
    print(f"X-Bereich: {points[:, 0].min():.3f} bis {points[:, 0].max():.3f} cm")
    print(f"Z-Bereich: {points[:, 2].min():.3f} bis {points[:, 2].max():.3f} cm")
    print(f"{'=' * 60}\n")

    return points


def compute_crater_metrics(gaussian_params, points):
    """
    Berechnet Krater-Kenngrößen aus Gauß-Fit-Parametern.

    Verwendet symmetrische Doppel-Gauß-Funktion:
    f(x) = A_rim * exp(-(x - mu1)² / (2σ²)) + A_rim * exp(-(x - mu2)² / (2σ²))
    wobei mu1 = mu_center - d, mu2 = mu_center + d

    Berechnet:
    - h: Maximale Rim-Höhe (numerisch bestimmt)
    - d_outer: Außendurchmesser (bei h/2 auf Außenseite)
    - r_inner: Rim-Innenradius (bei h/2 auf Innenseite)

    Args:
        gaussian_params: Dictionary mit Fit-Parametern
        points: numpy array mit bereinigten Messpunkten

    Returns:
        Dictionary mit Keys: h, d_outer, r_inner (jeweils float oder None)
    """
    print(f"\n{'=' * 60}")
    print("KRATER-KENNGROSSEN BERECHNUNG")
    print(f"{'=' * 60}\n")

    if gaussian_params is None:
        print("[ERROR] Keine Gauß-Parameter verfügbar")
        return {'h': None, 'd_outer': None, 'r_inner': None}

    A_rim = gaussian_params['A_rim']
    d = gaussian_params['d']
    sigma_rim = gaussian_params['sigma_rim']
    mu_center = gaussian_params['mu3']

    print(f"Gauß-Parameter:")
    print(f"  A_rim:      {A_rim:.6f} cm")
    print(f"  d:          {d:.6f} cm")
    print(f"  sigma_rim:  {sigma_rim:.6f} cm")
    print(f"  mu_center:  {mu_center:.6f} cm")

    # Berechne mu1 und mu2
    mu1 = mu_center - d
    mu2 = mu_center + d

    print(f"  mu1:        {mu1:.6f} cm")
    print(f"  mu2:        {mu2:.6f} cm\n")

    # Definiere symmetrische Doppel-Gauß-Funktion
    def symmetric_double_gaussian(x):
        gauss_left = A_rim * np.exp(-(x - mu1) ** 2 / (2 * sigma_rim ** 2))
        gauss_right = A_rim * np.exp(-(x - mu2) ** 2 / (2 * sigma_rim ** 2))
        return gauss_left + gauss_right

    # Erstelle dichtes x-Array für numerische Auswertung
    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    x_dense = np.linspace(x_min, x_max, 10000)
    y_dense = symmetric_double_gaussian(x_dense)

    # ============================================================
    # 1. BERECHNE h (MAXIMALE RIM-HÖHE)
    # ============================================================
    h = np.max(y_dense)
    h_idx = np.argmax(y_dense)
    h_position = x_dense[h_idx]

    print(f"[1] Rim-Höhe h:")
    print(f"    h = {h:.6f} cm (bei x = {h_position:.6f} cm)\n")

    # ============================================================
    # 2. BERECHNE d_outer (AUSSENDURCHMESSER)
    # ============================================================
    half_height = h / 2.0
    print(f"[2] Außendurchmesser d_outer (bei h/2 = {half_height:.6f} cm):")

    # Suche Kreuzungspunkt auf Außenseite des rechten Peaks (x > mu2)
    outer_mask = x_dense > mu2
    x_outer_region = x_dense[outer_mask]
    y_outer_region = y_dense[outer_mask]

    d_outer = None
    x_outer = None

    if len(x_outer_region) > 0:
        # Finde Vorzeichenwechsel von (y - h/2)
        diff_outer = y_outer_region - half_height
        sign_changes = np.where(np.diff(np.sign(diff_outer)))[0]

        if len(sign_changes) > 0:
            # Nimm ersten Vorzeichenwechsel (nächster zu mu2)
            idx = sign_changes[0]

            # Lineare Interpolation zwischen idx und idx+1
            x1, x2 = x_outer_region[idx], x_outer_region[idx + 1]
            y1, y2 = y_outer_region[idx], y_outer_region[idx + 1]

            # Interpoliere x-Position bei y = half_height
            x_outer = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)

            r_outer = abs(x_outer - mu_center)
            d_outer = 2 * r_outer

            print(f"    Kreuzungspunkt gefunden bei x = {x_outer:.6f} cm")
            print(f"    r_outer = {r_outer:.6f} cm")
            print(f"    d_outer = {d_outer:.6f} cm\n")
        else:
            print(f"    [WARNING] Kein Kreuzungspunkt gefunden\n")
    else:
        print(f"    [WARNING] Keine Datenpunkte auf Außenseite\n")

    # ============================================================
    # 3. BERECHNE r_inner (RIM-INNENRADIUS)
    # ============================================================
    print(f"[3] Rim-Innenradius r_inner (bei h/2 = {half_height:.6f} cm):")

    # Suche Kreuzungspunkt auf Innenseite des rechten Peaks (mu_center < x < mu2)
    inner_mask = (x_dense > mu_center) & (x_dense < mu2)
    x_inner_region = x_dense[inner_mask]
    y_inner_region = y_dense[inner_mask]

    r_inner = None
    x_inner = None

    if len(x_inner_region) > 0:
        # Finde Vorzeichenwechsel von (y - h/2)
        diff_inner = y_inner_region - half_height
        sign_changes = np.where(np.diff(np.sign(diff_inner)))[0]

        if len(sign_changes) > 0:
            # Nimm letzten Vorzeichenwechsel (nächster zu mu2)
            idx = sign_changes[-1]

            # Lineare Interpolation zwischen idx und idx+1
            x1, x2 = x_inner_region[idx], x_inner_region[idx + 1]
            y1, y2 = y_inner_region[idx], y_inner_region[idx + 1]

            # Interpoliere x-Position bei y = half_height
            x_inner = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)

            r_inner = abs(x_inner - mu_center)

            print(f"    Kreuzungspunkt gefunden bei x = {x_inner:.6f} cm")
            print(f"    r_inner = {r_inner:.6f} cm\n")
        else:
            print(f"    [WARNING] Kein Kreuzungspunkt gefunden\n")
    else:
        print(f"    [WARNING] Keine Datenpunkte auf Innenseite\n")

    # ============================================================
    # ZUSAMMENFASSUNG
    # ============================================================
    print(f"{'=' * 60}")
    print("KRATER-KENNGROSSEN (ZUSAMMENFASSUNG)")
    print(f"{'=' * 60}")
    print(f"h (Rim-Höhe):           {h:.6f} cm" if h is not None else "h:           N/A")
    print(f"d_outer (Außendurchm.): {d_outer:.6f} cm" if d_outer is not None else "d_outer:     N/A")
    print(f"r_inner (Innenradius):  {r_inner:.6f} cm" if r_inner is not None else "r_inner:     N/A")
    print(f"{'=' * 60}\n")

    return {
        'h': h,
        'd_outer': d_outer,
        'r_inner': r_inner
    }


def save_batch_results_csv(results_list, output_path):
    """
    Speichert Batch-Ergebnisse in CSV-Datei.

    Args:
        results_list: Liste von Ergebnis-Dictionaries
        output_path: Path-Objekt des Ausgabeordners
    """
    if len(results_list) == 0:
        print("[WARNING] Keine Ergebnisse zum Speichern")
        return

    csv_path = output_path / 'batch_results.csv'

    # CSV-Header
    fieldnames = [
        'folder_name',
        'grain_size',
        'volume_ml',
        'fill_height_cm',
        'h',
        'd_outer',
        'r_inner',
        'A_rim',
        'd_fit',
        'sigma_rim',
        'mu_center',
        'rms_error'
    ]

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results_list:
                # Konvertiere None zu leeren Strings für CSV
                row = {}
                for key in fieldnames:
                    value = result.get(key, None)
                    if value is None:
                        row[key] = ''
                    else:
                        row[key] = value

                writer.writerow(row)

        print(f"\n{'=' * 70}")
        print(f"[OK] Batch-Ergebnisse gespeichert: {csv_path}")
        print(f"    Anzahl Zeilen: {len(results_list)}")
        print(f"{'=' * 70}\n")

    except Exception as e:
        print(f"[ERROR] Fehler beim Speichern der CSV: {e}")


def create_summary_report_with_metadata(points, metadata, output_path):
    """
    Erstellt zusammenfassenden Bericht mit Metadaten als Textdatei.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    report_path = output_path / 'summary_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LASER-MESSPUNKTE ANALYSE - ZUSAMMENFASSUNG\n")
        f.write("=" * 70 + "\n\n")

        # Metadaten
        if metadata:
            f.write("-" * 70 + "\n")
            f.write("METADATEN\n")
            f.write("-" * 70 + "\n")
            f.write(f"Zeitstempel:           {metadata.get('timestamp', 'N/A')}\n")
            f.write(f"Kalibrierungsbild:     {metadata.get('calibration_image', 'N/A')}\n")
            f.write(f"Messbild:              {metadata.get('measurement_image', 'N/A')}\n\n")

            f.write(f"Brechungsindizes:\n")
            f.write(f"  n_air:               {metadata.get('n_air', 'N/A')}\n")
            f.write(f"  n_water:             {metadata.get('n_water', 'N/A')}\n\n")

            f.write(f"Kameraposition:\n")
            f.write(f"  X:                   {metadata.get('camera_x_cm', 'N/A')} cm\n")
            f.write(f"  Y:                   {metadata.get('camera_y_cm', 'N/A')} cm\n")
            f.write(f"  Z:                   {metadata.get('camera_z_cm', 'N/A')} cm\n\n")

            f.write(f"Kalibrierungsebene:\n")
            f.write(f"  Y-Ebene:             {metadata.get('y_calib_plane_cm', 'N/A')} cm\n\n")

        f.write("-" * 70 + "\n")
        f.write("MESSPUNKTE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Anzahl Messpunkte:     {len(points)}\n\n")

        f.write("-" * 70 + "\n")
        f.write("X-KOORDINATEN (POSITION)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Minimum:             {x.min():.6f} cm\n")
        f.write(f"  Maximum:             {x.max():.6f} cm\n")
        f.write(f"  Spannweite:          {x.max() - x.min():.6f} cm\n")
        f.write(f"  Mittelwert:          {x.mean():.6f} cm\n\n")

        f.write("-" * 70 + "\n")
        f.write("Y-KOORDINATEN (TIEFE)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Minimum:             {y.min():.6f} cm\n")
        f.write(f"  Maximum:             {y.max():.6f} cm\n")
        f.write(f"  Spannweite:          {y.max() - y.min():.6f} cm\n")
        f.write(f"  Mittelwert:          {y.mean():.6f} cm\n\n")

        f.write("-" * 70 + "\n")
        f.write("Z-KOORDINATEN (HÖHE)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Minimum:             {z.min():.6f} cm\n")
        f.write(f"  Maximum:             {z.max():.6f} cm\n")
        f.write(f"  Spannweite:          {z.max() - z.min():.6f} cm\n")
        f.write(f"  Mittelwert:          {z.mean():.6f} cm\n")
        f.write(f"  Median:              {np.median(z):.6f} cm\n")
        f.write(f"  Standardabw.:        {z.std():.6f} cm\n\n")

        f.write(f"  25% Perzentil:       {np.percentile(z, 25):.6f} cm\n")
        f.write(f"  75% Perzentil:       {np.percentile(z, 75):.6f} cm\n")
        f.write(f"  95% Perzentil:       {np.percentile(z, 95):.6f} cm\n")
        f.write(f"  99% Perzentil:       {np.percentile(z, 99):.6f} cm\n\n")

        f.write("-" * 70 + "\n")
        f.write("HÖHENVERTEILUNG\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Punkte über Null:    {np.sum(z > 0):5d} ({np.sum(z > 0) / len(z) * 100:5.1f}%)\n")
        f.write(f"  Punkte bei Null:     {np.sum(z == 0):5d} ({np.sum(z == 0) / len(z) * 100:5.1f}%)\n")
        f.write(f"  Punkte unter Null:   {np.sum(z < 0):5d} ({np.sum(z < 0) / len(z) * 100:5.1f}%)\n\n")

        # Steigung
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        z_sorted = z[sorted_idx]
        dz_dx = np.gradient(z_sorted, x_sorted)

        f.write("-" * 70 + "\n")
        f.write("STEIGUNG (dZ/dX)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Minimum:             {dz_dx.min():.6f}\n")
        f.write(f"  Maximum:             {dz_dx.max():.6f}\n")
        f.write(f"  Mittelwert:          {dz_dx.mean():.6f}\n")
        f.write(f"  Median:              {np.median(dz_dx):.6f}\n")
        f.write(f"  Std.abw.:            {dz_dx.std():.6f}\n")
        f.write(f"  Max. Betrag:         {np.abs(dz_dx).max():.6f}\n\n")

        # Metadaten-Statistiken (falls vorhanden)
        if metadata:
            f.write("-" * 70 + "\n")
            f.write("HÖHENSTATISTIKEN AUS METADATEN\n")
            f.write("-" * 70 + "\n")

            height_min = metadata.get('height_min_cm', '')
            height_max = metadata.get('height_max_cm', '')
            height_mean = metadata.get('height_mean_cm', '')
            height_std = metadata.get('height_std_cm', '')

            if height_min:
                f.write(f"  Min (Metadata):      {height_min} cm\n")
            if height_max:
                f.write(f"  Max (Metadata):      {height_max} cm\n")
            if height_mean:
                f.write(f"  Mean (Metadata):     {height_mean} cm\n")
            if height_std:
                f.write(f"  Std (Metadata):      {height_std} cm\n")
            f.write("\n")

        f.write("=" * 70 + "\n")

    print(f"[OK] Zusammenfassung gespeichert: {report_path}")



def plot_3d_points(points, output_path):
    """
    Erstellt 3D-Scatter-Plot der Messpunkte (Stil wie zweiter Code).
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Scatter-Plot mit konsistentem Stil
    scatter = ax.scatter(x, y, z, c='red', s=10, marker='o', alpha=0.8)

    # Verbinde Punkte mit Linie
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    z_sorted = z[sorted_idx]
    ax.plot(x_sorted, y_sorted, z_sorted, 'r-', linewidth=2, alpha=0.7)

    # Achsenbeschriftung (ohne bold/fontsize wie im zweiten Code)
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    ax.set_title('3D Messpunkte')

    # Setze Blickwinkel wie im zweiten Code
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(output_path / '3d_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 3D-Scatter gespeichert: {output_path / '3d_scatter.png'}")




def plot_3d_surface(points, output_path):
    """
    Erstellt 3D-Surface-Plot (Stil wie zweiter Code).
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Sortiere nach X
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    z_sorted = z[sorted_idx]
    y_sorted = y[sorted_idx]

    # Erstelle Linie (wie im zweiten Code)
    ax.plot(x_sorted, y_sorted, z_sorted, 'r-', linewidth=2, alpha=0.7, label='Laserlinie')

    # Füge Bodenfläche hinzu (z=0) - wie im zweiten Code
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_grid = np.linspace(x_min, x_max, 50)
    y_grid = np.linspace(y_min, y_max, 10)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_grid = np.zeros_like(X_grid)

    ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.2, color='gray')

    # Achsenbeschriftung (ohne bold/fontsize)
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    ax.set_title('3D Laserlinie mit Bodenfläche')
    ax.legend()

    # Blickwinkel
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(output_path / '3d_surface.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 3D-Surface gespeichert: {output_path / '3d_surface.png'}")


def plot_height_profile(points, output_path):
    """
    Erstellt 2D-Höhenprofil (Stil wie zweiter Code).
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x = points[:, 0]
    z = points[:, 2]

    # Sortiere nach X
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    z_sorted = z[sorted_idx]

    # Plot (einfacher Stil wie zweiter Code)
    ax.plot(x_sorted, z_sorted, 'b-', linewidth=1.5, label='Höhenprofil')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Boden (z=0)')

    # Fülle Bereich
    ax.fill_between(x_sorted, 0, z_sorted, where=(z_sorted >= 0),
                    color='green', alpha=0.3, label='Erhebung')
    ax.fill_between(x_sorted, 0, z_sorted, where=(z_sorted < 0),
                    color='red', alpha=0.3, label='Vertiefung')

    # Achsenbeschriftung (ohne bold/fontsize)
    ax.set_xlabel('X-Position [cm]')
    ax.set_ylabel('Höhe Z [cm]')
    ax.set_title('Höhenprofil')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'height_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Höhenprofil gespeichert: {output_path / 'height_profile.png'}")

def plot_height_profile_simple(points, output_path):
    """
    Erstellt einfaches 2D-Höhenprofil ohne Legenden und Füllungen.
    Nur Messpunkte als grüne Kreise.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x = points[:, 0]
    z = points[:, 2]

    # Sortiere nach X
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    z_sorted = z[sorted_idx]

    # Plot nur Datenpunkte
    ax.plot(x_sorted, z_sorted, 'o', color='green',
            markersize=4, alpha=0.6)

    # Achsenbeschriftung
    ax.set_xlabel('X-Position [cm]', fontsize=12)
    ax.set_ylabel('Höhe [cm]', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'height_profile_simple.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Einfaches Höhenprofil gespeichert: {output_path / 'height_profile_simple.png'}")


def plot_3d_slope_analysis(points, output_path):
    """
    Erstellt 3D-Steigungsanalyse des Höhenprofils (Stil wie Hauptcode).
    Mit Perzentil-Linien und Statistik-Box.
    """
    x = points[:, 0]
    z = points[:, 2]

    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    z_sorted = z[sorted_idx]

    # Berechne Ableitung
    dz_dx = np.gradient(z_sorted, x_sorted)
    dz_dx_abs = np.abs(dz_dx)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Plotte 3D-Steigung
    ax.plot(x_sorted, dz_dx_abs, 'purple', linewidth=2, marker='o', markersize=3,
            label='|dh/dx| (3D-Steigung des Höhenprofils)')
    ax.fill_between(x_sorted, 0, dz_dx_abs, alpha=0.3, color='purple')

    # Perzentil-Linien
    ax.axhline(y=np.percentile(dz_dx_abs, 95), color='orange', linestyle='--', linewidth=2,
               label=f'95. Perzentil = {np.percentile(dz_dx_abs, 95):.4f}')
    ax.axhline(y=np.percentile(dz_dx_abs, 99), color='red', linestyle='--', linewidth=2,
               label=f'99. Perzentil = {np.percentile(dz_dx_abs, 99):.4f}')
    ax.axhline(y=np.median(dz_dx_abs), color='green', linestyle=':', linewidth=1.5,
               label=f'Median = {np.median(dz_dx_abs):.4f}')

    ax.set_xlabel('X-Koordinate [cm]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Steigung |dh/dx| [cm/cm]', fontsize=12, fontweight='bold')
    ax.set_title('3D-STEIGUNG — Ableitung des Höhenprofils\n(Objektanalyse in 3D-Weltkoordinaten)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Statistik-Box
    stats_box = f'3D-STEIGUNG STATISTIKEN:\n'
    stats_box += f'Min: {dz_dx_abs.min():.4f}\n'
    stats_box += f'Max: {dz_dx_abs.max():.4f}\n'
    stats_box += f'Mittelwert: {dz_dx_abs.mean():.4f}\n'
    stats_box += f'Median: {np.median(dz_dx_abs):.4f}\n'
    stats_box += f'Std: {dz_dx_abs.std():.4f}\n\n'
    stats_box += f'PERZENTILE:\n'
    stats_box += f'50%: {np.percentile(dz_dx_abs, 50):.4f}\n'
    stats_box += f'75%: {np.percentile(dz_dx_abs, 75):.4f}\n'
    stats_box += f'90%: {np.percentile(dz_dx_abs, 90):.4f}\n'
    stats_box += f'95%: {np.percentile(dz_dx_abs, 95):.4f}\n'
    stats_box += f'99%: {np.percentile(dz_dx_abs, 99):.4f}'

    ax.text(0.02, 0.98, stats_box, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan',
                                               alpha=0.9, edgecolor='black', linewidth=1.5))

    plt.tight_layout()
    plt.savefig(output_path / '3d_slope_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 3D-Steigung gespeichert: {output_path / '3d_slope_analysis.png'}")


def plot_gaussian_fit(points, output_path, metadata=None):
    """
    Erstellt Gauß-Fit des Höhenprofils (Stil wie Hauptcode).
    Zeigt einzelne Gauß-Kurven, Krater-Kenngrößen und vertikale Peak-Markierungen.
    NUR HAUPTPLOT (ohne Residuen-Plot).
    """
    x = points[:, 0]
    z = points[:, 2]

    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    z_sorted = z[sorted_idx]

    def symmetric_double_gaussian(x, A_rim, d, sigma_rim, mu_center):
        mu1 = mu_center - d
        mu2 = mu_center + d
        gauss_left = A_rim * np.exp(-(x - mu1) ** 2 / (2 * sigma_rim ** 2))
        gauss_right = A_rim * np.exp(-(x - mu2) ** 2 / (2 * sigma_rim ** 2))
        return gauss_left + gauss_right

    try:
        # Startwert-Schätzung
        peaks, properties = find_peaks(z_sorted, prominence=0.1)

        if len(peaks) < 2:
            global_max_idx = np.argmax(z_sorted)
            x_center = x_sorted[len(x_sorted) // 2]
            if x_sorted[global_max_idx] < x_center:
                right_half_idx = np.where(x_sorted >= x_center)[0]
                second_max_idx = right_half_idx[np.argmax(z_sorted[right_half_idx])]
                mu1_init = x_sorted[global_max_idx]
                mu2_init = x_sorted[second_max_idx]
                A1_init = z_sorted[global_max_idx]
                A2_init = z_sorted[second_max_idx]
            else:
                left_half_idx = np.where(x_sorted < x_center)[0]
                second_max_idx = left_half_idx[np.argmax(z_sorted[left_half_idx])]
                mu1_init = x_sorted[second_max_idx]
                mu2_init = x_sorted[global_max_idx]
                A1_init = z_sorted[second_max_idx]
                A2_init = z_sorted[global_max_idx]
        else:
            peak_heights = z_sorted[peaks]
            sorted_peak_indices = np.argsort(peak_heights)[-2:]
            top_two_peaks = peaks[sorted_peak_indices]
            top_two_peaks = sorted(top_two_peaks)
            mu1_init = x_sorted[top_two_peaks[0]]
            mu2_init = x_sorted[top_two_peaks[1]]
            A1_init = z_sorted[top_two_peaks[0]]
            A2_init = z_sorted[top_two_peaks[1]]

        A_rim_init = (A1_init + A2_init) / 2.0

        # Kratermitte liegt jetzt bei X=0
        mu3_init = 0.0

        d_init = abs(mu2_init - mu1_init) / 2.0
        sigma_rim_init = d_init / 2.0

        x_min = x_sorted.min()
        x_max = x_sorted.max()
        x_range = x_max - x_min

        # Doppel-Gauß-Fit (ohne c-Parameter)
        p0_sym_double = [A_rim_init, d_init, sigma_rim_init, mu3_init]
        lower_bounds_sym_double = [0, 1.0, 0.5, x_min]
        upper_bounds_sym_double = [np.inf, x_range / 2, np.inf, x_max]

        popt_sym_double, pcov_sym_double = curve_fit(
            symmetric_double_gaussian, x_sorted, z_sorted,
            p0=p0_sym_double,
            bounds=(lower_bounds_sym_double, upper_bounds_sym_double),
            maxfev=10000
        )

        A_rim_fit, d_fit, sigma_rim_fit, mu3_fit = popt_sym_double
        c_fit = 0.0  # Fest auf 0 gesetzt
        mu1_fit = mu3_fit - d_fit
        mu2_fit = mu3_fit + d_fit

        x_fit = np.linspace(x_sorted.min(), x_sorted.max(), 500)
        y_fit = symmetric_double_gaussian(x_fit, *popt_sym_double)

        gauss_left_fit = A_rim_fit * np.exp(-(x_fit - mu1_fit) ** 2 / (2 * sigma_rim_fit ** 2))
        gauss_right_fit = A_rim_fit * np.exp(-(x_fit - mu2_fit) ** 2 / (2 * sigma_rim_fit ** 2))

        residuals = z_sorted - symmetric_double_gaussian(x_sorted, *popt_sym_double)
        rms_error = np.sqrt(np.mean(residuals ** 2))

        fit_model = 'symmetric_double'
        print(f"[OK] Doppel-Gauß-Fit erfolgreich (RMS: {rms_error:.4f} cm)")

        # Krater-Kenngrößen berechnen
        crater_center = mu3_fit
        crater_radius = d_fit

        center_region = (x_fit >= mu1_fit) & (x_fit <= mu2_fit)
        crater_depth = y_fit[center_region].min() if np.any(center_region) else 0.0

        # Plot erstellen
        fig_gauss, ax_gauss = plt.subplots(figsize=(14, 8))

        # Geglättete Linie statt Messpunkte
        ax_gauss.plot(x_sorted, z_sorted, '-', color='green', linewidth=1.5,
                      alpha=0.8, label='Höhenprofil (Daten)')
        ax_gauss.plot(x_fit, y_fit, '--', color='red', linewidth=2.5,
                      label='Doppel-Gauß-Fit (Summe)')

        ax_gauss.plot(x_fit, gauss_left_fit, ':', color='blue', linewidth=1.5,
                      label=f'Randhügel (links): μ={mu1_fit:.2f} cm')
        ax_gauss.plot(x_fit, gauss_right_fit, ':', color='blue', linewidth=1.5,
                      alpha=0.7, label=f'Randhügel (rechts): μ={mu2_fit:.2f} cm')

        # Vertikale Linien
        ax_gauss.axvline(mu1_fit, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax_gauss.axvline(mu2_fit, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax_gauss.axvline(crater_center, color='red', linestyle='-', alpha=0.7,
                         linewidth=2, label=f'Kratermitte: {crater_center:.2f} cm')

        ax_gauss.set_xlabel('X-Position [cm]', fontsize=12)
        ax_gauss.set_ylabel('Höhe [cm]', fontsize=12)
        ax_gauss.set_title('Symmetrischer Gauß-Fit des Kraterprofils', fontsize=14, fontweight='bold')
        ax_gauss.grid(True, alpha=0.3)

        # Setze X-Achsen-Limits symmetrisch um 0
        if metadata and 'tank_width_x_cm' in metadata:
            tank_width = float(metadata['tank_width_x_cm'])
            ax_gauss.set_xlim(-tank_width/2, tank_width/2)

        plt.tight_layout()
        plt.savefig(output_path / 'gaussian_fit.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Gauß-Fit gespeichert: {output_path / 'gaussian_fit.png'}")

        # Speichere Fit-Parameter
        with open(output_path / 'gaussian_fit_params.txt', 'w') as f:
            f.write(f"GAUSSPROFIL-FIT PARAMETER\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Modell: SYMMETRIC_DOUBLE\n")
            f.write(f"RMS-Fehler: {rms_error:.6f} cm\n\n")
            f.write(f"Parameter:\n")
            f.write(f"  A_rim (Randhöhe):     {A_rim_fit:.6f} cm\n")
            f.write(f"  d (Halbabstand):      {d_fit:.6f} cm\n")
            f.write(f"  sigma_rim (Breite):   {sigma_rim_fit:.6f} cm\n")
            f.write(f"  mu_center (Zentrum):  {mu3_fit:.6f} cm\n")
            f.write(f"  c (Offset):           {c_fit:.6f} cm (fest)\n\n")
            f.write(f"Berechnete Positionen:\n")
            f.write(f"  mu1 (Linker Rand):    {mu1_fit:.6f} cm\n")
            f.write(f"  mu2 (Rechter Rand):   {mu2_fit:.6f} cm\n")
            f.write(f"  Kraterradius:         {d_fit:.6f} cm\n\n")
            f.write(f"Krater-Kenngrößen:\n")
            f.write(f"  Kratermitte:          {crater_center:.6f} cm\n")
            f.write(f"  Kraterradius:         {crater_radius:.6f} cm\n")
            f.write(f"  Kratertiefe:          {crater_depth:.6f} cm\n")

        print(f"[OK] Fit-Parameter gespeichert: {output_path / 'gaussian_fit_params.txt'}")

        # Rückgabe für Topografie-Plots
        return {
            'fit_model': fit_model,
            'A_rim': A_rim_fit,
            'd': d_fit,
            'sigma_rim': sigma_rim_fit,
            'A3': None,
            'mu3': mu3_fit,
            'sigma3': None,
            'c': c_fit
        }

    except Exception as e:
        print(f"[ERROR] Gauß-Fit fehlgeschlagen: {e}")
        return None


def plot_topography_3d(points, output_path, gaussian_params=None, tank_height = 0):
    """
    Erstellt 3D-Topografie-Plot (rotationssymmetrisch, Stil wie Hauptcode).
    Verwendet plasma-Colormap und radiales Profil.
    """
    if gaussian_params is None:
        print(f"[WARNING] Keine Gauß-Parameter — überspringe 3D-Topografie")
        return

    fit_model = gaussian_params['fit_model']
    A_rim = gaussian_params['A_rim']
    d = gaussian_params['d']
    sigma_rim = gaussian_params['sigma_rim']
    A3 = gaussian_params['A3'] if gaussian_params['A3'] is not None else 0.0
    sigma3 = gaussian_params['sigma3'] if gaussian_params['sigma3'] is not None else 1.0
    c = gaussian_params['c']

    def radial_profile(r, A_rim, d, sigma_rim, A3, sigma3, c):
        rim = A_rim * np.exp(-(r - d) ** 2 / (2 * sigma_rim ** 2))
        center = A3 * np.exp(-r ** 2 / (2 * sigma3 ** 2))
        return np.maximum(rim + center + c, 0.0)

    # Verwende X-Spannweite als Tank-Größe
    x = points[:, 0]
    x_span = x.max() - x.min()
    grid_size = max(x_span * 1.5, 60.0)

    crater_center_x = grid_size / 2.0
    crater_center_y = grid_size / 2.0

    x_grid = np.linspace(0, grid_size, 500)
    y_grid = np.linspace(0, grid_size, 500)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

    R_mesh = np.sqrt((X_mesh - crater_center_x) ** 2 + (Y_mesh - crater_center_y) ** 2)
    H_mesh = radial_profile(R_mesh, A_rim, d, sigma_rim, A3, sigma3, c)

    # 3D-Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(vmin=H_mesh.min(), vmax=H_mesh.max())
    colors = plt.cm.plasma(norm(H_mesh))

    surf = ax.plot_surface(X_mesh, Y_mesh, H_mesh,
                           facecolors=colors,
                           rstride=5, cstride=5,
                           linewidth=0.2,
                           antialiased=True,
                           shade=False)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap='plasma')
    mappable.set_array(H_mesh)
    cbar = fig.colorbar(mappable, ax=ax, label='Höhe [cm]', shrink=0.6, aspect=15)

    ax.set_xlabel('X-Position [cm]', fontsize=11, labelpad=10)
    ax.set_ylabel('Y-Position [cm]', fontsize=11, labelpad=10)
    ax.set_zlabel('Höhe [cm]', fontsize=11, labelpad=10)
    ax.set_title(f'3D-Topografie des Kraters (rotationssymmetrisch)',
                 fontsize=14, fontweight='bold', pad=20)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    if tank_height > 0:
        ax.set_zlim(0, tank_height)
    ax.view_init(elev=30, azim=45)
    ax.grid(True, alpha=0.3, linestyle='--')

    info_text = f"KRATER-KENNGROSSEN:\n"
    info_text += f"Mitte: ({crater_center_x:.1f}, {crater_center_y:.1f}) cm\n"
    info_text += f"Radius: {d:.2f} cm\n"
    info_text += f"Rim-Höhe: {A_rim:.3f} cm\n"
    info_text += f"Max. Höhe: {H_mesh.max():.3f} cm\n"
    if fit_model == 'symmetric_triple':
        info_text += f"Zentralberg: {A3:.3f} cm"

    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
              fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path / 'topography_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 3D-Topografie gespeichert: {output_path / 'topography_3d.png'}")


def create_summary_report(points, output_path):
    """
    Erstellt zusammenfassenden Bericht als Textdatei.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    report_path = output_path / 'summary_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LASER-MESSPUNKTE ANALYSE - ZUSAMMENFASSUNG\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Datei: {output_path.name}\n")
        f.write(f"Anzahl Messpunkte: {len(points)}\n\n")

        f.write("-" * 70 + "\n")
        f.write("X-KOORDINATEN (POSITION)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Minimum:           {x.min():.6f} cm\n")
        f.write(f"  Maximum:           {x.max():.6f} cm\n")
        f.write(f"  Spannweite:        {x.max() - x.min():.6f} cm\n")
        f.write(f"  Mittelwert:        {x.mean():.6f} cm\n\n")

        f.write("-" * 70 + "\n")
        f.write("Y-KOORDINATEN (TIEFE)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Minimum:           {y.min():.6f} cm\n")
        f.write(f"  Maximum:           {y.max():.6f} cm\n")
        f.write(f"  Spannweite:        {y.max() - y.min():.6f} cm\n")
        f.write(f"  Mittelwert:        {y.mean():.6f} cm\n\n")

        f.write("-" * 70 + "\n")
        f.write("Z-KOORDINATEN (HÖHE)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Minimum:           {z.min():.6f} cm\n")
        f.write(f"  Maximum:           {z.max():.6f} cm\n")
        f.write(f"  Spannweite:        {z.max() - z.min():.6f} cm\n")
        f.write(f"  Mittelwert:        {z.mean():.6f} cm\n")
        f.write(f"  Median:            {np.median(z):.6f} cm\n")
        f.write(f"  Standardabw.:      {z.std():.6f} cm\n\n")

        f.write(f"  25% Perzentil:     {np.percentile(z, 25):.6f} cm\n")
        f.write(f"  75% Perzentil:     {np.percentile(z, 75):.6f} cm\n")
        f.write(f"  95% Perzentil:     {np.percentile(z, 95):.6f} cm\n")
        f.write(f"  99% Perzentil:     {np.percentile(z, 99):.6f} cm\n\n")

        f.write("-" * 70 + "\n")
        f.write("HÖHENVERTEILUNG\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Punkte über Null:  {np.sum(z > 0):5d} ({np.sum(z > 0) / len(z) * 100:5.1f}%)\n")
        f.write(f"  Punkte bei Null:   {np.sum(z == 0):5d} ({np.sum(z == 0) / len(z) * 100:5.1f}%)\n")
        f.write(f"  Punkte unter Null: {np.sum(z < 0):5d} ({np.sum(z < 0) / len(z) * 100:5.1f}%)\n\n")

        # Steigung
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        z_sorted = z[sorted_idx]
        dz_dx = np.gradient(z_sorted, x_sorted)

        f.write("-" * 70 + "\n")
        f.write("STEIGUNG (dZ/dX)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Minimum:           {dz_dx.min():.6f}\n")
        f.write(f"  Maximum:           {dz_dx.max():.6f}\n")
        f.write(f"  Mittelwert:        {dz_dx.mean():.6f}\n")
        f.write(f"  Median:            {np.median(dz_dx):.6f}\n")
        f.write(f"  Std.abw.:          {dz_dx.std():.6f}\n")
        f.write(f"  Max. Betrag:       {np.abs(dz_dx).max():.6f}\n\n")

        f.write("=" * 70 + "\n")

    print(f"[OK] Zusammenfassung gespeichert: {report_path}")

def plot_topography_2d(points, output_path, gaussian_params=None):
    """
    Erstellt 2D-Konturplot (Draufsicht) des rotationssymmetrischen Kraters.
    Mit Isohöhenlinien, Rim-Kreis und Kenngrößen-Box (Stil wie Hauptcode).
    """
    if gaussian_params is None:
        print(f"[WARNING] Keine Gauß-Parameter — überspringe 2D-Topografie")
        return

    fit_model = gaussian_params['fit_model']
    A_rim = gaussian_params['A_rim']
    d = gaussian_params['d']
    sigma_rim = gaussian_params['sigma_rim']
    A3 = gaussian_params['A3'] if gaussian_params['A3'] is not None else 0.0
    sigma3 = gaussian_params['sigma3'] if gaussian_params['sigma3'] is not None else 1.0
    c = gaussian_params['c']

    def radial_profile(r, A_rim, d, sigma_rim, A3, sigma3, c):
        rim = A_rim * np.exp(-(r - d) ** 2 / (2 * sigma_rim ** 2))
        center = A3 * np.exp(-r ** 2 / (2 * sigma3 ** 2))
        return np.maximum(rim + center + c, 0.0)

    x = points[:, 0]
    x_span = x.max() - x.min()
    grid_size = max(x_span * 1.5, 60.0)

    crater_center_x = grid_size / 2.0
    crater_center_y = grid_size / 2.0

    x_grid = np.linspace(0, grid_size, 500)
    y_grid = np.linspace(0, grid_size, 500)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

    R_mesh = np.sqrt((X_mesh - crater_center_x) ** 2 + (Y_mesh - crater_center_y) ** 2)
    H_mesh = radial_profile(R_mesh, A_rim, d, sigma_rim, A3, sigma3, c)

    fig, ax = plt.subplots(figsize=(10, 10))

    levels = np.linspace(H_mesh.min(), H_mesh.max(), 8)
    contourf = ax.contourf(X_mesh, Y_mesh, H_mesh, levels=levels,
                            cmap='plasma', extend='both')

    contour_lines = ax.contour(X_mesh, Y_mesh, H_mesh, levels=levels,
                                colors='black', linewidths=0.5, alpha=0.4)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f cm')

    cbar = plt.colorbar(contourf, ax=ax, label='Höhe [cm]', shrink=0.8)

    # Markierungen
    ax.plot(crater_center_x, crater_center_y, 'r+', markersize=15,
            markeredgewidth=2, label='Kratermitte')

    rim_circle = plt.Circle((crater_center_x, crater_center_y), d,
                            color='red', fill=False, linestyle='--',
                            linewidth=2, label=f'Rim-Radius: {d:.2f} cm')
    ax.add_patch(rim_circle)

    sigma_circle = plt.Circle((crater_center_x, crater_center_y),
                              d + sigma_rim,
                              color='orange', fill=False, linestyle=':',
                              linewidth=1.5, alpha=0.7,
                              label=f'Rim-Breite: {sigma_rim:.2f} cm')
    ax.add_patch(sigma_circle)

    ax.set_xlabel('X-Position [cm]', fontsize=12)
    ax.set_ylabel('Y-Position [cm]', fontsize=12)
    ax.set_title('Topografie des Kraters (rotationssymmetrisch)',
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    info_text = f"KRATER-KENNGROSSEN:\n"
    info_text += f"Mitte: ({crater_center_x:.1f}, {crater_center_y:.1f}) cm\n"
    info_text += f"Radius: {d:.2f} cm\n"
    info_text += f"Rim-Höhe: {A_rim:.3f} cm\n"
    if fit_model == 'symmetric_triple':
        info_text += f"Zentralberg: {A3:.3f} cm\n"
        info_text += f"ZB-Ratio: {A3 / A_rim:.3f}"
    else:
        info_text += f"(Kein Zentralberg)"

    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path / 'topography.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 2D-Topografie gespeichert: {output_path / 'topography.png'}")


def process_single_folder(input_dir):
    """
    Verarbeitet einen einzelnen Ordner mit measurement_points.csv und metadata.csv.

    Args:
        input_dir: Path-Objekt des Ordners

    Returns:
        Dictionary mit Ergebnissen oder None bei Fehler
    """
    print("\n" + "=" * 70)
    print(f"VERARBEITE ORDNER: {input_dir.name}")
    print("=" * 70 + "\n")

    measurement_csv = input_dir / 'measurement_points.csv'
    if not measurement_csv.exists():
        print(f"[ERROR] measurement_points.csv nicht gefunden in: {input_dir}")
        return None

    metadata_csv = input_dir / 'metadata.csv'
    metadata = None

    if metadata_csv.exists():
        print(f"[INFO] Lade Metadaten: {metadata_csv}")
        try:
            metadata = load_metadata(metadata_csv)
            print(f"[OK] Metadaten geladen")
        except Exception as e:
            print(f"[WARNING] Fehler beim Laden der Metadaten: {e}")
    else:
        print(f"[INFO] Keine metadata.csv gefunden (optional)")

    print(f"[INFO] Lade Messpunkte: {measurement_csv}")
    try:
        points = load_measurement_points(measurement_csv)
        print(f"[OK] {len(points)} Punkte geladen")
    except Exception as e:
        print(f"[ERROR] Fehler beim Laden der Messpunkte: {e}")
        return None

    if metadata:
        print_metadata_info(metadata)

    # Tank-Höhe aus Metadaten
    tank_height = 0
    if metadata:
        try:
            tank_height = float(metadata.get('tank_height_z_cm', 0))
        except (ValueError, TypeError):
            tank_height = 0

    # Preprocessing
    points = clean_measurement_points(points, metadata)

    # Plots erstellen
    output_dir = input_dir / 'plots'
    output_dir.mkdir(exist_ok=True)
    print(f"[INFO] Output-Ordner: {output_dir}\n")

    print("Erstelle Plots...")
    print("-" * 70)

    try:
        plot_3d_points(points, output_dir)
        plot_3d_surface(points, output_dir)
        plot_height_profile(points, output_dir)
        plot_height_profile_simple(points, output_dir)
        plot_3d_slope_analysis(points, output_dir)
        gaussian_params = plot_gaussian_fit(points, output_dir, metadata)
        plot_topography_2d(points, output_dir, gaussian_params)
        plot_topography_3d(points, output_dir, gaussian_params, tank_height)
        create_summary_report_with_metadata(points, metadata, output_dir)

        print("-" * 70)
        print(f"\n[OK] Alle Plots erstellt in: {output_dir}")
        print("\nErstellt:")
        print("  - 3d_scatter.png")
        print("  - 3d_surface.png")
        print("  - height_profile.png")
        print("  - height_profile_simple.png")
        print("  - 3d_slope_analysis.png")
        print("  - gaussian_fit.png")
        print("  - gaussian_fit_params.txt")
        print("  - topography.png")
        print("  - topography_3d.png")
        print("  - summary_report.txt")

        # Berechne Krater-Kenngrößen
        crater_metrics = compute_crater_metrics(gaussian_params, points)

        # Parse Ordnernamen
        folder_params = parse_folder_name(input_dir.name)

        # Erstelle Ergebnis-Dictionary
        result = {
            'folder_name': input_dir.name,
            'grain_size': folder_params['grain_size'] if folder_params else None,
            'volume_ml': folder_params['volume_ml'] if folder_params else None,
            'fill_height_cm': folder_params['fill_height_cm'] if folder_params else None,
            'h': crater_metrics['h'],
            'd_outer': crater_metrics['d_outer'],
            'r_inner': crater_metrics['r_inner'],
            'A_rim': gaussian_params['A_rim'] if gaussian_params else None,
            'd_fit': gaussian_params['d'] if gaussian_params else None,
            'sigma_rim': gaussian_params['sigma_rim'] if gaussian_params else None,
            'mu_center': gaussian_params['mu3'] if gaussian_params else None,
            'rms_error': None  # RMS-Error muss aus plot_gaussian_fit extrahiert werden
        }

        return result

    except Exception as e:
        print(f"[ERROR] Fehler beim Erstellen der Plots: {e}")
        return None


def main():
    print("\n" + "=" * 70)
    print("LASER-MESSPUNKTE PLOT-GENERATOR")
    print("=" * 70 + "\n")

    input_path = Path(INPUT_PATH)

    if not input_path.exists():
        print(f"[ERROR] Pfad nicht gefunden: {input_path}")
        sys.exit(1)

    if BATCH_MODE:
        # Batch-Modus: Suche nach allen Ordnern mit _results am Ende
        print(f"[INFO] Batch-Modus: Suche nach *_results Ordnern in: {input_path}")

        if not input_path.is_dir():
            print(f"[ERROR] Pfad ist kein Ordner: {input_path}")
            sys.exit(1)

        # Finde alle Ordner die mit _results enden
        result_folders = [f for f in input_path.iterdir() if f.is_dir() and f.name.endswith('_results')]

        if len(result_folders) == 0:
            print(f"[WARNING] Keine Ordner mit '_results' Endung gefunden in: {input_path}")
            sys.exit(1)

        print(f"[INFO] Gefunden: {len(result_folders)} Ordner")
        print("-" * 70)

        success_count = 0
        fail_count = 0
        results_list = []

        for folder in sorted(result_folders):
            result = process_single_folder(folder)
            if result is not None:
                success_count += 1
                results_list.append(result)
            else:
                fail_count += 1
            print("\n")

        # Speichere Batch-Ergebnisse
        if len(results_list) > 0:
            save_batch_results_csv(results_list, input_path)

        print("=" * 70)
        print("BATCH-VERARBEITUNG ABGESCHLOSSEN")
        print("=" * 70)
        print(f"Erfolgreich: {success_count} Ordner")
        print(f"Fehlgeschlagen: {fail_count} Ordner")
        print("=" * 70 + "\n")

    else:
        # Einzelner Ordner-Modus
        print(f"[INFO] Einzelordner-Modus: {input_path}")

        if not input_path.is_dir():
            print(f"[ERROR] Pfad ist kein Ordner: {input_path}")
            sys.exit(1)

        result = process_single_folder(input_path)

        if result is not None:
            # Speichere auch im Einzelordner-Modus eine CSV (mit einer Zeile)
            save_batch_results_csv([result], input_path.parent)

            print("\n" + "=" * 70)
            print("FERTIG!")
            print("=" * 70 + "\n")
        else:
            print("\n" + "=" * 70)
            print("FEHLER BEI DER VERARBEITUNG")
            print("=" * 70 + "\n")
            sys.exit(1)


if __name__ == "__main__":
    import numpy as np

    main()
