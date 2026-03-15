#!/usr/bin/env python3
"""
Kamera-Kalibrierung mit Schachbrettmuster
==========================================
Liest alle Bilder aus einem Ordner, erkennt das Schachbrettmuster,
berechnet Kameramatrix + Verzerrungskoeffizienten und speichert
alles in einer .npz-Datei.

Verwendung:
    1. Pfade unten anpassen (IMAGE_DIR, OUTPUT_NPZ)
    2. Schachbrett-Größe prüfen (CHECKERBOARD_ROWS/COLS, SQUARE_SIZE_MM)
    3. Skript starten: python kamera_kalibrierung.py

Oder per Kommandozeile:
    python kamera_kalibrierung.py pfad/zum/ordner

Ausgabe:
    - camera_calib.npz (Kameramatrix + Distortion Coefficients)
    - Vorher/Nachher-Vergleichsbild
    - Zusammenfassung aller erkannten Bilder
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
import glob
import time

# ============================================================
# KONFIGURATION – hier anpassen
# ============================================================

# Ordner mit den Schachbrett-Fotos
IMAGE_DIR = r"C:\Users\PCUser\PycharmProjects\Crater\camCalib\Cam_calib(2)"

# Ausgabe-Datei
OUTPUT_NPZ = r"C:\Users\PCUser\PycharmProjects\Crater\camCalib\camera_calib(1).npz"

# ----------------------------------------------------------
# SCHACHBRETT-KONFIGURATION
# ----------------------------------------------------------
# WICHTIG: Anzahl der INNEREN Ecken, nicht der Felder!
# Ein Standard-Schachbrett mit 10x7 Feldern hat 9x6 innere Ecken.
# Zähle die Kreuzungspunkte in einer Reihe/Spalte.
CHECKERBOARD_COLS = 39   # Innere Ecken horizontal
CHECKERBOARD_ROWS = 27   # Innere Ecken vertikal

# Größe eines Feldes in mm (für korrekte Skalierung)
# Muss nur stimmen, wenn du echte Maße brauchst.
# Für reine Entzerrung ist der Wert egal.
SQUARE_SIZE_MM = 20

# ----------------------------------------------------------
# ERKENNUNGS-PARAMETER
# ----------------------------------------------------------
# Unterstützte Bildformate
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.JPG", "*.JPEG", "*.PNG")

# Subpixel-Verfeinerung (verbessert Genauigkeit erheblich)
SUBPIXEL_WINSIZE = (11, 11)      # Suchfenster für Subpixel-Verfeinerung
SUBPIXEL_ZERO_ZONE = (-1, -1)    # Kein Dead-Zone
SUBPIXEL_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# findChessboardCorners Flags
CHESS_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH +
    cv2.CALIB_CB_NORMALIZE_IMAGE +
    cv2.CALIB_CB_FAST_CHECK
)

# calibrateCamera Flags (0 = Standard, alle Parameter frei)
CALIBRATE_FLAGS = 0

# Vorschau während der Erkennung anzeigen?
SHOW_PREVIEW = True
PREVIEW_WAIT_MS = 500  # Wie lange jedes erkannte Bild angezeigt wird (ms)

# Mindestanzahl gültiger Bilder für Kalibrierung
MIN_VALID_IMAGES = 4


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def find_images(image_dir):
    """Findet alle Bilddateien im angegebenen Ordner."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"FEHLER: Ordner nicht gefunden: {image_dir}")
        sys.exit(1)

    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(image_dir.glob(ext))

    # Sortieren für konsistente Reihenfolge
    files = sorted(set(files))

    if len(files) == 0:
        print(f"FEHLER: Keine Bilddateien in {image_dir} gefunden!")
        print(f"        Unterstützte Formate: {', '.join(IMAGE_EXTENSIONS)}")
        sys.exit(1)

    return files


def detect_chessboard(image_path, checkerboard_size, show_preview=False):
    """
    Erkennt Schachbrettecken in einem Bild.

    Returns:
        (corners, gray_shape) oder (None, None) bei Fehler
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [SKIP] Konnte nicht geladen werden: {image_path.name}")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Schachbrett suchen
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, CHESS_FLAGS)

    if not ret:
        print(f"  [FAIL] Kein Schachbrett erkannt: {image_path.name}")
        return None, None

    # Subpixel-Verfeinerung für höhere Genauigkeit
    corners_refined = cv2.cornerSubPix(
        gray, corners,
        SUBPIXEL_WINSIZE,
        SUBPIXEL_ZERO_ZONE,
        SUBPIXEL_CRITERIA
    )

    print(f"  [ OK ] Erkannt: {image_path.name} ({gray.shape[1]}x{gray.shape[0]})")

    # Vorschau anzeigen
    if show_preview:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, checkerboard_size, corners_refined, ret)

        # Skalieren für Anzeige
        max_dim = 1200
        h, w = vis.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

        cv2.imshow("Schachbrett-Erkennung", vis)
        cv2.waitKey(PREVIEW_WAIT_MS)

    return corners_refined, gray.shape[::-1]  # (width, height)


def calibrate(obj_points, img_points, image_size):
    """
    Führt die eigentliche Kamerakalibrierung durch.

    Returns:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs
    """
    print(f"\nKalibriere mit {len(obj_points)} Bildern...")
    print(f"Bildgröße: {image_size[0]} x {image_size[1]} px")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
        flags=CALIBRATE_FLAGS
    )

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def calculate_reprojection_errors(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    """Berechnet den Reprojektionsfehler pro Bild."""
    errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i],
            camera_matrix, dist_coeffs
        )
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        errors.append(error)
    return errors


def save_calibration(output_path, camera_matrix, dist_coeffs, image_size, reprojection_error,
                     num_images, checkerboard_size, square_size):
    """Speichert Kalibrierungsdaten als .npz-Datei."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(output_path),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        reprojection_error=np.array([reprojection_error]),
        num_images=np.array([num_images]),
        checkerboard_size=np.array(checkerboard_size),
        square_size_mm=np.array([square_size])
    )

    print(f"\n[OK] Kalibrierung gespeichert: {output_path}")
    print(f"     Enthaltene Keys: camera_matrix, dist_coeffs, image_size,")
    print(f"                      reprojection_error, num_images, checkerboard_size, square_size_mm")


def save_comparison_image(output_dir, sample_image_path, camera_matrix, dist_coeffs):
    """Erstellt ein Vorher/Nachher-Vergleichsbild."""
    img = cv2.imread(str(sample_image_path))
    if img is None:
        return

    h, w = img.shape[:2]

    # Entzerren mit alpha=1 (alle Pixel behalten)
    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam_mtx)

    # Nebeneinander (skaliert)
    max_w = 1800
    scale = min(1.0, max_w / (w * 2))
    h_s, w_s = int(h * scale), int(w * scale)

    left = cv2.resize(img, (w_s, h_s))
    right = cv2.resize(undistorted, (w_s, h_s))

    # Beschriftung
    cv2.putText(left, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(right, "ENTZERRT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    comparison = np.hstack([left, right])

    comp_path = Path(output_dir) / "calibration_comparison.jpg"
    cv2.imwrite(str(comp_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[OK] Vergleichsbild gespeichert: {comp_path}")


def print_results(camera_matrix, dist_coeffs, reprojection_error, per_image_errors, image_files_used):
    """Gibt die Ergebnisse formatiert aus."""
    print(f"\n{'=' * 60}")
    print("KALIBRIERUNGS-ERGEBNISSE")
    print(f"{'=' * 60}")

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print(f"\nKameramatrix:")
    print(f"  Brennweite:  fx = {fx:.2f} px,  fy = {fy:.2f} px")
    print(f"  Bildzentrum: cx = {cx:.2f} px,  cy = {cy:.2f} px")
    print(f"\n  Vollständig:")
    print(f"  {camera_matrix}")

    k1, k2, p1, p2, k3 = dist_coeffs.flatten()[:5]
    print(f"\nVerzerrungskoeffizienten:")
    print(f"  k1 = {k1:.6f}  (radial, Hauptverzerrung)")
    print(f"  k2 = {k2:.6f}  (radial, höhere Ordnung)")
    print(f"  p1 = {p1:.6f}  (tangential)")
    print(f"  p2 = {p2:.6f}  (tangential)")
    print(f"  k3 = {k3:.6f}  (radial, 6. Ordnung)")

    print(f"\nReprojektionsfehler (gesamt): {reprojection_error:.4f} px")

    if reprojection_error < 0.5:
        quality = "AUSGEZEICHNET"
    elif reprojection_error < 1.0:
        quality = "GUT"
    elif reprojection_error < 2.0:
        quality = "AKZEPTABEL"
    else:
        quality = "SCHLECHT - Kalibrierung wiederholen!"

    print(f"Qualität: {quality}")

    print(f"\nFehler pro Bild:")
    for i, (err, path) in enumerate(zip(per_image_errors, image_files_used), 1):
        marker = " <<<" if err > reprojection_error * 2 else ""
        print(f"  {i:2d}. {path.name:30s}  Fehler: {err:.4f} px{marker}")

    worst_idx = int(np.argmax(per_image_errors))
    print(f"\n  Schlechtestes Bild: {image_files_used[worst_idx].name} ({per_image_errors[worst_idx]:.4f} px)")
    print(f"  Bestes Bild:       {image_files_used[int(np.argmin(per_image_errors))].name} ({min(per_image_errors):.4f} px)")


def main():
    """Hauptfunktion."""

    # Ordner bestimmen
    if len(sys.argv) >= 2:
        image_dir = sys.argv[1]
        output_npz = sys.argv[2] if len(sys.argv) >= 3 else str(Path(image_dir) / "camera_calib.npz")
    else:
        image_dir = IMAGE_DIR
        output_npz = OUTPUT_NPZ

    print(f"\n{'=' * 60}")
    print("KAMERA-KALIBRIERUNG MIT SCHACHBRETTMUSTER")
    print(f"{'=' * 60}")
    print(f"\nBildordner:  {image_dir}")
    print(f"Ausgabe:     {output_npz}")
    print(f"Schachbrett: {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS} innere Ecken")
    print(f"Feldgröße:   {SQUARE_SIZE_MM} mm")

    # Bilder finden
    image_files = find_images(image_dir)
    print(f"\n{len(image_files)} Bilder gefunden.\n")

    # Schachbrett-Konfiguration
    checkerboard_size = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)

    # 3D-Punkte im Weltkoordinatensystem vorbereiten
    # Z=0, da das Schachbrett flach ist
    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM  # Skalierung auf reale Maße

    # Sammlungen für alle erkannten Bilder
    obj_points = []   # 3D-Punkte (gleich für jedes Bild)
    img_points = []   # 2D-Punkte (erkannte Ecken pro Bild)
    image_size = None
    image_files_used = []

    # Schachbrett in jedem Bild suchen
    print("Suche Schachbrettmuster...")
    if SHOW_PREVIEW:
        cv2.namedWindow("Schachbrett-Erkennung", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Schachbrett-Erkennung", 1000, 700)

    for img_path in image_files:
        corners, img_sz = detect_chessboard(img_path, checkerboard_size, SHOW_PREVIEW)

        if corners is not None:
            obj_points.append(objp)
            img_points.append(corners)
            image_files_used.append(img_path)

            if image_size is None:
                image_size = img_sz
            elif img_sz != image_size:
                print(f"  [WARN] Bildgröße {img_sz} weicht ab von {image_size}!")

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    # Prüfen ob genug Bilder erkannt wurden
    print(f"\n{len(img_points)} von {len(image_files)} Bildern erfolgreich erkannt.")

    if len(img_points) < MIN_VALID_IMAGES:
        print(f"\nFEHLER: Mindestens {MIN_VALID_IMAGES} gültige Bilder benötigt!")
        print("Tipps:")
        print(f"  - Stimmt CHECKERBOARD_COLS={CHECKERBOARD_COLS} x CHECKERBOARD_ROWS={CHECKERBOARD_ROWS}?")
        print("    (Zähle die INNEREN Ecken, nicht die Felder!)")
        print("  - Sind die Bilder scharf genug?")
        print("  - Ist das Muster vollständig sichtbar?")
        print("  - Ist die Beleuchtung gleichmäßig?")
        sys.exit(1)

    # Kalibrierung durchführen
    start_time = time.time()
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate(
        obj_points, img_points, image_size
    )
    elapsed = time.time() - start_time
    print(f"Kalibrierung abgeschlossen in {elapsed:.1f}s")

    # Reprojektionsfehler berechnen
    per_image_errors = calculate_reprojection_errors(
        obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs
    )

    # Ergebnisse ausgeben
    print_results(camera_matrix, dist_coeffs, ret, per_image_errors, image_files_used)

    # Speichern
    save_calibration(
        output_npz, camera_matrix, dist_coeffs,
        image_size, ret, len(img_points),
        checkerboard_size, SQUARE_SIZE_MM
    )

    # Vergleichsbild erstellen
    output_dir = Path(output_npz).parent
    save_comparison_image(output_dir, image_files_used[0], camera_matrix, dist_coeffs)

    print(f"\n{'=' * 60}")
    print("FERTIG!")
    print(f"{'=' * 60}")
    print(f"\nNächster Schritt: Verwende '{Path(output_npz).name}' in deinem Entzerrungs-Skript.")


if __name__ == "__main__":
    main()