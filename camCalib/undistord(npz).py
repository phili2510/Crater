#!/usr/bin/env python3
"""
Bildentzerrung mit Kamera-Kalibrierungsdaten

Verwendung:
  1. Kommandozeile: python entzerren.py calibration.npz foto.jpg
  2. Variablen anpassen: NPZ_PATH und IMAGE_PATH unten setzen
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# ============================================================================
# VARIABLEN ZUM ANPASSEN (für direkte Ausführung ohne Kommandozeilenargumente)
# ============================================================================
NPZ_PATH = r"C:\Users\PCUser\PycharmProjects\Crater\camCalib\camera_calib(1).npz"
IMAGE_PATH = r"C:\Users\PCUser\PycharmProjects\Crater\GP011281.JPG"


def entzerren(npz_path, image_path):
    """
    Entzerrt ein Bild mit Kalibrierungsdaten aus einer .npz-Datei

    Args:
        npz_path: Pfad zur .npz-Datei mit camera_matrix und dist_coeffs
        image_path: Pfad zum zu entzerrenden Bild
    """

    # .npz-Datei laden
    print(f"\n{'=' * 60}")
    print(f"Lade Kalibrierungsdaten aus: {npz_path}")
    print(f"{'=' * 60}")

    try:
        calib_data = np.load(npz_path)
        camera_matrix = calib_data['camera_matrix']
        dist_coeffs = calib_data['dist_coeffs']
    except FileNotFoundError:
        print(f"FEHLER: Datei '{npz_path}' nicht gefunden!")
        sys.exit(1)
    except KeyError as e:
        print(f"FEHLER: Erforderlicher Schlüssel {e} nicht in .npz-Datei gefunden!")
        sys.exit(1)

    print("\nKameramatrix (3x3):")
    print(camera_matrix)
    print("\nVerzerrungskoeffizienten [k1, k2, p1, p2, k3]:")
    print(dist_coeffs)

    # Bild laden
    print(f"\n{'=' * 60}")
    print(f"Lade Bild: {image_path}")
    print(f"{'=' * 60}")

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Bild konnte nicht geladen werden")
    except Exception as e:
        print(f"FEHLER: Bild '{image_path}' konnte nicht geladen werden!")
        print(f"Details: {e}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"\nBildgröße: {w} x {h} Pixel")

    # Optimale neue Kameramatrix berechnen
    print(f"\n{'=' * 60}")
    print("Berechne optimale Kameramatrix (alpha=0)...")
    print(f"{'=' * 60}")

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        alpha=1,
        newImgSize=(w, h)
    )

    print("\nNeue Kameramatrix:")
    print(new_camera_matrix)
    print(f"\nROI (Region of Interest): x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    # Bild entzerren
    print(f"\n{'=' * 60}")
    print("Entzerren des Bildes...")
    print(f"{'=' * 60}")

    undistorted_image = cv2.undistort(
        image,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix
    )

    # Ausgabepfad erstellen
    input_path = Path(image_path)
    output_path = input_path.parent / f"{input_path.stem}_entzerrt{input_path.suffix}"

    # Entzerrtes Bild speichern
    cv2.imwrite(str(output_path), undistorted_image)

    print(f"\n{'=' * 60}")
    print(f"✓ Entzerrtes Bild gespeichert: {output_path}")
    print(f"{'=' * 60}\n")


def main():
    """Hauptfunktion mit Kommandozeilen-Unterstützung"""

    # Prüfe, ob Kommandozeilenargumente übergeben wurden
    if len(sys.argv) == 3:
        # Kommandozeilenargumente verwenden
        npz_path = sys.argv[1]
        image_path = sys.argv[2]
        print("\nModus: Kommandozeilenargumente")
    elif len(sys.argv) == 1:
        # Variablen aus dem Skript verwenden
        npz_path = NPZ_PATH
        image_path = IMAGE_PATH
        print("\nModus: Vordefinierte Variablen")
        print(f"  NPZ_PATH = '{NPZ_PATH}'")
        print(f"  IMAGE_PATH = '{IMAGE_PATH}'")
    else:
        # Falsche Anzahl an Argumenten
        print("\nFEHLER: Falsche Anzahl an Argumenten!")
        print("\nVerwendung:")
        print("  1. Kommandozeile:")
        print("     python entzerren.py <pfad_zur_npz_datei> <pfad_zum_bild>")
        print("     Beispiel: python entzerren.py calibration.npz foto.jpg")
        print("\n  2. Variablen im Skript anpassen:")
        print("     NPZ_PATH und IMAGE_PATH am Anfang des Skripts setzen")
        print("     Dann einfach ausführen: python entzerren.py")
        sys.exit(1)

    # Entzerrung durchführen
    entzerren(npz_path, image_path)


if __name__ == "__main__":
    main()
