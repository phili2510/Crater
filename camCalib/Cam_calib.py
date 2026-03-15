"""
GoPro Hero 13 – Manuelle Entzerrung (Linear-Modus, 4K)
=======================================================
Öffnet ein Bild und zeigt Schieberegler, mit denen du die
Verzerrungskoeffizienten (k1, k2, k3, p1, p2) live anpassen kannst.
Wenn das Ergebnis passt: Taste 's' → speichert das entzerrte Bild.
Taste 'q' oder ESC → beendet das Programm.

Nutzung in PyCharm:
    1. Passe IMAGE_PATH unten an (Pfad zu deinem GoPro-Foto).
    2. Starte das Skript.
    3. Schieberegler bewegen, bis gerade Linien im Bild wirklich gerade sind.
    4. 's' drücken zum Speichern, 'p' zum Drucken der aktuellen Parameter.

Tipp: Nimm ein Foto auf, das gerade Linien enthält (Türrahmen, Fliesen,
      Lineal, Tischkante), damit du die Verzerrung gut beurteilen kannst.
"""

import cv2
import numpy as np
import sys
import os

# ============================================================
# KONFIGURATION – hier anpassen
# ============================================================
IMAGE_PATH = r"C:\Users\PCUser\PycharmProjects\Crater\GP011281.JPG"  # <-- Pfad zu deinem GoPro-Foto

# Bildauflösung (wird automatisch erkannt, aber hier als Fallback)
IMG_W = 3840
IMG_H = 2160

# Startwerte für die Verzerrungskoeffizienten
# (gute Ausgangswerte für GoPro Hero 13 im Linear-Modus bei 4K)
# k1: Hauptverzerrung (negativ = tonnenförmig, positiv = kissenförmig)
# k2: Feinkorrektur höherer Ordnung
# k3: noch feinere Korrektur
# p1, p2: tangentiale Verzerrung (meist nahe 0)
INITIAL_K1 = -0.12   # Typischer Startwert für Linear-Modus
INITIAL_K2 = 0.02
INITIAL_K3 = 0.0
INITIAL_P1 = 0.0
INITIAL_P2 = 0.0

# Skalierung der Vorschau (4K ist zu groß für die meisten Monitore)
PREVIEW_SCALE = 0.35  # 35% → ca. 1344x756

# ============================================================
# SCHIEBEREGLER-KONFIGURATION
# ============================================================
# Schieberegler gehen von 0–2000, Mitte (1000) = 0.0
# Bereich: k1 von -0.50 bis +0.50
#           k2 von -0.20 bis +0.20
#           k3 von -0.10 bis +0.10
#           p1, p2 von -0.01 bis +0.01

SLIDER_CENTER = 1000
SLIDER_MAX = 2000

K1_RANGE = 0.50   # ±0.50
K2_RANGE = 0.20   # ±0.20
K3_RANGE = 0.10   # ±0.10
P_RANGE = 0.01    # ±0.01


def slider_to_value(slider_val, value_range):
    """Wandelt Schieberegler-Position (0–2000) in Float-Wert um."""
    return (slider_val - SLIDER_CENTER) / SLIDER_CENTER * value_range


def value_to_slider(value, value_range):
    """Wandelt Float-Wert in Schieberegler-Position um."""
    return int(SLIDER_CENTER + (value / value_range) * SLIDER_CENTER)


def undistort_image(img, k1, k2, k3, p1, p2):
    """Entzerrt das Bild mit den gegebenen Koeffizienten."""
    h, w = img.shape[:2]

    # Kameramatrix: Brennweite geschätzt für 4K GoPro Linear-Modus
    # fx ≈ fy ≈ Bildbreite (typisch für ~80° FOV nach Linear-Korrektur)
    fx = w * 0.85
    fy = fx
    cx = w / 2.0
    cy = h / 2.0

    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)

    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    # Optimale neue Kameramatrix (alpha=1 behält alle Pixel)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0.5
    )

    # Entzerren
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Optional: Auf ROI zuschneiden (entfernt schwarze Ränder)
    # x, y, rw, rh = roi
    # if rw > 0 and rh > 0:
    #     undistorted = undistorted[y:y+rh, x:x+rw]

    return undistorted


def draw_grid_overlay(img, step=100, color=(0, 255, 0), alpha=0.3):
    """Zeichnet ein Hilfsraster über das Bild (zum Prüfen gerader Linien)."""
    overlay = img.copy()
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), color, 1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def nothing(x):
    """Dummy-Callback für Trackbars."""
    pass


def main():
    # Bild laden
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = IMAGE_PATH

    if not os.path.exists(image_path):
        print(f"FEHLER: Bild nicht gefunden: {image_path}")
        print(f"Bitte passe IMAGE_PATH im Skript an oder übergib den Pfad als Argument:")
        print(f"  python gopro_undistort.py pfad/zum/bild.jpg")
        sys.exit(1)

    print(f"Lade Bild: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("FEHLER: Bild konnte nicht geladen werden.")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Bildgröße: {w}x{h}")

    # Fenster erstellen
    window_name = "GoPro Entzerrung (s=Speichern, g=Gitter, p=Parameter, q=Beenden)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    preview_w = int(w * PREVIEW_SCALE)
    preview_h = int(h * PREVIEW_SCALE)
    cv2.resizeWindow(window_name, preview_w, preview_h + 200)

    # Schieberegler erstellen
    cv2.createTrackbar("k1", window_name, value_to_slider(INITIAL_K1, K1_RANGE), SLIDER_MAX, nothing)
    cv2.createTrackbar("k2", window_name, value_to_slider(INITIAL_K2, K2_RANGE), SLIDER_MAX, nothing)
    cv2.createTrackbar("k3", window_name, value_to_slider(INITIAL_K3, K3_RANGE), SLIDER_MAX, nothing)
    cv2.createTrackbar("p1", window_name, value_to_slider(INITIAL_P1, P_RANGE), SLIDER_MAX, nothing)
    cv2.createTrackbar("p2", window_name, value_to_slider(INITIAL_P2, P_RANGE), SLIDER_MAX, nothing)

    show_grid = False

    print("\n=== STEUERUNG ===")
    print("  Schieberegler: Verzerrungsparameter anpassen")
    print("  g: Hilfsraster ein/aus")
    print("  s: Entzerrtes Bild speichern (volle Auflösung)")
    print("  p: Aktuelle Parameter in Konsole drucken")
    print("  r: Auf Startwerte zurücksetzen")
    print("  q/ESC: Beenden\n")

    while True:
        # Parameter von Schiebereglern lesen
        k1 = slider_to_value(cv2.getTrackbarPos("k1", window_name), K1_RANGE)
        k2 = slider_to_value(cv2.getTrackbarPos("k2", window_name), K2_RANGE)
        k3 = slider_to_value(cv2.getTrackbarPos("k3", window_name), K3_RANGE)
        p1 = slider_to_value(cv2.getTrackbarPos("p1", window_name), P_RANGE)
        p2 = slider_to_value(cv2.getTrackbarPos("p2", window_name), P_RANGE)

        # Entzerren
        result = undistort_image(img, k1, k2, k3, p1, p2)

        # Vorschau skalieren
        preview = cv2.resize(result, (preview_w, preview_h))

        # Gitter anzeigen
        if show_grid:
            grid_step = int(80 * PREVIEW_SCALE)
            preview = draw_grid_overlay(preview, step=max(grid_step, 30))

        # Parameterwerte als Text einblenden
        info = f"k1={k1:.4f}  k2={k2:.4f}  k3={k3:.4f}  p1={p1:.5f}  p2={p2:.5f}"
        cv2.putText(preview, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(window_name, preview)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:  # q oder ESC
            break

        elif key == ord('s'):
            # In voller Auflösung speichern
            full_result = undistort_image(img, k1, k2, k3, p1, p2)
            base, ext = os.path.splitext(image_path)
            out_path = f"{base}_entzerrt{ext}"
            cv2.imwrite(out_path, full_result)
            print(f"\n>>> Gespeichert: {out_path}")
            print(f"    Parameter: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}, p1={p1:.6f}, p2={p2:.6f}")

        elif key == ord('g'):
            show_grid = not show_grid
            print(f"Gitter: {'AN' if show_grid else 'AUS'}")

        elif key == ord('p'):
            print(f"\n--- Aktuelle Parameter ---")
            print(f"k1 = {k1:.6f}")
            print(f"k2 = {k2:.6f}")
            print(f"k3 = {k3:.6f}")
            print(f"p1 = {p1:.6f}")
            print(f"p2 = {p2:.6f}")
            print(f"--------------------------")
            print(f"\nZum Kopieren:")
            print(f"dist_coeffs = np.array([{k1:.6f}, {k2:.6f}, {p1:.6f}, {p2:.6f}, {k3:.6f}])")

        elif key == ord('r'):
            # Auf Startwerte zurücksetzen
            cv2.setTrackbarPos("k1", window_name, value_to_slider(INITIAL_K1, K1_RANGE))
            cv2.setTrackbarPos("k2", window_name, value_to_slider(INITIAL_K2, K2_RANGE))
            cv2.setTrackbarPos("k3", window_name, value_to_slider(INITIAL_K3, K3_RANGE))
            cv2.setTrackbarPos("p1", window_name, value_to_slider(INITIAL_P1, P_RANGE))
            cv2.setTrackbarPos("p2", window_name, value_to_slider(INITIAL_P2, P_RANGE))
            print("Zurückgesetzt auf Startwerte.")

    cv2.destroyAllWindows()
    print("Fertig.")


if __name__ == "__main__":
    main()