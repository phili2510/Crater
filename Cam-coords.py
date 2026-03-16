from __future__ import annotations

import cv2
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
import platform
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
from scipy.interpolate import interp1d

# =======================
# EINGABEN
# =======================
CALIBRATION_IMAGE_PATH = r'C:\Users\PCUser\PycharmProjects\Crater\Measure1\GP011329.JPG'
MEASUREMENT_IMAGE_PATH = r'C:\Users\PCUser\PycharmProjects\Crater\Measure1\GP011330.JPG'  # NEUES MESSBILD

# =======================
# KAMERA-KALIBRIERUNG (LINSENENTZERRUNG)
# =======================
# Pfad zur NPZ-Datei mit Kamera-Kalibrierungsdaten (camera_matrix, dist_coeffs)
# None = keine Entzerrung
CAMERA_CALIBRATION_NPZ = r'C:\Users\PCUser\PycharmProjects\Crater\camCalib\camera_calib(1).npz'  # z.B. r'C:\Users\PCUser\camera_calibration.npz'

# Alpha-Parameter für cv2.getOptimalNewCameraMatrix()
# 0.0 = alle schwarzen Pixel entfernen, 1.0 = alle Pixel behalten
UNDISTORT_ALPHA = 1.0

# =======================
# LICHTBRECHUNG
# =======================
N_AIR = 1.0  # Brechungsindex Luft
N_WATER = 1.333  # Brechungsindex Wasser

# =======================
# REALE KOORDINATEN DER KALIBRIERPUNKTE (3D-Weltkoordinaten)
# =======================
REAL_POINTS = np.array([
    [0.0, 60.8, 36.0],
    [60.8, 60.8, 36.0],
    [60.8, 60.8, 0.0],
    [0.0, 60.8, 0.0],
], dtype=float)

# =======================
# VORDEFINIERTE WERTE (None = manuelle Auswahl)
# =======================
# Kameraposition [x, y, z] in cm oder None für manuelle Kalibrierung
PREDEFINED_CAMERA_POSITION = None  # Setze auf None um Brechungskorrektur zu testen

# ROI-Quad: 4 Punkte [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] oder None
PREDEFINED_ROI_QUAD = np.array([[361, 1326], [5290, 1184], [5494, 3843], [500, 4439]], dtype=np.float32)

# Boden-ROI-Quad: 4 Punkte [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] oder None
PREDEFINED_FLOOR_QUAD = np.array([[1253, 1847], [3198, 1875], [4957, 2879], [35, 2887]], dtype=np.float32)

# =======================
# AQUARIUM-GEOMETRIE
# =======================
TANK_WIDTH_X = 60.8
TANK_DEPTH_Y = 60.8
TANK_HEIGHT_Z = 36.0

# =======================
# ROI-KONFIGURATION
# =======================
DESIRED_RATIO = TANK_WIDTH_X / TANK_HEIGHT_Z

# =======================
# BLINDSPOT-EINSTELLUNGEN
# =======================
AUTO_BLINDSPOT_ENABLED = True
AUTO_BLINDSPOT_TOP_PERCENT = 5.0
AUTO_EDGE_BLINDSPOT_ENABLED = True
AUTO_EDGE_BLINDSPOT_PERCENT = 1.0

# =======================
# VERTIKALE LINIEN EINSTELLUNGEN
# =======================
NUM_VERTICAL_LINES = 500  # Anzahl der vertikalen Linien
PIXEL_MARKER_SIZE = 5  # Größe der Markierung für roteste Pixel

# =======================
# LASER-LINIEN-REKONSTRUKTION
# =======================
NUM_LASER_POINTS = 200  # Anzahl der Punkte auf der gefitteten Laserlinie (Calibration)
NUM_DENSE_SAMPLES = 1000  # Anzahl der dicht gesampelten Punkte auf der Polyline (Measurement)
SLOPE_THRESHOLD = 0.5  # Maximale erlaubte Steigung (Δy/Δx) zwischen benachbarten Punkten

# Fit-Methode für Calibration-Laserlinie: 'none', 'line', 'parabola'
# 'none' = Originallinie (nur Ausreißer-Filterung + Interpolation)
# 'line' = Gerade fitten (cv2.fitLine)
# 'parabola' = Parabel fitten (Polynom 2. Grades, Scheitelpunkt in Mitte)
CALIBRATION_FIT_METHOD = 'parabola'  # Optionen: 'none', 'line', 'parabola'

# Fit-Ausschluss-Bereiche (in % der Boden-ROI-Größe)
# Punkte in diesen Bereichen werden beim Fitting ignoriert
FIT_EXCLUDE_CENTER_PERCENT = 20.0  # Bereich um die Mitte (X-Richtung)
FIT_EXCLUDE_EDGE_PERCENT = 20  # Bereich an den Rändern (links/rechts)

# System-Erkennung für macOS-Fixes
IS_MACOS = platform.system() == 'Darwin'
WINDOW_DELAY = 1.5 if IS_MACOS else 0.5


# ============================================================
# HILFSFUNKTIONEN - macOS-SICHER
# ============================================================

def safe_destroy_window(window_name):
    """Sicheres Schließen von OpenCV-Fenstern (macOS-kompatibel)."""
    try:
        cv2.destroyWindow(window_name)
        for _ in range(3):
            cv2.waitKey(1)
        if IS_MACOS:
            time.sleep(0.5)
            cv2.waitKey(1)
    except:
        pass


def safe_create_window(window_name, width=1280, height=720):
    """Sicheres Erstellen von OpenCV-Fenstern (macOS-kompatibel)."""
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        if IS_MACOS:
            cv2.waitKey(100)
    except Exception as e:
        print(f"[WARNING] Fenster-Erstellung: {e}")


def visualize_combined_3d_scene(calibration_data, measurement_data, output_path,
                                show_refraction_comparison=True):
    """
    Erstellt eine umfassende 3D-Visualisierung mit:
    - Projizierten Punkten (Calibration & Measurement)
    - Echten Punkten (Calibration & Measurement)
    - Kalibrierungslinie mit und ohne Lichtbrechung
    - Measurement-Linie

    Args:
        calibration_data: Dictionary mit Kalibrierungsdaten
        measurement_data: Dictionary mit Messdaten (kann None sein)
        output_path: Pfad zum Speichern der Visualisierung
        show_refraction_comparison: Wenn True, zeige Vergleich mit/ohne Brechung
    """
    LX = TANK_WIDTH_X
    LY = TANK_DEPTH_Y
    LZ = TANK_HEIGHT_Z

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # ============================================================
    # AQUARIUM ZEICHNEN
    # ============================================================
    corners = np.array([
        [0, 0, 0], [LX, 0, 0], [0, LY, 0], [LX, LY, 0],
        [0, 0, LZ], [LX, 0, LZ], [0, LY, LZ], [LX, LY, LZ],
    ], dtype=float)

    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]], 'b-', linewidth=1.5, alpha=0.4)

    # Boden (z=0)
    floor_corners = np.array([
        [0, 0, 0], [LX, 0, 0], [LX, LY, 0], [0, LY, 0]
    ])
    floor_poly = [[floor_corners[0], floor_corners[1], floor_corners[2], floor_corners[3]]]
    ax.add_collection3d(Poly3DCollection(floor_poly, alpha=0.15, facecolor='gray', edgecolor='black'))

    # Frontscheibe (y=0)
    front_corners = np.array([
        [0, 0, 0], [LX, 0, 0], [LX, 0, LZ], [0, 0, LZ]
    ])
    front_poly = [[front_corners[0], front_corners[1], front_corners[2], front_corners[3]]]
    ax.add_collection3d(Poly3DCollection(front_poly, alpha=0.1, facecolor='cyan',
                                         edgecolor='blue', linewidth=2))

    # Rückwand (y=LY)
    back_corners = np.array([
        [0, LY, 0], [LX, LY, 0], [LX, LY, LZ], [0, LY, LZ]
    ])
    back_poly = [[back_corners[0], back_corners[1], back_corners[2], back_corners[3]]]
    ax.add_collection3d(Poly3DCollection(back_poly, alpha=0.05, facecolor='lightblue',
                                         edgecolor='blue', linewidth=1))

    # ============================================================
    # KAMERA
    # ============================================================
    K = calibration_data['camera_position']
    ax.scatter([K[0]], [K[1]], [K[2]], s=300, c='purple', marker='*',
               label=f'Kamera ({K[0]:.1f}, {K[1]:.1f}, {K[2]:.1f})',
               edgecolors='black', linewidths=2, zorder=1000)

    # ============================================================
    # CALIBRATION: PROJIZIERTE & REALE PUNKTE
    # ============================================================
    if calibration_data.get('virtual_points') is not None:
        virtual_points = calibration_data['virtual_points']
        real_points = calibration_data['real_points']

        # Projizierte Punkte (auf Frontscheibe, y=0)
        ax.scatter(virtual_points[:, 0], virtual_points[:, 1], virtual_points[:, 2],
                   s=150, c='green', marker='^', label='Calib: Projiziert (Frontscheibe)',
                   edgecolors='darkgreen', linewidths=1.5, zorder=100)

        # Reale Punkte (an Rückwand, y=LY)
        ax.scatter(real_points[:, 0], real_points[:, 1], real_points[:, 2],
                   s=150, c='red', marker='o', label='Calib: Real (Rückwand)',
                   edgecolors='darkred', linewidths=1.5, zorder=100)

        # Verbindungslinien: Kamera -> Projiziert -> Real (MIT Brechung)
        normal_water_to_air = np.array([0.0, -1.0, 0.0], dtype=float)

        for i, (vp, rp) in enumerate(zip(virtual_points, real_points)):
            # Luftstrahl: Kamera -> Frontscheibe
            ax.plot([K[0], vp[0]], [K[1], vp[1]], [K[2], vp[2]],
                    'c--', linewidth=1.5, alpha=0.6,
                    label='Luftstrahl' if i == 0 else '')

            # Wasserstrahl: Frontscheibe -> Rückwand
            ax.plot([vp[0], rp[0]], [vp[1], rp[1]], [vp[2], rp[2]],
                    'g--', linewidth=1.5, alpha=0.6,
                    label='Wasserstrahl' if i == 0 else '')

    # ============================================================
    # CALIBRATION: LASERLINIE MIT BRECHUNG
    # ============================================================
    calib_laser_3d = calibration_data.get('laser_points_3d', [])
    if len(calib_laser_3d) > 0:
        ax.scatter(calib_laser_3d[:, 0], calib_laser_3d[:, 1], calib_laser_3d[:, 2],
                   c='lime', s=20, marker='o', label=f'Calib: Laserlinie MIT Brechung ({len(calib_laser_3d)} Pkt)',
                   alpha=0.8, zorder=50)
        ax.plot(calib_laser_3d[:, 0], calib_laser_3d[:, 1], calib_laser_3d[:, 2],
                'lime', linewidth=3, alpha=0.7, zorder=50)

    # ============================================================
    # CALIBRATION: LASERLINIE OHNE BRECHUNG (VERGLEICH)
    # ============================================================
    if show_refraction_comparison and len(calib_laser_3d) > 0:
        # Rekonstruiere ohne Brechung (geradlinig)
        calib_laser_no_refraction = reconstruct_laser_line_3d_no_refraction(
            calibration_data.get('sampled_points_floor_roi', []),
            K,
            calibration_data.get('floor_transform_inv'),
            calibration_data['cm_per_px_x'],
            calibration_data['cm_per_px_z']
        )

        if len(calib_laser_no_refraction) > 0:
            ax.scatter(calib_laser_no_refraction[:, 0],
                       calib_laser_no_refraction[:, 1],
                       calib_laser_no_refraction[:, 2],
                       c='orange', s=15, marker='x',
                       label=f'Calib: Laserlinie OHNE Brechung (Vergleich)',
                       alpha=0.6, zorder=40)
            ax.plot(calib_laser_no_refraction[:, 0],
                    calib_laser_no_refraction[:, 1],
                    calib_laser_no_refraction[:, 2],
                    'orange', linewidth=2, linestyle=':', alpha=0.5, zorder=40)

    # ============================================================
    # MEASUREMENT: LASERLINIE
    # ============================================================
    if measurement_data is not None:
        meas_laser_3d = measurement_data.get('laser_points_3d', [])
        if len(meas_laser_3d) > 0:
            ax.scatter(meas_laser_3d[:, 0], meas_laser_3d[:, 1], meas_laser_3d[:, 2],
                       c='red', s=20, marker='s',
                       label=f'Measurement: Laserlinie ({len(meas_laser_3d)} Pkt)',
                       alpha=0.8, zorder=60)
            ax.plot(meas_laser_3d[:, 0], meas_laser_3d[:, 1], meas_laser_3d[:, 2],
                    'red', linewidth=3, alpha=0.7, zorder=60)

    # ============================================================
    # ACHSEN & LEGENDE
    # ============================================================
    ax.set_xlabel('X [cm]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [cm]', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z [cm]', fontsize=12, fontweight='bold')

    title = '3D-Rekonstruktion: Calibration & Measurement\n'
    title += f'MIT Lichtbrechung (n_air={N_AIR}, n_water={N_WATER})'
    if show_refraction_comparison:
        title += ' + Vergleich OHNE Brechung'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Legende (entferne Duplikate)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9,
              framealpha=0.9, ncol=1)

    # Gleiche Skalierung
    all_points = [corners, K[None, :]]
    if len(calib_laser_3d) > 0:
        all_points.append(calib_laser_3d)
    if measurement_data is not None and len(measurement_data.get('laser_points_3d', [])) > 0:
        all_points.append(measurement_data['laser_points_3d'])

    all_points = np.vstack(all_points)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    spans = maxs - mins
    half = 0.5 * float(np.max(spans))
    mid = 0.5 * (mins + maxs)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    # Setze Blickwinkel
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Kombinierte 3D-Visualisierung gespeichert: {output_path}")
    plt.show()
    plt.close(fig)


def reconstruct_laser_line_3d_no_refraction(sampled_points_floor_roi, K,
                                            floor_roi_to_main_roi_transform,
                                            cm_per_px_x, cm_per_px_z):
    """
    Rekonstruiert die 3D-Position der Laserlinie OHNE LICHTBRECHUNG (zum Vergleich).
    Geradlinige Projektion von K durch P_front bis z=0.

    Args:
        sampled_points_floor_roi: Liste von (x, y) Punkten im Boden-ROI
        K: Kameraposition [x, y, z] in cm
        floor_roi_to_main_roi_transform: Inverse Transformationsmatrix von Boden-ROI zu Haupt-ROI
        cm_per_px_x: cm pro Pixel in X-Richtung (im Haupt-ROI)
        cm_per_px_z: cm pro Pixel in Z-Richtung (im Haupt-ROI)

    Returns:
        Liste von 3D-Punkten [x, y, z] auf dem Boden (z=0) OHNE Brechung
    """
    laser_points_3d = []

    for i, (x_floor, y_floor) in enumerate(sampled_points_floor_roi):
        # 1. Transformiere Punkt vom Boden-ROI zum Haupt-ROI
        pt_floor = np.array([[x_floor, y_floor]], dtype=np.float32)
        pt_main = cv2.perspectiveTransform(pt_floor.reshape(1, 1, 2), floor_roi_to_main_roi_transform)

        x_main = float(pt_main[0, 0, 0])
        y_main = float(pt_main[0, 0, 1])

        # 2. Konvertiere Pixel-Koordinaten im Haupt-ROI zu 3D-Koordinaten auf der FRONTSCHEIBE
        u_cm = x_main * cm_per_px_x
        z_from_top = y_main * cm_per_px_z
        z_cm = TANK_HEIGHT_Z - z_from_top

        # Punkt auf der Frontscheibe (y = 0)
        P_front = np.array([u_cm, 0.0, z_cm], dtype=float)

        # 3. Berechne Richtungsvektor (OHNE Brechung - geradlinig)
        d = P_front - K
        d_norm = np.linalg.norm(d)

        if d_norm < 1e-9:
            continue

        d = d / d_norm  # Normalisiere

        # 4. Berechne Schnittpunkt mit Ebene z=0 (geradlinige Fortsetzung)
        # Strahl: P(t) = K + t * d
        # Für z=0: K[2] + t * d[2] = 0
        # => t = -K[2] / d[2]

        if abs(d[2]) < 1e-9:
            # Strahl parallel zur z=0 Ebene
            continue

        t = -K[2] / d[2]

        if t < 0:
            # Schnittpunkt liegt hinter der Kamera
            continue

        # Berechne 3D-Punkt auf dem Boden
        P_floor = K + t * d
        P_floor[2] = 0.0  # Stelle sicher, dass z exakt 0 ist

        laser_points_3d.append(P_floor)

    return np.array(laser_points_3d, dtype=float)


# ============================================================
# LICHTBRECHUNG
# ============================================================

def refract_ray(direction, normal, n1, n2):
    """
    Berechnet den gebrochenen Richtungsvektor nach Snell (Vektorform).

    Args:
        direction: Einheitsvektor der einfallenden Richtung
        normal: Einheitsvektor der Flächennormale (zeigt von Medium 1 nach Medium 2)
        n1: Brechungsindex des einfallenden Mediums
        n2: Brechungsindex des ausfallenden Mediums

    Returns:
        Gebrochener Einheitsvektor, oder None bei Totalreflexion
    """
    # Normalisiere Eingabevektoren
    d = direction / np.linalg.norm(direction)
    n = normal / np.linalg.norm(normal)

    # Berechne cos(theta_i) - Winkel zwischen einfallendem Strahl und Normale
    cos_i = -np.dot(n, d)

    # Berechne sin²(theta_t) nach Snell
    ratio = n1 / n2
    sin2_t = ratio * ratio * (1.0 - cos_i * cos_i)

    # Prüfe auf Totalreflexion
    if sin2_t > 1.0:
        return None

    # Berechne gebrochenen Richtungsvektor
    cos_t = np.sqrt(1.0 - sin2_t)
    d_refracted = ratio * d + (ratio * cos_i - cos_t) * n

    # Normalisiere Ergebnis
    d_refracted = d_refracted / np.linalg.norm(d_refracted)

    return d_refracted


# ============================================================
# LINSENENTZERRUNG
# ============================================================

def load_and_undistort_image(image_path, calibration_npz_path=None, alpha=1.0):
    """
    Lädt ein Bild und entzerrt es optional mit Kamera-Kalibrierungsdaten aus NPZ-Datei.

    Args:
        image_path: Pfad zum Eingabebild
        calibration_npz_path: Pfad zur NPZ-Datei mit 'camera_matrix' und 'dist_coeffs', oder None
        alpha: Alpha-Parameter für cv2.getOptimalNewCameraMatrix() (0.0 bis 1.0)

    Returns:
        Entzerrtes Bild (BGR) oder Original falls keine Kalibrierung angegeben
    """
    # Lade Bild
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[ERROR] Bild nicht gefunden: {image_path}")
        return None

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[ERROR] Bild konnte nicht geladen werden: {image_path}")
        return None

    h, w = image.shape[:2]
    print(f"[INFO] Bild geladen: {img_path.name} ({w} x {h} px)")

    # Prüfe ob Kalibrierungsdaten vorhanden
    if calibration_npz_path is None:
        print(f"[INFO] Keine Kamera-Kalibrierung angegeben — verwende Originalbild")
        return image

    calib_path = Path(calibration_npz_path)
    if not calib_path.exists():
        print(f"[WARNING] Kalibrierungs-NPZ nicht gefunden: {calibration_npz_path}")
        print(f"[INFO] Verwende Originalbild ohne Entzerrung")
        return image

    # Lade Kalibrierungsdaten
    try:
        calib_data = np.load(str(calib_path))
        camera_matrix = calib_data['camera_matrix']
        dist_coeffs = calib_data['dist_coeffs']

        print(f"[INFO] Kamera-Kalibrierung geladen: {calib_path.name}")
        print(f"       Camera Matrix:\n{camera_matrix}")
        print(f"       Distortion Coeffs: {dist_coeffs.flatten()}")
        print(f"       Alpha: {alpha}")

    except Exception as e:
        print(f"[ERROR] Fehler beim Laden der Kalibrierungsdaten: {e}")
        print(f"[INFO] Verwende Originalbild ohne Entzerrung")
        return image

    # Berechne optimale neue Kameramatrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        alpha,
        (w, h)
    )

    # Entzerrung durchführen
    print(f"[INFO] Entzerrung wird durchgeführt...")
    undistorted = cv2.undistort(
        image,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix
    )

    # Optional: ROI zuschneiden (falls alpha < 1.0)
    x, y, w_roi, h_roi = roi
    if alpha < 1.0 and w_roi > 0 and h_roi > 0:
        undistorted = undistorted[y:y + h_roi, x:x + w_roi]
        print(f"       ROI zugeschnitten: {w_roi} x {h_roi} px")

    print(f"[OK] Entzerrung abgeschlossen")

    return undistorted


# ============================================================
# ROTESTE PIXEL DETECTION
# ============================================================

def find_reddest_pixels_per_line(floor_roi, num_lines=100, marker_size=5):
    """
    Teilt das Boden-ROI in vertikale Linien ein und findet in jeder Linie den rotesten Pixel.

    Args:
        floor_roi: Boden-ROI Bild (BGR)
        num_lines: Anzahl der vertikalen Linien
        marker_size: Größe der Markierung (Radius in Pixeln)

    Returns:
        Visualisiertes Bild mit markierten rotesten Pixeln, Liste der Pixel-Koordinaten
    """
    h, w = floor_roi.shape[:2]

    print(f"[INFO] Boden-ROI-Größe: {w} x {h} px")
    print(f"[INFO] Anzahl vertikaler Linien: {num_lines}")
    print(f"[INFO] Marker-Größe: {marker_size} px")

    # Erstelle schwarzes Ausgabebild
    output_img = np.zeros_like(floor_roi)

    # Berechne Breite jeder vertikalen Linie
    line_width = w / num_lines

    reddest_pixels = []

    for i in range(num_lines):
        # Berechne Start- und End-X-Koordinate für diese vertikale Linie
        x_start = int(i * line_width)
        x_end = int((i + 1) * line_width)

        # Extrahiere die vertikale Linie
        line_strip = floor_roi[:, x_start:x_end]

        if line_strip.size == 0:
            continue

        # Berechne "Röte" für jeden Pixel in dieser Linie
        # Röte = R - max(G, B)
        b = line_strip[:, :, 0].astype(np.float32)
        g = line_strip[:, :, 1].astype(np.float32)
        r = line_strip[:, :, 2].astype(np.float32)

        redness = r - np.maximum(g, b)

        # Finde den rotesten Pixel
        max_idx = np.argmax(redness)
        max_y, max_x_local = np.unravel_index(max_idx, redness.shape)

        # Konvertiere zu globalen Koordinaten
        max_x_global = x_start + max_x_local

        reddest_pixels.append((max_x_global, max_y))

        # Zeichne Markierung (weißer Kreis)
        cv2.circle(output_img, (max_x_global, max_y), marker_size, (255, 255, 255), -1)

    print(f"[INFO] Gefunden: {len(reddest_pixels)} roteste Pixel")

    return output_img, reddest_pixels


def fit_line_to_reddest_pixels(reddest_pixels, floor_size):
    """
    Fittet eine Gerade durch die rotesten Pixel.

    Args:
        reddest_pixels: Liste von (x, y) Koordinaten
        floor_size: Größe des Boden-ROI (Breite = Höhe)

    Returns:
        Zwei Endpunkte der Linie: (x1, y1), (x2, y2)
    """
    if len(reddest_pixels) < 2:
        print("[ERROR] Zu wenige Pixel für Line-Fitting")
        return None

    # Konvertiere zu numpy array
    points = np.array(reddest_pixels, dtype=np.float32)

    # Fitte Gerade mit cv2.fitLine
    # Output: [vx, vy, x0, y0] wobei (vx, vy) Richtungsvektor und (x0, y0) Punkt auf Linie
    line_params = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    # Extrahiere Parameter (cv2.fitLine gibt Arrays zurück)
    vx = float(line_params[0][0])
    vy = float(line_params[1][0])
    x0 = float(line_params[2][0])
    y0 = float(line_params[3][0])

    print(f"[INFO] Line-Fit Parameter: vx={vx:.4f}, vy={vy:.4f}, x0={x0:.2f}, y0={y0:.2f}")

    # Berechne Endpunkte an den Rändern des Boden-ROI
    # Linie: (x, y) = (x0, y0) + t * (vx, vy)

    # Für x = 0:
    if abs(vx) > 1e-6:
        t_left = -x0 / vx
        y_left = y0 + t_left * vy
    else:
        # Vertikale Linie
        y_left = 0

    # Für x = floor_size - 1:
    if abs(vx) > 1e-6:
        t_right = (floor_size - 1 - x0) / vx
        y_right = y0 + t_right * vy
    else:
        # Vertikale Linie
        y_right = floor_size - 1

    # Begrenze y-Werte auf [0, floor_size - 1]
    y_left = np.clip(y_left, 0, floor_size - 1)
    y_right = np.clip(y_right, 0, floor_size - 1)

    p1 = (0, int(y_left))
    p2 = (floor_size - 1, int(y_right))

    print(f"[INFO] Gefittete Linie: P1={p1}, P2={p2}")

    return p1, p2


def sample_points_on_line(p1, p2, num_points=200):
    """
    Sampelt gleichmäßig verteilte Punkte auf einer Linie.

    Args:
        p1: Startpunkt (x1, y1)
        p2: Endpunkt (x2, y2)
        num_points: Anzahl der zu samplenden Punkte

    Returns:
        Liste von (x, y) Koordinaten
    """
    x1, y1 = p1
    x2, y2 = p2

    # Lineare Interpolation
    t_values = np.linspace(0, 1, num_points)
    x_values = x1 + t_values * (x2 - x1)
    y_values = y1 + t_values * (y2 - y1)

    sampled_points = [(int(x), int(y)) for x, y in zip(x_values, y_values)]

    return sampled_points


def create_measurement_polyline(reddest_pixels, floor_size, num_dense_samples=1000):
    """
    Erstellt eine dichte Polyline durch die rotesten Pixel (KEINE Gerade!).

    1. Sortiert die Pixel nach X-Koordinate
    2. Verbindet aufeinanderfolgende Punkte linear
    3. Sampelt dicht entlang der Polyline

    Args:
        reddest_pixels: Liste von (x, y) Pixel-Koordinaten
        floor_size: Größe des Boden-ROI
        num_dense_samples: Anzahl der dicht gesampelten Punkte

    Returns:
        Liste von (x, y) Koordinaten — dicht gesampelt entlang der Polyline
    """
    if len(reddest_pixels) < 2:
        print("[ERROR] Zu wenige Pixel für Polyline-Erstellung")
        return []

    # Konvertiere zu numpy array und sortiere nach X-Koordinate
    points = np.array(reddest_pixels, dtype=np.float32)
    sorted_indices = np.argsort(points[:, 0])
    points_sorted = points[sorted_indices]

    x_sorted = points_sorted[:, 0]
    y_sorted = points_sorted[:, 1]

    print(f"[INFO] Polyline: {len(points_sorted)} Punkte, X-Bereich: {x_sorted.min():.1f} bis {x_sorted.max():.1f}")

    # Erstelle stückweise lineare Interpolation
    interp_func = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')

    # Sample dicht entlang der X-Achse
    x_min = x_sorted.min()
    x_max = x_sorted.max()
    x_dense = np.linspace(x_min, x_max, num_dense_samples)
    y_dense = interp_func(x_dense)

    # Begrenze Y-Werte auf gültigen Bereich
    y_dense = np.clip(y_dense, 0, floor_size - 1)

    sampled_points = [(int(x), int(y)) for x, y in zip(x_dense, y_dense)]

    print(f"[INFO] Polyline: {len(sampled_points)} dicht gesampelte Punkte")

    return sampled_points


def create_polyline_with_outlier_filter(reddest_pixels, floor_size, slope_threshold=5.0,
                                        num_dense_samples=1000, fit_method='none',
                                        exclude_center_percent=10.0, exclude_edge_percent=5.0):
    """
    Erstellt eine dichte Polyline durch die rotesten Pixel mit Ausreißer-Filterung.

    1. Sortiert die Pixel nach X-Koordinate
    2. Erkennt Ausreißer anhand der Steigung zu Nachbarpunkten
    3. Ersetzt Ausreißer durch interpolierte Werte
    4. Optional: Fittet Gerade oder Parabel durch die Punkte (mit Ausschluss-Bereichen)
    5. Sampelt dicht entlang der Linie/Kurve

    Args:
        reddest_pixels: Liste von (x, y) Pixel-Koordinaten
        floor_size: Größe des Boden-ROI
        slope_threshold: Maximale erlaubte Steigung |Δy/Δx| zu Nachbarn
        num_dense_samples: Anzahl der dicht gesampelten Punkte
        fit_method: 'none' (Originallinie), 'line' (Gerade), 'parabola' (Parabel 2. Grades)
        exclude_center_percent: Prozent der ROI-Größe um die Mitte herum zu ignorieren
        exclude_edge_percent: Prozent der ROI-Größe an den Rändern zu ignorieren

    Returns:
        Liste von (x, y) Koordinaten — dicht gesampelt entlang der Polyline
    """
    if len(reddest_pixels) < 2:
        print("[ERROR] Zu wenige Pixel für Polyline-Erstellung")
        return None

    # Schritt 1: Sortierung nach X-Koordinate
    points = np.array(reddest_pixels, dtype=np.float32)
    sorted_indices = np.argsort(points[:, 0])
    points_sorted = points[sorted_indices]

    x_vals = points_sorted[:, 0].copy()
    y_vals = points_sorted[:, 1].copy()

    n = len(x_vals)

    print(f"[INFO] Polyline mit Ausreißer-Filterung: {n} Punkte, X-Bereich: {x_vals.min():.1f} bis {x_vals.max():.1f}")
    print(f"       Slope-Threshold: {slope_threshold}")
    print(f"       Fit-Methode: {fit_method}")

    # Schritt 2: Ausreißer-Erkennung
    outlier_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        slopes_exceeding = []

        # Steigung zum linken Nachbarn
        if i > 0:
            dx_left = x_vals[i] - x_vals[i - 1]
            dy_left = y_vals[i] - y_vals[i - 1]
            if abs(dx_left) > 1e-6:
                slope_left = abs(dy_left / dx_left)
                slopes_exceeding.append(slope_left > slope_threshold)

        # Steigung zum rechten Nachbarn
        if i < n - 1:
            dx_right = x_vals[i + 1] - x_vals[i]
            dy_right = y_vals[i + 1] - y_vals[i]
            if abs(dx_right) > 1e-6:
                slope_right = abs(dy_right / dx_right)
                slopes_exceeding.append(slope_right > slope_threshold)

        # Ausreißer: ALLE Steigungen überschreiten Threshold
        if len(slopes_exceeding) > 0 and all(slopes_exceeding):
            outlier_mask[i] = True

    num_outliers = np.sum(outlier_mask)
    print(f"[INFO] Ausreißer erkannt: {num_outliers} von {n} Punkten")

    # Schritt 3: Ausreißer ersetzen
    if num_outliers > 0:
        valid_mask = ~outlier_mask
        valid_x = x_vals[valid_mask]
        valid_y = y_vals[valid_mask]

        if len(valid_x) < 2:
            print("[WARNING] Zu wenige gültige Punkte nach Filterung — überspringe Ausreißer-Filterung")
        else:
            # Interpolationsfunktion erstellen
            interp_func = interp1d(valid_x, valid_y, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')

            # Ausreißer durch interpolierte Werte ersetzen
            outlier_indices = np.where(outlier_mask)[0]
            for idx in outlier_indices:
                y_vals[idx] = interp_func(x_vals[idx])
                print(f"       Ausreißer @ x={x_vals[idx]:.1f}: y={points_sorted[idx, 1]:.1f} → {y_vals[idx]:.1f}")

    # Schritt 4: Optional Fitting
    x_min = x_vals.min()
    x_max = x_vals.max()
    x_dense = np.linspace(x_min, x_max, num_dense_samples)

    if fit_method in ['line', 'parabola']:
        # Berechne Ausschluss-Bereiche
        x_center = floor_size / 2.0
        center_half_width = (exclude_center_percent / 100.0) * floor_size / 2.0
        edge_width = (exclude_edge_percent / 100.0) * floor_size

        # Definiere Ausschluss-Bereiche
        center_min = x_center - center_half_width
        center_max = x_center + center_half_width
        left_edge_max = edge_width
        right_edge_min = floor_size - edge_width

        print(f"[INFO] Fit-Ausschluss-Bereiche:")
        print(f"       Linker Rand: [0, {left_edge_max:.1f}] ({exclude_edge_percent}%)")
        print(f"       Mitte: [{center_min:.1f}, {center_max:.1f}] ({exclude_center_percent}%)")
        print(f"       Rechter Rand: [{right_edge_min:.1f}, {floor_size}] ({exclude_edge_percent}%)")

        # Erstelle Maske für Punkte, die beim Fitting verwendet werden
        fit_mask = np.ones(n, dtype=bool)

        for i in range(n):
            x = x_vals[i]
            # Ausschließen wenn in einem der Bereiche
            if (x <= left_edge_max or
                    x >= right_edge_min or
                    (center_min <= x <= center_max)):
                fit_mask[i] = False

        fit_x = x_vals[fit_mask]
        fit_y = y_vals[fit_mask]

        num_fit_points = len(fit_x)
        num_excluded = n - num_fit_points

        print(f"[INFO] Fitting: {num_fit_points} Punkte verwendet, {num_excluded} ausgeschlossen")

        if num_fit_points < 3:
            print(f"[WARNING] Zu wenige Punkte für Fitting ({num_fit_points}) — verwende alle Punkte")
            fit_x = x_vals
            fit_y = y_vals

    if fit_method == 'line':
        # Gerade fitten mit cv2.fitLine
        print(f"[INFO] Fitte Gerade durch {len(fit_x)} Punkte...")
        points_for_fit = np.column_stack([fit_x, fit_y]).astype(np.float32)
        line_params = cv2.fitLine(points_for_fit, cv2.DIST_L2, 0, 0.01, 0.01)

        vx = float(line_params[0][0])
        vy = float(line_params[1][0])
        x0 = float(line_params[2][0])
        y0 = float(line_params[3][0])

        print(f"       Line-Fit: vx={vx:.4f}, vy={vy:.4f}, x0={x0:.2f}, y0={y0:.2f}")

        # Berechne Y-Werte auf der Geraden
        # Linie: (x, y) = (x0, y0) + t * (vx, vy)
        # Für gegebenes x: t = (x - x0) / vx, dann y = y0 + t * vy
        if abs(vx) > 1e-6:
            t_dense = (x_dense - x0) / vx
            y_dense = y0 + t_dense * vy
        else:
            # Vertikale Linie (sollte nicht vorkommen bei X-sortierter Polyline)
            y_dense = np.full_like(x_dense, y0)


    elif fit_method == 'parabola':

        # Parabel fitten (Scheitelpunkt frei gefittet)

        print(f"[INFO] Fitte Parabel (Grad 2) durch {len(fit_x)} Punkte...")

        # Fitte Parabel y = a*x^2 + b*x + c (3 Parameter)

        coeffs = np.polyfit(fit_x, fit_y, deg=2)

        a = coeffs[0]

        b = coeffs[1]

        c = coeffs[2]

        # Berechne Scheitelpunkt-Position: x_v = -b/(2a)

        if abs(a) > 1e-9:

            x_vertex = -b / (2 * a)

            y_vertex = a * x_vertex ** 2 + b * x_vertex + c

        else:

            x_vertex = float('nan')

            y_vertex = float('nan')

        print(f"       Parabel-Koeffizienten: a={a:.6f}, b={b:.6f}, c={c:.6f}")

        print(f"       Parabel-Form: y = {a:.6f}*x^2 + {b:.6f}*x + {c:.6f}")

        print(f"       Scheitelpunkt: x={x_vertex:.1f}, y={y_vertex:.1f}")

        # Berechne Y-Werte auf der Parabel

        y_dense = np.polyval(coeffs, x_dense)


    else:  # fit_method == 'none'
        # Keine Fitting — verwende stückweise lineare Interpolation durch gefilterte Punkte
        print(f"[INFO] Keine Fitting-Methode — verwende Originallinie (interpoliert)")
        interp_func_final = interp1d(x_vals, y_vals, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
        y_dense = interp_func_final(x_dense)

    # Clamp Y-Werte auf gültigen Bereich
    y_dense = np.clip(y_dense, 0, floor_size - 1)

    sampled_points = [(int(x), int(y)) for x, y in zip(x_dense, y_dense)]

    print(f"[INFO] Polyline: {len(sampled_points)} dicht gesampelte Punkte")

    return sampled_points


def reconstruct_laser_line_3d(sampled_points_floor_roi, K, floor_roi_to_main_roi_transform,
                              cm_per_px_x, cm_per_px_z, store_sampled_points=False):
    """
    Rekonstruiert die 3D-Position der Laserlinie auf dem Boden (z=0) MIT LICHTBRECHUNG.

    Args:
        sampled_points_floor_roi: Liste von (x, y) Punkten im Boden-ROI
        K: Kameraposition [x, y, z] in cm
        floor_roi_to_main_roi_transform: Inverse Transformationsmatrix von Boden-ROI zu Haupt-ROI
        cm_per_px_x: cm pro Pixel in X-Richtung (im Haupt-ROI)
        cm_per_px_z: cm pro Pixel in Z-Richtung (im Haupt-ROI)
        store_sampled_points: Wenn True, gebe auch sampled_points zurück

    Returns:
        Liste von 3D-Punkten [x, y, z] auf dem Boden (z=0)
        Wenn store_sampled_points=True: (laser_points_3d, sampled_points_floor_roi)
    """
    laser_points_3d = []

    print(f"[INFO] Rekonstruiere {len(sampled_points_floor_roi)} Punkte MIT LICHTBRECHUNG...")
    print(f"       n_air = {N_AIR}, n_water = {N_WATER}")

    # Flächennormale der Frontscheibe (zeigt von Luft ins Wasser, also +y)
    normal_air_to_water = np.array([0.0, 1.0, 0.0], dtype=float)

    for i, (x_floor, y_floor) in enumerate(sampled_points_floor_roi):
        # 1. Transformiere Punkt vom Boden-ROI zum Haupt-ROI
        pt_floor = np.array([[x_floor, y_floor]], dtype=np.float32)
        pt_main = cv2.perspectiveTransform(pt_floor.reshape(1, 1, 2), floor_roi_to_main_roi_transform)

        x_main = float(pt_main[0, 0, 0])
        y_main = float(pt_main[0, 0, 1])

        # 2. Konvertiere Pixel-Koordinaten im Haupt-ROI zu 3D-Koordinaten auf der FRONTSCHEIBE
        u_cm = x_main * cm_per_px_x
        z_from_top = y_main * cm_per_px_z
        z_cm = TANK_HEIGHT_Z - z_from_top

        # Punkt auf der Frontscheibe (y = 0)
        P_front = np.array([u_cm, 0.0, z_cm], dtype=float)

        # 3. Berechne Richtungsvektor in Luft (von K nach P_front)
        d_air = P_front - K
        d_air_norm = np.linalg.norm(d_air)

        if d_air_norm < 1e-9:
            if i < 3:
                print(f"  Punkt {i}: WARNUNG - Kamera und Frontpunkt identisch (übersprungen)")
            continue

        d_air = d_air / d_air_norm  # Normalisiere

        # 4. Berechne gebrochenen Richtungsvektor im Wasser
        d_water = refract_ray(d_air, normal_air_to_water, N_AIR, N_WATER)

        if d_water is None:
            if i < 3:
                print(f"  Punkt {i}: WARNUNG - Totalreflexion (übersprungen)")
            continue

        # Debug: Zeige Einfalls- und Brechungswinkel für erste paar Punkte
        if i < 3 or i == len(sampled_points_floor_roi) - 1:
            # Einfallswinkel (Winkel zwischen d_air und Normale)
            cos_theta_i = -np.dot(normal_air_to_water, d_air)
            theta_i_deg = np.degrees(np.arccos(np.clip(cos_theta_i, -1.0, 1.0)))

            # Brechungswinkel (Winkel zwischen d_water und Normale)
            cos_theta_t = np.dot(normal_air_to_water, d_water)
            theta_t_deg = np.degrees(np.arccos(np.clip(cos_theta_t, -1.0, 1.0)))

            print(f"  Punkt {i}: Floor-ROI ({x_floor}, {y_floor}) -> Haupt-ROI ({x_main:.2f}, {y_main:.2f})")
            print(f"    Front: ({P_front[0]:.2f}, {P_front[1]:.2f}, {P_front[2]:.2f})")
            print(f"    Einfallswinkel: {theta_i_deg:.2f}°, Brechungswinkel: {theta_t_deg:.2f}°")

        # 5. Berechne Schnittpunkt des gebrochenen Strahls mit Ebene z=0
        # Strahl im Wasser: P(t) = P_front + t * d_water
        # Für z=0: P_front[2] + t * d_water[2] = 0
        # => t = -P_front[2] / d_water[2]

        if abs(d_water[2]) < 1e-9:
            # Strahl parallel zur z=0 Ebene
            if i < 3:
                print(f"    WARNUNG: Gebrochener Strahl parallel zu z=0 Ebene (übersprungen)")
            continue

        t = -P_front[2] / d_water[2]

        if t < 0:
            # Schnittpunkt liegt hinter der Frontscheibe (sollte nicht vorkommen)
            if i < 3:
                print(f"    WARNUNG: Schnittpunkt hinter Frontscheibe (t={t:.3f}, übersprungen)")
            continue

        # Berechne 3D-Punkt auf dem Boden
        P_floor = P_front + t * d_water
        P_floor[2] = 0.0  # Stelle sicher, dass z exakt 0 ist

        # Debug: Zeige erste paar 3D-Punkte
        if i < 3 or i == len(sampled_points_floor_roi) - 1:
            print(f"    3D Floor: ({P_floor[0]:.2f}, {P_floor[1]:.2f}, {P_floor[2]:.2f}), t={t:.3f}")

        laser_points_3d.append(P_floor)

    laser_points_3d = np.array(laser_points_3d, dtype=float)

    if store_sampled_points:
        return laser_points_3d, sampled_points_floor_roi
    else:
        return laser_points_3d


def reconstruct_ray_sheet_at_y_plane(sampled_points_floor_roi, K, floor_roi_to_main_roi_transform,
                                     cm_per_px_x, cm_per_px_z, y_plane):
    """
    Schießt Strahlen von K durch die Frontscheiben-Projektion jedes gesampelten Punktes
    und schneidet jeden Strahl mit der Ebene y = y_plane MIT LICHTBRECHUNG.

    Args:
        sampled_points_floor_roi: Liste von (x, y) Punkten im Boden-ROI
        K: Kameraposition [x, y, z] in cm
        floor_roi_to_main_roi_transform: Inverse Transformationsmatrix Boden-ROI → Haupt-ROI
        cm_per_px_x: cm pro Pixel in X-Richtung
        cm_per_px_z: cm pro Pixel in Z-Richtung
        y_plane: Y-Koordinate der Schnittebene (= mittlere Y-Koordinate der Calibration-Linie)

    Returns:
        Array von 3D-Punkten [x, y_plane, z] — die Schnittpunkte der Strahlen mit der Y-Ebene
    """
    ray_hits_3d = []

    print(
        f"[INFO] Rekonstruiere Ray-Sheet MIT LICHTBRECHUNG: {len(sampled_points_floor_roi)} Strahlen, y_plane = {y_plane:.3f} cm")
    print(f"       n_air = {N_AIR}, n_water = {N_WATER}")

    # Flächennormale der Frontscheibe (zeigt von Luft ins Wasser, also +y)
    normal_air_to_water = np.array([0.0, 1.0, 0.0], dtype=float)

    for i, (x_floor, y_floor) in enumerate(sampled_points_floor_roi):
        # 1. Transformiere Punkt vom Boden-ROI zum Haupt-ROI
        pt_floor = np.array([[x_floor, y_floor]], dtype=np.float32)
        pt_main = cv2.perspectiveTransform(pt_floor.reshape(1, 1, 2), floor_roi_to_main_roi_transform)

        x_main = float(pt_main[0, 0, 0])
        y_main = float(pt_main[0, 0, 1])

        # 2. Konvertiere Pixel-Koordinaten im Haupt-ROI zu 3D-Koordinaten auf der FRONTSCHEIBE
        u_cm = x_main * cm_per_px_x
        z_from_top = y_main * cm_per_px_z
        z_cm = TANK_HEIGHT_Z - z_from_top

        P_front = np.array([u_cm, 0.0, z_cm], dtype=float)

        # 3. Berechne Richtungsvektor in Luft (von K nach P_front)
        d_air = P_front - K
        d_air_norm = np.linalg.norm(d_air)

        if d_air_norm < 1e-9:
            if i < 3:
                print(f"  Punkt {i}: WARNUNG - Kamera und Frontpunkt identisch (übersprungen)")
            continue

        d_air = d_air / d_air_norm  # Normalisiere

        # 4. Berechne gebrochenen Richtungsvektor im Wasser
        d_water = refract_ray(d_air, normal_air_to_water, N_AIR, N_WATER)

        if d_water is None:
            if i < 3:
                print(f"  Punkt {i}: WARNUNG - Totalreflexion (übersprungen)")
            continue

        # 5. Berechne Schnittpunkt des gebrochenen Strahls mit Ebene y = y_plane
        # Strahl im Wasser: P(t) = P_front + t * d_water
        # Für y = y_plane: P_front[1] + t * d_water[1] = y_plane
        # => t = (y_plane - P_front[1]) / d_water[1]

        if abs(d_water[1]) < 1e-9:
            # Strahl parallel zur y = y_plane Ebene
            if i < 3:
                print(f"  Punkt {i}: WARNUNG - Gebrochener Strahl parallel zu y={y_plane:.3f} Ebene (übersprungen)")
            continue

        t = (y_plane - P_front[1]) / d_water[1]

        if t < 0:
            # Schnittpunkt liegt hinter der Frontscheibe
            if i < 3:
                print(f"  Punkt {i}: WARNUNG - Schnittpunkt hinter Frontscheibe (t={t:.3f}, übersprungen)")
            continue

        # Berechne 3D-Punkt auf der y = y_plane Ebene
        P_hit = P_front + t * d_water
        P_hit[1] = y_plane  # Stelle sicher, dass y exakt y_plane ist

        # Debug: Zeige erste paar Punkte
        if i < 3 or i == len(sampled_points_floor_roi) - 1:
            # Einfallswinkel
            cos_theta_i = -np.dot(normal_air_to_water, d_air)
            theta_i_deg = np.degrees(np.arccos(np.clip(cos_theta_i, -1.0, 1.0)))

            # Brechungswinkel
            cos_theta_t = np.dot(normal_air_to_water, d_water)
            theta_t_deg = np.degrees(np.arccos(np.clip(cos_theta_t, -1.0, 1.0)))

            print(f"  Punkt {i}: Floor-ROI ({x_floor}, {y_floor}) -> Haupt-ROI ({x_main:.2f}, {y_main:.2f})")
            print(f"    Front: ({P_front[0]:.2f}, {P_front[1]:.2f}, {P_front[2]:.2f})")
            print(f"    Einfallswinkel: {theta_i_deg:.2f}°, Brechungswinkel: {theta_t_deg:.2f}°")
            print(f"    Hit @ y={y_plane:.3f}: ({P_hit[0]:.2f}, {P_hit[1]:.2f}, {P_hit[2]:.2f}), t={t:.3f}")

        ray_hits_3d.append(P_hit)

    return np.array(ray_hits_3d, dtype=float)


def visualize_laser_line_3d(laser_points_3d, K, output_path):
    """
    Erstellt 3D-Visualisierung der rekonstruierten Laserlinie.

    Args:
        laser_points_3d: Array von 3D-Punkten [x, y, z]
        K: Kameraposition [x, y, z]
        output_path: Pfad zum Speichern der Visualisierung
    """
    LX = TANK_WIDTH_X
    LY = TANK_DEPTH_Y
    LZ = TANK_HEIGHT_Z

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Zeichne Aquarium
    corners = np.array([
        [0, 0, 0], [LX, 0, 0], [0, LY, 0], [LX, LY, 0],
        [0, 0, LZ], [LX, 0, LZ], [0, LY, LZ], [LX, LY, LZ],
    ], dtype=float)

    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]], 'b-', linewidth=1.2, alpha=0.5)

    # Zeichne Boden (z=0)
    floor_corners = np.array([
        [0, 0, 0], [LX, 0, 0], [LX, LY, 0], [0, LY, 0]
    ])
    floor_poly = [[floor_corners[0], floor_corners[1], floor_corners[2], floor_corners[3]]]
    ax.add_collection3d(Poly3DCollection(floor_poly, alpha=0.2, facecolor='gray', edgecolor='black'))

    # Zeichne Frontscheibe (y=0) zur Verdeutlichung
    front_corners = np.array([
        [0, 0, 0], [LX, 0, 0], [LX, 0, LZ], [0, 0, LZ]
    ])
    front_poly = [[front_corners[0], front_corners[1], front_corners[2], front_corners[3]]]
    ax.add_collection3d(Poly3DCollection(front_poly, alpha=0.1, facecolor='cyan', edgecolor='blue', linewidth=2))

    # Zeichne Laserlinie
    if len(laser_points_3d) > 0:
        ax.scatter(laser_points_3d[:, 0], laser_points_3d[:, 1], laser_points_3d[:, 2],
                   c='red', s=10, marker='o', label=f'Laserlinie ({len(laser_points_3d)} Punkte)')

        # Verbinde Punkte mit Linie
        ax.plot(laser_points_3d[:, 0], laser_points_3d[:, 1], laser_points_3d[:, 2],
                'r-', linewidth=2, alpha=0.7)

    # Zeichne Kamera
    ax.scatter([K[0]], [K[1]], [K[2]], s=200, c='purple', marker='*',
               label=f'Kamera\n({K[0]:.1f}, {K[1]:.1f}, {K[2]:.1f})')

    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    ax.set_title(f'3D-Rekonstruktion der Laserlinie (MIT LICHTBRECHUNG: n_air={N_AIR}, n_water={N_WATER})')
    ax.legend()

    # Setze gleiche Skalierung
    all_points = np.vstack([corners, laser_points_3d, K[None, :]])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    spans = maxs - mins
    half = 0.5 * float(np.max(spans))
    mid = 0.5 * (mins + maxs)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 3D-Visualisierung gespeichert: {output_path}")
    plt.show()
    plt.close(fig)


def select_quad(frame, window_name="ROI-Quad", predefined_quad=None):
    """Wähle 4 Punkte für ROI-Quad aus oder verwende vordefinierte Werte."""

    # Wenn vordefinierte Quad-Punkte vorhanden sind, verwende diese
    if predefined_quad is not None:
        print(f"[INFO] Verwende vordefinierte Quad-Punkte: {predefined_quad.tolist()}")
        return predefined_quad

    print(f"[INFO] Starte {window_name} - Klicke 4 Punkte (ESC=abbrechen)")

    unique_name = f"{window_name}_{int(time.time() * 1000) % 10000}"

    if IS_MACOS:
        time.sleep(2.0)
    else:
        time.sleep(0.5)

    clone = frame.copy()
    pts = []
    window_created = False

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            print(f"Punkt {len(pts)}: ({x}, {y})")

    try:
        try:
            cv2.namedWindow(unique_name, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow(unique_name, 1280, 720)

            if IS_MACOS:
                for _ in range(10):
                    cv2.waitKey(10)
                time.sleep(0.3)

            cv2.setMouseCallback(unique_name, mouse_cb)
        except Exception as e:
            print(f"[ERROR] Konnte Fenster nicht erstellen: {e}")
            return None

        loop_counter = 0
        while True:
            try:
                disp = clone.copy()

                for i, p in enumerate(pts):
                    cv2.circle(disp, p, 8, (0, 255, 0), -1)
                    cv2.putText(disp, str(i + 1), (p[0] + 10, p[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if i > 0:
                        cv2.line(disp, pts[i - 1], p, (0, 255, 0), 2)

                if len(pts) == 4:
                    cv2.line(disp, pts[-1], pts[0], (0, 255, 0), 2)

                status = f"Punkte: {len(pts)}/4 | ENTER=OK | r=reset | ESC=abbrechen"
                cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow(unique_name, disp)

                wait_time = 50 if IS_MACOS else 1
                key = cv2.waitKey(wait_time) & 0xFF

                if key == 27:
                    print("[INFO] Abgebrochen")
                    break
                elif key == ord('r'):
                    pts = []
                    print("[INFO] Punkte zurückgesetzt")
                elif key in (13, 32) and len(pts) == 4:
                    print("[INFO] 4 Punkte ausgewählt")

                    if window_created:
                        cv2.destroyWindow(unique_name)
                        if IS_MACOS:
                            for _ in range(10):
                                cv2.waitKey(10)
                            time.sleep(1.0)
                        else:
                            cv2.waitKey(1)

                    return np.array(pts, dtype=np.float32)

                loop_counter += 1

            except Exception as e:
                print(f"[ERROR] Fehler in Loop-Iteration {loop_counter}: {e}")
                break

        if window_created:
            try:
                cv2.destroyWindow(unique_name)
                if IS_MACOS:
                    for _ in range(10):
                        cv2.waitKey(10)
                    time.sleep(1.0)
                else:
                    cv2.waitKey(1)
            except:
                pass

        return None

    except Exception as e:
        print(f"[ERROR] Kritischer Fenster-Fehler: {e}")
        if window_created:
            try:
                cv2.destroyWindow(unique_name)
                if IS_MACOS:
                    time.sleep(1.0)
            except:
                pass
        return None


def create_edge_blindspots(roi_size, edge_percent=1.0):
    """Erstellt Blindspots für alle vier Ränder."""
    W_rect, H_rect = roi_size

    left_width = int(round(W_rect * (edge_percent / 100.0)))
    right_width = int(round(W_rect * (edge_percent / 100.0)))
    top_height = int(round(H_rect * (edge_percent / 100.0)))
    bottom_height = int(round(H_rect * (edge_percent / 100.0)))

    edge_blindspots = []

    if left_width > 0:
        edge_blindspots.append([(0, 0), (left_width, 0), (left_width, H_rect - 1), (0, H_rect - 1)])

    if right_width > 0:
        edge_blindspots.append([(W_rect - right_width, 0), (W_rect - 1, 0),
                                (W_rect - 1, H_rect - 1), (W_rect - right_width, H_rect - 1)])

    if top_height > 0:
        edge_blindspots.append([(0, 0), (W_rect - 1, 0), (W_rect - 1, top_height), (0, top_height)])

    if bottom_height > 0:
        edge_blindspots.append([(0, H_rect - bottom_height), (W_rect - 1, H_rect - bottom_height),
                                (W_rect - 1, H_rect - 1), (0, H_rect - 1)])

    return edge_blindspots


def select_blindspot(roi_img, window_name="Blindspot", roi_size=None, auto_top_percent=5.0):
    """Blindspot-Auswahl mit automatischen und manuellen Bereichen."""
    if roi_size is None:
        roi_size = (roi_img.shape[1], roi_img.shape[0])

    W_rect, H_rect = roi_size
    auto_blindspots = []

    if AUTO_BLINDSPOT_ENABLED and auto_top_percent > 0:
        blindspot_height = int(round(H_rect * (auto_top_percent / 100.0)))
        auto_top = [(0, 0), (W_rect - 1, 0), (W_rect - 1, blindspot_height), (0, blindspot_height)]
        auto_blindspots.append(auto_top)
        print(f"[INFO] Automatischer oberer Blindspot: {auto_top_percent}% ({blindspot_height} px)")

    if AUTO_EDGE_BLINDSPOT_ENABLED and AUTO_EDGE_BLINDSPOT_PERCENT > 0:
        edge_blindspots = create_edge_blindspots(roi_size, AUTO_EDGE_BLINDSPOT_PERCENT)
        auto_blindspots.extend(edge_blindspots)

    print(f"[INFO] {len(auto_blindspots)} automatische Blindspot(s) aktiv")
    print(f"       Linksklick = Punkt | Rechtsklick = löschen | ENTER = abschließen | ESC = Fertig")

    unique_name = f"{window_name}_{int(time.time() * 1000) % 10000}"
    time.sleep(WINDOW_DELAY)

    max_height = 900
    h, w = roi_img.shape[:2]
    if h > max_height:
        scale = max_height / h
        display_img = cv2.resize(roi_img, None, fx=scale, fy=scale)
    else:
        scale = 1.0
        display_img = roi_img.copy()

    clone = display_img.copy()
    finished_polygons = []
    current_polygon = []

    def mouse_cb(event, x, y, flags, param):
        nonlocal current_polygon

        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            current_polygon.append((orig_x, orig_y))
            print(f"Polygon {len(finished_polygons) + 1}, Punkt {len(current_polygon)}: ({orig_x}, {orig_y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if current_polygon:
                removed = current_polygon.pop()
                print(f"Punkt gelöscht: {removed}")

    try:
        safe_create_window(unique_name, int(w * scale), int(h * scale))
        cv2.setMouseCallback(unique_name, mouse_cb)

        while True:
            disp = clone.copy()

            for auto_poly in auto_blindspots:
                auto_pts_scaled = [(int(x * scale), int(y * scale)) for x, y in auto_poly]
                overlay = disp.copy()
                cv2.fillPoly(overlay, [np.array(auto_pts_scaled, dtype=np.int32)], color=(0, 255, 0))
                cv2.addWeighted(overlay, 0.3, disp, 0.7, 0, disp)
                cv2.polylines(disp, [np.array(auto_pts_scaled, dtype=np.int32)],
                              isClosed=True, color=(0, 255, 0), thickness=2)

            for poly_idx, polygon in enumerate(finished_polygons):
                poly_scaled = [(int(x * scale), int(y * scale)) for x, y in polygon]
                overlay = disp.copy()
                cv2.fillPoly(overlay, [np.array(poly_scaled, dtype=np.int32)], color=(255, 100, 0))
                cv2.addWeighted(overlay, 0.3, disp, 0.7, 0, disp)
                cv2.polylines(disp, [np.array(poly_scaled, dtype=np.int32)],
                              isClosed=True, color=(255, 100, 0), thickness=2)

            for i, (px, py) in enumerate(current_polygon):
                disp_x = int(px * scale)
                disp_y = int(py * scale)
                cv2.circle(disp, (disp_x, disp_y), 5, (0, 0, 255), -1)
                cv2.putText(disp, str(i + 1), (disp_x + 10, disp_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if i > 0:
                    prev_x = int(current_polygon[i - 1][0] * scale)
                    prev_y = int(current_polygon[i - 1][1] * scale)
                    cv2.line(disp, (prev_x, prev_y), (disp_x, disp_y), (0, 0, 255), 2)

            if len(current_polygon) >= 3:
                first_x = int(current_polygon[0][0] * scale)
                first_y = int(current_polygon[0][1] * scale)
                last_x = int(current_polygon[-1][0] * scale)
                last_y = int(current_polygon[-1][1] * scale)
                cv2.line(disp, (last_x, last_y), (first_x, first_y), (0, 0, 255), 2)

            status_lines = [
                f"Auto-Blindspots: {len(auto_blindspots)} | Manuelle: {len(finished_polygons)} | Aktuell: {len(current_polygon)}",
                "ENTER = abschließen | ESC = Fertig | r = Reset | Rechtsklick = löschen"
            ]
            for i, txt in enumerate(status_lines):
                cv2.putText(disp, txt, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

            cv2.imshow(unique_name, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                if len(current_polygon) >= 3:
                    finished_polygons.append(current_polygon[:])
                safe_destroy_window(unique_name)
                return auto_blindspots + finished_polygons

            elif key == ord('r'):
                if current_polygon:
                    print(f"[INFO] Polygon zurückgesetzt")
                    current_polygon = []

            elif key in (13, 32):
                if len(current_polygon) >= 3:
                    finished_polygons.append(current_polygon[:])
                    print(f"[INFO] Polygon {len(finished_polygons)} abgeschlossen")
                    current_polygon = []

    except Exception as e:
        print(f"[ERROR] Fenster-Fehler: {e}")
        safe_destroy_window(unique_name)
        return auto_blindspots


def collect_calibration_points(roi_preview_bgr, roi_name: str, cm_per_px_x: float, cm_per_px_z: float,
                               n_points: int = 4):
    """Kalibrierungspunkt-Auswahl."""
    print(f"[INFO] {roi_name} Kalibrierung - Klicke 4 Punkte: UL, UR, LR, LL")

    unique_name = f"Kalibrierung_{roi_name.replace(' ', '_')}_{int(time.time() * 1000) % 10000}"
    labels = ["P1 UL", "P2 UR", "P3 LR", "P4 LL"]

    max_height = 900
    h, w = roi_preview_bgr.shape[:2]
    if h > max_height:
        scale = max_height / h
        display_img = cv2.resize(roi_preview_bgr, None, fx=scale, fy=scale)
    else:
        scale = 1.0
        display_img = roi_preview_bgr.copy()

    clone = display_img.copy()
    pts = []

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n_points:
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            pts.append((orig_x, orig_y))
            print(f"[{roi_name}] {labels[len(pts) - 1]}: ({orig_x}, {orig_y})")

    try:
        safe_create_window(unique_name, int(w * scale), int(h * scale))
        cv2.setMouseCallback(unique_name, mouse_cb)

        while True:
            disp = clone.copy()

            for i, (px, py) in enumerate(pts):
                disp_x = int(px * scale)
                disp_y = int(py * scale)
                cv2.circle(disp, (disp_x, disp_y), 8, (0, 255, 0), -1)
                cv2.putText(disp, labels[i], (disp_x + 10, disp_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            status = f"Klicke: {labels[len(pts)] if len(pts) < n_points else 'ENTER'} | r=reset | ESC=abbrechen"
            cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(unique_name, disp)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord('r'):
                pts = []
            elif key in (13, 32) and len(pts) == n_points:
                safe_destroy_window(unique_name)

                virtual = []
                for (u, v) in pts:
                    u_cm = float(u) * cm_per_px_x
                    z_from_top = float(v) * cm_per_px_z
                    z_cm = TANK_HEIGHT_Z - z_from_top
                    virtual.append([u_cm, 0.0, z_cm])

                return np.array(virtual, dtype=float)

        safe_destroy_window(unique_name)
        return None

    except Exception as e:
        print(f"[ERROR] Fenster-Fehler: {e}")
        safe_destroy_window(unique_name)
        return None


def best_point_to_lines(P0, P1):
    """Berechnet optimale Kameraposition aus Geradenpaaren."""
    I = np.eye(3)
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)

    for P0_i, P1_i in zip(P0, P1):
        d = P1_i - P0_i
        n = np.linalg.norm(d)
        if n < 1e-9:
            raise ValueError("Zwei identische Punkte für eine Gerade.")
        d /= n

        M = I - np.outer(d, d)
        A += M
        b += M @ P0_i

    if abs(np.linalg.det(A)) < 1e-12:
        raise ValueError("System singulär - Geraden zu parallel.")

    K = np.linalg.solve(A, b)
    return K


def distance_point_to_line(K, P0, P1):
    """Berechnet Abstand eines Punktes zu einer Geraden."""
    d = P1 - P0
    d /= np.linalg.norm(d)
    v = K - P0
    proj = d * np.dot(v, d)
    return np.linalg.norm(v - proj)


def calibrate_camera_with_refraction(virtual_points_front, real_points):
    """
    Bestimmt die Kameraposition K unter Berücksichtigung der Lichtbrechung.

    Für jedes Punktpaar (P_front_i auf Scheibe bei y=0, P_real_i im Wasser):
    1. Berechne die Wasserrichtung: d_water_i = normalize(P_real_i - P_front_i)
    2. Berechne die Luftrichtung durch UMKEHRUNG der Brechung:
       d_air_i = refract_ray(d_water_i, [0, -1, 0], N_WATER, N_AIR)
       (Beachte: Normale ist jetzt umgekehrt [-y], weil wir von Wasser nach Luft gehen!)
    3. K muss auf dem Strahl P_front_i - t * d_air_i liegen (rückwärts in die Luft, t > 0)
    4. Minimiere die Summe der quadratischen Abstände von K zu allen diesen Geraden.

    Args:
        virtual_points_front: Array von 3D-Punkten auf der Frontscheibe (y=0)
        real_points: Array von 3D-Punkten im Wasser (an der Rückwand, y=TANK_DEPTH_Y)

    Returns:
        Kameraposition K [x, y, z]
    """
    print(f"\n[INFO] Kamera-Kalibrierung MIT LICHTBRECHUNG")
    print(f"       n_air = {N_AIR}, n_water = {N_WATER}")

    # Flächennormale der Frontscheibe (zeigt von Wasser nach Luft, also -y)
    normal_water_to_air = np.array([0.0, -1.0, 0.0], dtype=float)

    # Arrays für die Geraden im Luftraum
    air_ray_starts = []
    air_ray_ends = []

    for i, (P_front, P_real) in enumerate(zip(virtual_points_front, real_points)):
        # 1. Berechne Richtung im Wasser (von Frontscheibe zur Rückwand)
        d_water = P_real - P_front
        d_water_norm = np.linalg.norm(d_water)

        if d_water_norm < 1e-9:
            print(f"[WARNING] Punkt {i}: Front- und Realpunkt identisch (übersprungen)")
            continue

        d_water = d_water / d_water_norm  # Normalisiere

        # 2. Berechne Luftrichtung durch Rückwärts-Brechung (Wasser → Luft)
        d_air = refract_ray(d_water, normal_water_to_air, N_WATER, N_AIR)

        if d_air is None:
            print(f"[WARNING] Punkt {i}: Totalreflexion bei Rückwärts-Brechung (übersprungen)")
            continue

        # Debug: Zeige Winkel
        cos_theta_water = -np.dot(normal_water_to_air, d_water)
        theta_water_deg = np.degrees(np.arccos(np.clip(cos_theta_water, -1.0, 1.0)))

        cos_theta_air = np.dot(normal_water_to_air, d_air)
        theta_air_deg = np.degrees(np.arccos(np.clip(cos_theta_air, -1.0, 1.0)))

        print(f"  Punkt {i}: Wasser-Winkel: {theta_water_deg:.2f}°, Luft-Winkel: {theta_air_deg:.2f}°")

        # 3. Erstelle Gerade im Luftraum
        # Die Kamera liegt auf dem Strahl: P_front + t * (-d_air) für t > 0
        # Äquivalent: Gerade von P_front in Richtung -d_air
        # Für best_point_to_lines brauchen wir zwei Punkte auf der Geraden:
        # P0 = P_front, P1 = P_front + (-d_air) = P_front - d_air

        air_ray_starts.append(P_front)
        air_ray_ends.append(P_front - d_air)  # Punkt in Luft-Richtung

    if len(air_ray_starts) < 2:
        raise ValueError("Zu wenige gültige Punktpaare für Kalibrierung (mindestens 2 benötigt)")

    air_ray_starts = np.array(air_ray_starts, dtype=float)
    air_ray_ends = np.array(air_ray_ends, dtype=float)

    print(f"[INFO] {len(air_ray_starts)} gültige Luftstrahlen für Kalibrierung")

    # 4. Berechne Kameraposition als Punkt mit minimalem Abstand zu allen Luftstrahlen
    K = best_point_to_lines(air_ray_starts, air_ray_ends)

    return K


def plot_calibration_scene(save_path: Path, real_points: np.ndarray, virtual_points: np.ndarray,
                           K: np.ndarray, title: str):
    """Erstellt 3D-Visualisierung der Kalibrierung."""
    LX = TANK_WIDTH_X
    LY = TANK_DEPTH_Y
    LZ = TANK_HEIGHT_Z

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    corners = np.array([
        [0, 0, 0], [LX, 0, 0], [0, LY, 0], [LX, LY, 0],
        [0, 0, LZ], [LX, 0, LZ], [0, LY, LZ], [LX, LY, LZ],
    ], dtype=float)

    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]], 'b-', linewidth=1.2)

    ax.scatter(real_points[:, 0], real_points[:, 1], real_points[:, 2],
               s=100, c='red', marker='o', label='Reale Punkte')
    ax.scatter(virtual_points[:, 0], virtual_points[:, 1], virtual_points[:, 2],
               s=100, c='green', marker='^', label='Virtuelle Punkte')

    # Zeichne gebrochene Strahlen
    normal_water_to_air = np.array([0.0, -1.0, 0.0], dtype=float)

    for vp, rp in zip(virtual_points, real_points):
        # Strahl im Wasser (von Frontscheibe zur Rückwand)
        ax.plot([vp[0], rp[0]], [vp[1], rp[1]], [vp[2], rp[2]],
                'g--', linewidth=1.5, alpha=0.7, label='Wasser-Strahl' if vp is virtual_points[0] else '')

        # Berechne Luftstrahl (Rückwärts-Brechung)
        d_water = rp - vp
        d_water = d_water / np.linalg.norm(d_water)
        d_air = refract_ray(d_water, normal_water_to_air, N_WATER, N_AIR)

        if d_air is not None:
            # Zeichne Luftstrahl von Frontscheibe zur Kamera
            air_end = vp - d_air * 20  # Verlängere Strahl für Visualisierung
            ax.plot([vp[0], air_end[0]], [vp[1], air_end[1]], [vp[2], air_end[2]],
                    'c--', linewidth=1.5, alpha=0.7, label='Luft-Strahl' if vp is virtual_points[0] else '')

    ax.scatter([K[0]], [K[1]], [K[2]], s=200, c='purple', marker='*',
               label=f'Kamera\n({K[0]:.1f}, {K[1]:.1f}, {K[2]:.1f})')

    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    ax.set_title(title + f'\n(MIT LICHTBRECHUNG: n_air={N_AIR}, n_water={N_WATER})')

    # Entferne doppelte Labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    all_points = np.vstack([corners, real_points, virtual_points, K[None, :]])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    spans = maxs - mins
    half = 0.5 * float(np.max(spans))
    mid = 0.5 * (mins + maxs)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Kalibrier-Plot gespeichert: {save_path}")
    plt.show()
    plt.close(fig)


def save_calibration_data(calibration_result, output_path):
    """Speichert Kalibrierungsdaten als Pickle-Datei."""
    with open(output_path, 'wb') as f:
        pickle.dump(calibration_result, f)
    print(f"[OK] Kalibrierungsdaten gespeichert: {output_path}")


def load_calibration_data(calibration_path):
    """Lädt Kalibrierungsdaten aus Pickle-Datei."""
    with open(calibration_path, 'rb') as f:
        calibration_data = pickle.load(f)
    print(f"[OK] Kalibrierungsdaten geladen: {calibration_path}")
    return calibration_data


def compare_calibration_measurement_profiles(calibration_data, measurement_result, output_path):
    """
    Vergleicht Calibration- und Measurement-Profile.

    1. Berechnet die mittlere Y-Koordinate der Calibration-Laserlinie
    2. Erstellt eine Polyline durch die Measurement-Pixel
    3. Schießt Strahlen durch die Polyline und schneidet mit y = y_calib
    4. Interpoliert Z-Werte an den Calibration-X-Positionen
    5. Erstellt Vergleichs-Graph

    Args:
        calibration_data: Dictionary mit Kalibrierungsdaten
        measurement_result: Dictionary mit Messergebnissen
        output_path: Pfad zum Speichern des Vergleichs-Graphen

    Returns:
        Dictionary mit Vergleichsergebnissen
    """
    print(f"\n{'=' * 60}")
    print("PROFIL-VERGLEICH")
    print(f"{'=' * 60}\n")

    # Lade Calibration-Daten
    K = calibration_data['camera_position']
    calib_laser_3d = calibration_data['laser_points_3d']
    floor_size = calibration_data['floor_size']
    M_floor = calibration_data['floor_transform_matrix']
    cm_per_px_x = calibration_data['cm_per_px_x']
    cm_per_px_z = calibration_data['cm_per_px_z']

    # Lade Measurement-Daten
    meas_reddest_pixels = measurement_result['reddest_pixels']

    if len(calib_laser_3d) == 0:
        print("[ERROR] Keine Calibration-Laserpunkte vorhanden")
        return None

    if len(meas_reddest_pixels) == 0:
        print("[ERROR] Keine Measurement-Pixel vorhanden")
        return None

    # 1. Berechne mittlere Y-Koordinate der Calibration-Linie
    y_calib = np.mean(calib_laser_3d[:, 1])
    print(f"[INFO] Calibration Y-Ebene: {y_calib:.3f} cm")

    # 2. Erstelle Polyline durch Measurement-Pixel (mit Ausreißer-Filterung)
    print(f"\n[INFO] Erstelle Measurement-Polyline...")
    meas_polyline = create_polyline_with_outlier_filter(
        meas_reddest_pixels,
        floor_size,
        slope_threshold=SLOPE_THRESHOLD,
        num_dense_samples=NUM_DENSE_SAMPLES,
        fit_method='none',  # Keine Fitting beim Measurement (Originallinie)
        exclude_center_percent=0.0,  # Keine Ausschluss-Bereiche beim Measurement
        exclude_edge_percent=0.0
    )

    if len(meas_polyline) == 0:
        print("[ERROR] Polyline-Erstellung fehlgeschlagen")
        return None

    # 3. Berechne inverse Transformation Boden-ROI → Haupt-ROI
    floor_dst_pts = np.array([
        [0, 0],
        [floor_size - 1, 0],
        [floor_size - 1, floor_size - 1],
        [0, floor_size - 1]
    ], dtype=np.float32)

    floor_quad = calibration_data['floor_quad']
    M_floor_inv = cv2.getPerspectiveTransform(floor_dst_pts, floor_quad)

    # 4. Schieße Strahlen durch Polyline und schneide mit y = y_calib
    print(f"\n[INFO] Rekonstruiere Ray-Sheet @ y = {y_calib:.3f} cm...")
    meas_ray_hits = reconstruct_ray_sheet_at_y_plane(
        meas_polyline,
        K,
        M_floor_inv,
        cm_per_px_x,
        cm_per_px_z,
        y_calib
    )

    if len(meas_ray_hits) == 0:
        print("[ERROR] Ray-Sheet-Rekonstruktion fehlgeschlagen")
        return None

    print(f"[INFO] {len(meas_ray_hits)} Ray-Hits berechnet")

    # 5. Sortiere Ray-Hits nach X-Koordinate
    sorted_indices = np.argsort(meas_ray_hits[:, 0])
    meas_x_sorted = meas_ray_hits[sorted_indices, 0]
    meas_z_sorted = meas_ray_hits[sorted_indices, 2]

    print(f"[INFO] Measurement X-Bereich: {meas_x_sorted.min():.2f} bis {meas_x_sorted.max():.2f} cm")
    print(f"[INFO] Measurement Z-Bereich: {meas_z_sorted.min():.2f} bis {meas_z_sorted.max():.2f} cm")

    # 6. Erstelle Interpolationsfunktion: x → z
    meas_z_func = interp1d(
        meas_x_sorted,
        meas_z_sorted,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )

    # 7. Werte an Calibration-X-Positionen aus
    calib_x = calib_laser_3d[:, 0]
    calib_z = calib_laser_3d[:, 2]  # Sollte ≈ 0 sein

    # Sortiere Calibration-Punkte nach X
    calib_sorted_indices = np.argsort(calib_x)
    calib_x_sorted = calib_x[calib_sorted_indices]
    calib_z_sorted = calib_z[calib_sorted_indices]

    # Interpoliere Measurement-Z-Werte an Calibration-X-Positionen
    meas_z_at_calib = meas_z_func(calib_x_sorted)

    # 8. Berechne Höhendifferenz (Objektprofil)
    height_profile = meas_z_at_calib - calib_z_sorted

    print(f"\n[INFO] Höhenprofil-Statistiken:")
    print(f"       Min: {height_profile.min():.3f} cm")
    print(f"       Max: {height_profile.max():.3f} cm")
    print(f"       Mittelwert: {height_profile.mean():.3f} cm")
    print(f"       Standardabweichung: {height_profile.std():.3f} cm")

    # 9. Erstelle Vergleichs-Graph
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Subplot 1: Calibration vs. Measurement
    ax1.plot(calib_x_sorted, calib_z_sorted, 'b-', linewidth=2, label='Calibration (Referenz)', alpha=0.8)
    ax1.plot(calib_x_sorted, meas_z_at_calib, 'r-', linewidth=2, label='Measurement (Objekt)', alpha=0.8)
    ax1.set_xlabel('X-Koordinate [cm]', fontsize=12)
    ax1.set_ylabel('Z-Koordinate [cm]', fontsize=12)
    ax1.set_title('Profil-Vergleich: Calibration vs. Measurement (MIT LICHTBRECHUNG)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

    # Subplot 2: Höhenprofil (Differenz)
    ax2.plot(calib_x_sorted, height_profile, 'g-', linewidth=2, label='Höhenprofil (Measurement - Calibration)')
    ax2.fill_between(calib_x_sorted, 0, height_profile, alpha=0.3, color='green')
    ax2.set_xlabel('X-Koordinate [cm]', fontsize=12)
    ax2.set_ylabel('Höhendifferenz [cm]', fontsize=12)
    ax2.set_title('Objekthöhe über Referenzebene', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

    # Füge Statistik-Text hinzu
    stats_text = f'Min: {height_profile.min():.3f} cm\nMax: {height_profile.max():.3f} cm\nMittel: {height_profile.mean():.3f} cm\nStd: {height_profile.std():.3f} cm'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Profil-Vergleich gespeichert: {output_path}")
    plt.show()
    plt.close(fig)

    # 10. Speichere Profildaten als Textdatei
    profile_txt_path = output_path.parent / "profile_comparison_data.txt"
    with open(profile_txt_path, 'w') as f:
        f.write("PROFIL-VERGLEICH DATEN (MIT LICHTBRECHUNG)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Brechungsindizes: n_air = {N_AIR}, n_water = {N_WATER}\n")
        f.write(f"Calibration Y-Ebene: {y_calib:.3f} cm\n")
        f.write(f"Anzahl Calibration-Punkte: {len(calib_x_sorted)}\n")
        f.write(f"Anzahl Measurement-Punkte: {len(meas_x_sorted)}\n\n")
        f.write("HÖHENPROFIL-STATISTIKEN:\n")
        f.write(f"  Min: {height_profile.min():.3f} cm\n")
        f.write(f"  Max: {height_profile.max():.3f} cm\n")
        f.write(f"  Mittelwert: {height_profile.mean():.3f} cm\n")
        f.write(f"  Standardabweichung: {height_profile.std():.3f} cm\n\n")
        f.write("X [cm]\tCalib Z [cm]\tMeas Z [cm]\tHöhe [cm]\n")
        f.write("-" * 60 + "\n")
        for x, cz, mz, h in zip(calib_x_sorted, calib_z_sorted, meas_z_at_calib, height_profile):
            f.write(f"{x:.4f}\t{cz:.4f}\t{mz:.4f}\t{h:.4f}\n")

    print(f"[OK] Profildaten gespeichert: {profile_txt_path}")

    return {
        'calib_x': calib_x_sorted,
        'calib_z': calib_z_sorted,
        'meas_z': meas_z_at_calib,
        'height_profile': height_profile,
        'y_plane': y_calib,
        'stats': {
            'min': height_profile.min(),
            'max': height_profile.max(),
            'mean': height_profile.mean(),
            'std': height_profile.std()
        }
    }


def calibrate_camera(image_path: str, real_points: np.ndarray, roi_name: str = "Kamera"):
    """Hauptfunktion für Kamerakalibrierung aus Einzelbild."""

    # ============================================================
    # BILD LADEN UND ENTZERREN
    # ============================================================
    frame = load_and_undistort_image(
        image_path,
        calibration_npz_path=CAMERA_CALIBRATION_NPZ,
        alpha=UNDISTORT_ALPHA
    )

    if frame is None:
        print(f"[ERROR] Bild konnte nicht geladen werden")
        return None

    img_path = Path(image_path)

    # Erstelle Output-Ordner mit Bildnamen
    output_dir = img_path.parent / img_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output-Ordner: {output_dir}")

    # Speichere entzerrtes Bild
    if CAMERA_CALIBRATION_NPZ is not None:
        undistorted_path = output_dir / "undistorted_image.jpg"
        cv2.imwrite(str(undistorted_path), frame)
        print(f"[OK] Entzerrtes Bild gespeichert: {undistorted_path}")

    print(f"\n{'=' * 60}")
    print(f"KAMERA-KALIBRIERUNG: {img_path.name}")
    print(f"{'=' * 60}\n")

    # ============================================================
    # PRÜFE VORDEFINIERTE KAMERAPOSITION
    # ============================================================
    if PREDEFINED_CAMERA_POSITION is not None:
        print(f"[INFO] Verwende vordefinierte Kameraposition: {PREDEFINED_CAMERA_POSITION}")
        K = PREDEFINED_CAMERA_POSITION

        # Wenn Kameraposition vordefiniert ist, überspringe Kalibrierung
        # aber führe trotzdem ROI-Auswahl durch falls nötig
        quad = select_quad(frame, window_name=f"{roi_name} - Quad auswählen", predefined_quad=PREDEFINED_ROI_QUAD)
        if quad is None or len(quad) != 4:
            print("[ERROR] ROI-Auswahl abgebrochen")
            return None

        def dist(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        w1 = dist(quad[0], quad[1])
        w2 = dist(quad[2], quad[3])
        h1 = dist(quad[1], quad[2])
        h2 = dist(quad[3], quad[0])

        W_rect = int(round((w1 + w2) / 2.0))
        H_rect = int(round((h1 + h2) / 2.0))
        W_rect = max(W_rect, 10)
        H_rect = max(H_rect, 10)

        print(f"[INFO] ROI-Größe (1:1 Seitenverhältnis): {W_rect} x {H_rect} px")

        dst_pts = np.array([
            [0, 0],
            [W_rect - 1, 0],
            [W_rect - 1, H_rect - 1],
            [0, H_rect - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(quad, dst_pts)
        roi_preview = cv2.warpPerspective(frame, M, (W_rect, H_rect))

        blind_poly = select_blindspot(
            roi_preview,
            window_name=f"{roi_name} - Blindspot (optional)",
            roi_size=(W_rect, H_rect),
            auto_top_percent=AUTO_BLINDSPOT_TOP_PERCENT if AUTO_BLINDSPOT_ENABLED else 0.0
        )

        # Setze virtuelle Punkte auf None, da keine Kalibrierung durchgeführt wurde
        virtual_points = None
        dists = []

    else:
        # ============================================================
        # NORMALE KALIBRIERUNG MIT LICHTBRECHUNG
        # ============================================================
        quad = select_quad(frame, window_name=f"{roi_name} - Quad auswählen", predefined_quad=PREDEFINED_ROI_QUAD)
        if quad is None or len(quad) != 4:
            print("[ERROR] ROI-Auswahl abgebrochen")
            return None

        def dist(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        w1 = dist(quad[0], quad[1])
        w2 = dist(quad[2], quad[3])
        h1 = dist(quad[1], quad[2])
        h2 = dist(quad[3], quad[0])

        # Behalte das originale Seitenverhältnis bei
        W_rect = int(round((w1 + w2) / 2.0))
        H_rect = int(round((h1 + h2) / 2.0))
        W_rect = max(W_rect, 10)
        H_rect = max(H_rect, 10)

        print(f"[INFO] ROI-Größe (1:1 Seitenverhältnis): {W_rect} x {H_rect} px")

        dst_pts = np.array([
            [0, 0],
            [W_rect - 1, 0],
            [W_rect - 1, H_rect - 1],
            [0, H_rect - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(quad, dst_pts)
        roi_preview = cv2.warpPerspective(frame, M, (W_rect, H_rect))

        blind_poly = select_blindspot(
            roi_preview,
            window_name=f"{roi_name} - Blindspot (optional)",
            roi_size=(W_rect, H_rect),
            auto_top_percent=AUTO_BLINDSPOT_TOP_PERCENT if AUTO_BLINDSPOT_ENABLED else 0.0
        )

        cm_per_px_x = TANK_WIDTH_X / float(W_rect)
        cm_per_px_z = TANK_HEIGHT_Z / float(H_rect)

        print(f"[INFO] Umrechnung: {cm_per_px_x:.4f} cm/px (X), {cm_per_px_z:.4f} cm/px (Z)")

        virtual_points = collect_calibration_points(
            roi_preview,
            roi_name,
            cm_per_px_x,
            cm_per_px_z,
            n_points=4
        )

        if virtual_points is None:
            print("[ERROR] Kalibrierung abgebrochen")
            return None

        print("\n--- KALIBRIERUNGS-ERGEBNISSE ---")
        print(f"Virtuelle Punkte (cm):\n{virtual_points}")
        print(f"Reale Punkte (cm):\n{real_points}")

        try:
            # NEUE KALIBRIERUNG MIT LICHTBRECHUNG
            K = calibrate_camera_with_refraction(virtual_points, real_points)

            print(f"\n>>> KAMERAPOSITION (MIT LICHTBRECHUNG): ({K[0]:.3f}, {K[1]:.3f}, {K[2]:.3f}) cm\n")

            # Berechne Fehler (Abstand Kamera zu Luftstrahlen)
            dists = []
            normal_water_to_air = np.array([0.0, -1.0, 0.0], dtype=float)

            for i, (P_front, P_real) in enumerate(zip(virtual_points, real_points), start=1):
                # Berechne Luftstrahl
                d_water = P_real - P_front
                d_water = d_water / np.linalg.norm(d_water)
                d_air = refract_ray(d_water, normal_water_to_air, N_WATER, N_AIR)

                if d_air is not None:
                    # Berechne Abstand von K zur Luftgeraden
                    P_air_end = P_front - d_air
                    d = distance_point_to_line(K, P_front, P_air_end)
                    dists.append(d)
                    print(f"    Luftstrahl {i}: Abstand = {d:.6f} cm")

            if len(dists) > 0:
                print(f"    Mittelwert: {float(np.mean(dists)):.6f} cm")

            # Speichere Kalibrier-Plot im Output-Ordner
            plot_path = output_dir / "calibration_plot.png"

            plot_calibration_scene(
                save_path=plot_path,
                real_points=real_points,
                virtual_points=virtual_points,
                K=K,
                title=f"Kamera-Kalibrierung: {img_path.name}"
            )

        except Exception as e:
            print(f"[ERROR] Kalibrierung fehlgeschlagen: {e}")
            return None

    # Berechne cm_per_px für spätere Verwendung
    cm_per_px_x = TANK_WIDTH_X / float(W_rect)
    cm_per_px_z = TANK_HEIGHT_Z / float(H_rect)

    # ============================================================
    # ROTESTE PIXEL DETECTION
    # ============================================================
    print(f"\n{'=' * 60}")
    print("ROTESTE PIXEL DETECTION")
    print(f"{'=' * 60}\n")

    print("[INFO] Wähle Boden-ROI (4 Punkte: UL, UR, LR, LL)")
    floor_quad = select_quad(roi_preview, window_name="Boden-ROI auswählen", predefined_quad=PREDEFINED_FLOOR_QUAD)

    if floor_quad is not None and len(floor_quad) == 4:
        def dist_floor(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        floor_w1 = dist_floor(floor_quad[0], floor_quad[1])
        floor_w2 = dist_floor(floor_quad[2], floor_quad[3])
        floor_h1 = dist_floor(floor_quad[1], floor_quad[2])
        floor_h2 = dist_floor(floor_quad[3], floor_quad[0])

        # Boden-ROI: 1:1 Seitenverhältnis (quadratisch)
        floor_w = int(round((floor_w1 + floor_w2) / 2.0))
        floor_h = int(round((floor_h1 + floor_h2) / 2.0))
        floor_size = int(round((floor_w + floor_h) / 2.0))  # Durchschnitt für quadratisches ROI
        floor_size = max(floor_size, 10)

        print(f"[INFO] Boden-ROI-Größe (1:1 quadratisch): {floor_size} x {floor_size} px")

        floor_dst_pts = np.array([
            [0, 0],
            [floor_size - 1, 0],
            [floor_size - 1, floor_size - 1],
            [0, floor_size - 1]
        ], dtype=np.float32)

        M_floor = cv2.getPerspectiveTransform(floor_quad, floor_dst_pts)
        floor_roi = cv2.warpPerspective(roi_preview, M_floor, (floor_size, floor_size))

        # Speichere originales Boden-ROI
        floor_roi_path = output_dir / "floor_roi_original.png"
        cv2.imwrite(str(floor_roi_path), floor_roi)
        print(f"[OK] Originales Boden-ROI gespeichert: {floor_roi_path}")

        # Finde roteste Pixel
        reddest_img, reddest_pixels = find_reddest_pixels_per_line(
            floor_roi,
            num_lines=NUM_VERTICAL_LINES,
            marker_size=PIXEL_MARKER_SIZE
        )

        # Speichere Ergebnis
        reddest_path = output_dir / "reddest_pixels.png"
        cv2.imwrite(str(reddest_path), reddest_img)
        print(f"[OK] Roteste-Pixel-Bild gespeichert: {reddest_path}")

        # ============================================================
        # LASER-LINIEN-REKONSTRUKTION (CALIBRATION) MIT LICHTBRECHUNG
        # ============================================================
        print(f"\n{'=' * 60}")
        print("LASER-LINIEN-REKONSTRUKTION (CALIBRATION) MIT LICHTBRECHUNG")
        print(f"{'=' * 60}\n")

        # 1. Erstelle Polyline durch roteste Pixel (mit optionalem Fitting)
        sampled_points = create_polyline_with_outlier_filter(
            reddest_pixels,
            floor_size,
            slope_threshold=SLOPE_THRESHOLD,
            num_dense_samples=NUM_LASER_POINTS,
            fit_method=CALIBRATION_FIT_METHOD,
            exclude_center_percent=FIT_EXCLUDE_CENTER_PERCENT,
            exclude_edge_percent=FIT_EXCLUDE_EDGE_PERCENT
        )

        if sampled_points is not None:
            # Visualisiere Linie/Kurve
            fitted_line_img = floor_roi.copy()

            if CALIBRATION_FIT_METHOD == 'line':
                # Bei Geraden-Fit: Zeichne eine durchgehende Linie
                if len(sampled_points) >= 2:
                    cv2.line(fitted_line_img, sampled_points[0], sampled_points[-1], (0, 255, 0), 3)
            else:
                # Bei Parabel/None: Zeichne viele kurze Liniensegmente
                for k in range(1, len(sampled_points)):
                    cv2.line(fitted_line_img, sampled_points[k - 1], sampled_points[k], (0, 255, 0), 2)

            fitted_line_path = output_dir / "fitted_laser_line.png"
            cv2.imwrite(str(fitted_line_path), fitted_line_img)

            method_name = {'none': 'Originallinie', 'line': 'Gerade', 'parabola': 'Parabel'}
            print(f"[OK] {method_name.get(CALIBRATION_FIT_METHOD, 'Linie')} gespeichert: {fitted_line_path}")
            print(f"[INFO] {len(sampled_points)} Punkte auf der Linie gesampelt")

            # 3. Berechne inverse Transformation von Boden-ROI zu Haupt-ROI
            M_floor_inv = cv2.getPerspectiveTransform(floor_dst_pts, floor_quad)

            # 4. Rekonstruiere 3D-Punkte MIT LICHTBRECHUNG
            laser_points_3d, sampled_points_stored = reconstruct_laser_line_3d(
                sampled_points,
                K,
                M_floor_inv,
                cm_per_px_x,
                cm_per_px_z,
                store_sampled_points=True
            )


            print(f"[INFO] {len(laser_points_3d)} 3D-Punkte rekonstruiert (MIT LICHTBRECHUNG)")

            if len(laser_points_3d) > 0:
                # 5. Visualisiere 3D-Rekonstruktion
                laser_3d_path = output_dir / "laser_line_3d.png"
                visualize_laser_line_3d(laser_points_3d, K, laser_3d_path)

                # 6. Speichere 3D-Koordinaten
                laser_coords_path = output_dir / "laser_line_3d_coords.txt"
                with open(laser_coords_path, 'w') as f:
                    f.write("LASER-LINIEN 3D-KOORDINATEN (MIT LICHTBRECHUNG)\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Brechungsindizes: n_air = {N_AIR}, n_water = {N_WATER}\n")
                    f.write(f"Anzahl Punkte: {len(laser_points_3d)}\n")
                    f.write(f"Kameraposition: ({K[0]:.3f}, {K[1]:.3f}, {K[2]:.3f}) cm\n\n")
                    f.write("X [cm]\tY [cm]\tZ [cm]\n")
                    f.write("-" * 50 + "\n")
                    for pt in laser_points_3d:
                        f.write(f"{pt[0]:.4f}\t{pt[1]:.4f}\t{pt[2]:.4f}\n")

                print(f"[OK] 3D-Koordinaten gespeichert: {laser_coords_path}")

                # Statistiken
                x_min, x_max = laser_points_3d[:, 0].min(), laser_points_3d[:, 0].max()
                y_min, y_max = laser_points_3d[:, 1].min(), laser_points_3d[:, 1].max()

                print(f"\n[INFO] Laserlinie Statistiken:")
                print(f"       X-Bereich: {x_min:.2f} bis {x_max:.2f} cm (Spannweite: {x_max - x_min:.2f} cm)")
                print(f"       Y-Bereich: {y_min:.2f} bis {y_max:.2f} cm (Spannweite: {y_max - y_min:.2f} cm)")
                print(f"       Z-Position: 0.00 cm (Boden)")

        # Zeige Ergebnis
        cv2.imshow("Roteste Pixel", reddest_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Speichere Koordinaten
        coords_path = output_dir / "reddest_pixels_coords.txt"
        with open(coords_path, 'w') as f:
            f.write("ROTESTE PIXEL KOORDINATEN\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Anzahl vertikaler Linien: {NUM_VERTICAL_LINES}\n")
            f.write(f"Marker-Größe: {PIXEL_MARKER_SIZE} px\n")
            f.write(f"Boden-ROI-Größe: {floor_size} x {floor_size} px\n\n")
            f.write(f"Gefundene roteste Pixel: {len(reddest_pixels)}\n\n")
            for i, (x, y) in enumerate(reddest_pixels, 1):
                f.write(f"  Linie {i}: x={x}, y={y}\n")

        print(f"[OK] Koordinaten gespeichert: {coords_path}")

    else:
        print("[INFO] Boden-ROI-Auswahl übersprungen")
        reddest_pixels = []
        laser_points_3d = []
        floor_quad = None
        floor_size = None
        M_floor = None

    # Speichere Ergebnis-Textdatei im Output-Ordner
    result_txt = output_dir / "calibration_results.txt"
    with open(result_txt, 'w') as f:
        f.write(f"KAMERA-KALIBRIERUNG (MIT LICHTBRECHUNG)\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Brechungsindizes: n_air = {N_AIR}, n_water = {N_WATER}\n")
        f.write(f"Bild: {img_path.name}\n")
        f.write(f"Kamera-Kalibrierung: {'VERWENDET' if CAMERA_CALIBRATION_NPZ else 'KEINE'}\n")
        if CAMERA_CALIBRATION_NPZ:
            f.write(f"  NPZ-Datei: {CAMERA_CALIBRATION_NPZ}\n")
            f.write(f"  Alpha: {UNDISTORT_ALPHA}\n")
        f.write(f"ROI-Größe: {W_rect} x {H_rect} px (1:1 Seitenverhältnis)\n\n")
        f.write(f"KAMERAPOSITION (cm):\n")
        f.write(f"  X = {K[0]:.3f}\n")
        f.write(f"  Y = {K[1]:.3f}\n")
        f.write(f"  Z = {K[2]:.3f}\n")

        if PREDEFINED_CAMERA_POSITION is not None:
            f.write(f"  (Vordefiniert)\n\n")
        else:
            f.write(f"\nFEHLERANALYSE (MIT LICHTBRECHUNG):\n")
            for i, d in enumerate(dists, 1):
                f.write(f"  Luftstrahl {i}: {d:.6f} cm\n")
            f.write(f"  Mittelwert: {np.mean(dists):.6f} cm\n\n")
            f.write(f"VIRTUELLE PUNKTE (cm):\n{virtual_points}\n\n")
            f.write(f"REALE PUNKTE (cm):\n{real_points}\n\n")

        f.write(f"ROTESTE PIXEL DETECTION:\n")
        f.write(f"  Anzahl vertikaler Linien: {NUM_VERTICAL_LINES}\n")
        f.write(f"  Marker-Größe: {PIXEL_MARKER_SIZE} px\n")
        f.write(f"  Gefundene Pixel: {len(reddest_pixels)}\n\n")

        f.write(f"LASER-LINIEN-REKONSTRUKTION (MIT LICHTBRECHUNG):\n")
        f.write(f"  Anzahl 3D-Punkte: {len(laser_points_3d)}\n")
        if len(laser_points_3d) > 0:
            x_min, x_max = laser_points_3d[:, 0].min(), laser_points_3d[:, 0].max()
            y_min, y_max = laser_points_3d[:, 1].min(), laser_points_3d[:, 1].max()
            f.write(f"  X-Bereich: {x_min:.2f} bis {x_max:.2f} cm\n")
            f.write(f"  Y-Bereich: {y_min:.2f} bis {y_max:.2f} cm\n")

    print(f"\n[OK] Ergebnisse gespeichert: {result_txt}")

    if IS_MACOS:
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        time.sleep(0.5)

    # Erstelle Kalibrierungsdaten-Dictionary
    calibration_result = {
        'camera_position': K,
        'virtual_points': virtual_points,
        'real_points': real_points,
        'roi_size': (W_rect, H_rect),
        'roi_quad': quad,
        'transform_matrix': M,
        'blindspots': blind_poly,
        'mean_error': np.mean(dists) if len(dists) > 0 else 0.0,
        'reddest_pixels': reddest_pixels,
        'laser_points_3d': laser_points_3d if 'laser_points_3d' in locals() else [],
        'sampled_points_floor_roi': sampled_points_stored if 'sampled_points_stored' in locals() else [],
        'floor_transform_inv': M_floor_inv if 'M_floor_inv' in locals() else None,
        'output_dir': output_dir,
        'undistorted': CAMERA_CALIBRATION_NPZ is not None,
        'predefined_camera': PREDEFINED_CAMERA_POSITION is not None,
        'floor_quad': floor_quad,
        'floor_size': floor_size,
        'floor_transform_matrix': M_floor,
        'cm_per_px_x': cm_per_px_x,
        'cm_per_px_z': cm_per_px_z,
        'n_air': N_AIR,
        'n_water': N_WATER
    }

    # Speichere Kalibrierungsdaten
    calibration_data_path = output_dir / "calibration_data.pkl"
    save_calibration_data(calibration_result, calibration_data_path)

    return calibration_result


def process_measurement_image(measurement_image_path: str, calibration_data: dict):
    """
    Verarbeitet ein Messbild mit den Kalibrierungsdaten vom Kalibrierungsbild.

    Args:
        measurement_image_path: Pfad zum Messbild
        calibration_data: Dictionary mit Kalibrierungsdaten

    Returns:
        Dictionary mit Messergebnissen
    """
    # ============================================================
    # MESSBILD LADEN UND ENTZERREN
    # ============================================================
    frame = load_and_undistort_image(
        measurement_image_path,
        calibration_npz_path=CAMERA_CALIBRATION_NPZ,
        alpha=UNDISTORT_ALPHA
    )

    if frame is None:
        print(f"[ERROR] Messbild konnte nicht geladen werden")
        return None

    img_path = Path(measurement_image_path)

    # Erstelle Output-Ordner
    output_dir = img_path.parent / f"{img_path.stem}_measurement"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Measurement Output-Ordner: {output_dir}")

    print(f"\n{'=' * 60}")
    print(f"MESSBILD-VERARBEITUNG: {img_path.name}")
    print(f"{'=' * 60}\n")

    # Lade Kalibrierungsdaten
    K = calibration_data['camera_position']
    M = calibration_data['transform_matrix']
    W_rect, H_rect = calibration_data['roi_size']
    floor_quad = calibration_data['floor_quad']
    floor_size = calibration_data['floor_size']
    M_floor = calibration_data['floor_transform_matrix']
    cm_per_px_x = calibration_data['cm_per_px_x']
    cm_per_px_z = calibration_data['cm_per_px_z']

    print(f"[INFO] Verwende Kalibrierungsdaten:")
    print(f"       Kameraposition: {K}")
    print(f"       ROI-Größe: {W_rect} x {H_rect} px")
    print(f"       Boden-ROI-Größe: {floor_size} x {floor_size} px")
    print(f"       Brechungsindizes: n_air = {N_AIR}, n_water = {N_WATER}")

    # Wende ROI-Transformation an
    roi_preview = cv2.warpPerspective(frame, M, (W_rect, H_rect))
    roi_preview_path = output_dir / "roi_preview.png"
    cv2.imwrite(str(roi_preview_path), roi_preview)
    print(f"[OK] ROI-Preview gespeichert: {roi_preview_path}")

    # Wende Boden-ROI-Transformation an
    if floor_quad is not None and M_floor is not None:
        floor_roi = cv2.warpPerspective(roi_preview, M_floor, (floor_size, floor_size))
        floor_roi_path = output_dir / "floor_roi_original.png"
        cv2.imwrite(str(floor_roi_path), floor_roi)
        print(f"[OK] Boden-ROI gespeichert: {floor_roi_path}")

        # Finde roteste Pixel
        print(f"\n{'=' * 60}")
        print("ROTESTE PIXEL DETECTION (MESSBILD)")
        print(f"{'=' * 60}\n")

        reddest_img, reddest_pixels = find_reddest_pixels_per_line(
            floor_roi,
            num_lines=NUM_VERTICAL_LINES,
            marker_size=PIXEL_MARKER_SIZE
        )

        # Speichere Ergebnis
        reddest_path = output_dir / "reddest_pixels.png"
        cv2.imwrite(str(reddest_path), reddest_img)
        print(f"[OK] Roteste-Pixel-Bild gespeichert: {reddest_path}")

        # Zeige Ergebnis
        cv2.imshow("Roteste Pixel (Messbild)", reddest_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Speichere Koordinaten
        coords_path = output_dir / "reddest_pixels_coords.txt"
        with open(coords_path, 'w') as f:
            f.write("ROTESTE PIXEL KOORDINATEN (MESSBILD)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Anzahl vertikaler Linien: {NUM_VERTICAL_LINES}\n")
            f.write(f"Marker-Größe: {PIXEL_MARKER_SIZE} px\n")
            f.write(f"Boden-ROI-Größe: {floor_size} x {floor_size} px\n\n")
            f.write(f"Gefundene roteste Pixel: {len(reddest_pixels)}\n\n")
            for i, (x, y) in enumerate(reddest_pixels, 1):
                f.write(f"  Linie {i}: x={x}, y={y}\n")

        print(f"[OK] Koordinaten gespeichert: {coords_path}")

    else:
        print("[INFO] Keine Boden-ROI-Daten vorhanden")
        reddest_pixels = []

    # Speichere Ergebnis-Textdatei
    result_txt = output_dir / "measurement_results.txt"
    with open(result_txt, 'w') as f:
        f.write(f"MESSBILD-VERARBEITUNG (MIT LICHTBRECHUNG)\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Brechungsindizes: n_air = {N_AIR}, n_water = {N_WATER}\n")
        f.write(f"Messbild: {img_path.name}\n")
        f.write(f"Kalibrierungsdaten verwendet: JA\n")
        f.write(f"Kameraposition: ({K[0]:.3f}, {K[1]:.3f}, {K[2]:.3f}) cm\n")
        f.write(f"ROI-Größe: {W_rect} x {H_rect} px\n")
        f.write(f"Boden-ROI-Größe: {floor_size} x {floor_size} px\n\n")
        f.write(f"ROTESTE PIXEL DETECTION:\n")
        f.write(f"  Anzahl vertikaler Linien: {NUM_VERTICAL_LINES}\n")
        f.write(f"  Marker-Größe: {PIXEL_MARKER_SIZE} px\n")
        f.write(f"  Gefundene Pixel: {len(reddest_pixels)}\n")

    print(f"\n[OK] Messergebnisse gespeichert: {result_txt}")

    # Rekonstruiere Measurement-Laserlinie
    measurement_laser_3d = []
    if len(reddest_pixels) > 0:
        print(f"\n{'=' * 60}")
        print("LASER-LINIEN-REKONSTRUKTION (MEASUREMENT)")
        print(f"{'=' * 60}\n")

        # Erstelle Polyline
        meas_polyline = create_polyline_with_outlier_filter(
            reddest_pixels,
            floor_size,
            slope_threshold=SLOPE_THRESHOLD,
            num_dense_samples=NUM_DENSE_SAMPLES,
            fit_method='none',
            exclude_center_percent=0.0,
            exclude_edge_percent=0.0
        )

        if meas_polyline is not None and len(meas_polyline) > 0:
            # Berechne inverse Transformation
            floor_dst_pts = np.array([
                [0, 0],
                [floor_size - 1, 0],
                [floor_size - 1, floor_size - 1],
                [0, floor_size - 1]
            ], dtype=np.float32)

            M_floor_inv = cv2.getPerspectiveTransform(floor_dst_pts, floor_quad)

            # Rekonstruiere 3D-Punkte
            measurement_laser_3d = reconstruct_laser_line_3d(
                meas_polyline,
                K,
                M_floor_inv,
                cm_per_px_x,
                cm_per_px_z
            )

            print(f"[INFO] {len(measurement_laser_3d)} Measurement 3D-Punkte rekonstruiert")

    return {
        'reddest_pixels': reddest_pixels,
        'laser_points_3d': measurement_laser_3d,
        'output_dir': output_dir,
        'camera_position': K
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # SCHRITT 1: KALIBRIERUNG
    print("\n" + "=" * 80)
    print("SCHRITT 1: KALIBRIERUNGSBILD VERARBEITEN (MIT LICHTBRECHUNG)")
    print("=" * 80 + "\n")

    calibration_result = calibrate_camera(
        image_path=CALIBRATION_IMAGE_PATH,
        real_points=REAL_POINTS,
        roi_name="Frontkamera"
    )

    if calibration_result:
        print(f"\n{'=' * 60}")
        print("KALIBRIERUNG ERFOLGREICH! (MIT LICHTBRECHUNG)")
        print(f"{'=' * 60}")
        print(f"Brechungsindizes: n_air = {N_AIR}, n_water = {N_WATER}")
        print(f"Kamera-Kalibrierung: {'VERWENDET' if calibration_result['undistorted'] else 'KEINE'}")
        print(f"Kameraposition: {calibration_result['camera_position']}")

        if calibration_result['predefined_camera']:
            print(f"Kameraposition: VORDEFINIERT")
        else:
            print(f"Mittlerer Fehler: {calibration_result['mean_error']:.6f} cm")

        print(f"Roteste Pixel gefunden: {len(calibration_result['reddest_pixels'])}")
        print(f"3D-Laserpunkte rekonstruiert: {len(calibration_result['laser_points_3d'])}")

        # SCHRITT 2: MESSBILD VERARBEITEN
        print("\n" + "=" * 80)
        print("SCHRITT 2: MESSBILD VERARBEITEN (MIT LICHTBRECHUNG)")
        print("=" * 80 + "\n")

        measurement_result = process_measurement_image(
            measurement_image_path=MEASUREMENT_IMAGE_PATH,
            calibration_data=calibration_result
        )

        if measurement_result:
            print(f"\n{'=' * 60}")
            print("MESSBILD-VERARBEITUNG ERFOLGREICH! (MIT LICHTBRECHUNG)")
            print(f"{'=' * 60}")
            print(f"Roteste Pixel gefunden: {len(measurement_result['reddest_pixels'])}")

            # SCHRITT 3: PROFIL-VERGLEICH
            print("\n" + "=" * 80)
            print("SCHRITT 3: PROFIL-VERGLEICH (MIT LICHTBRECHUNG)")
            print("=" * 80 + "\n")

            comparison_result = compare_calibration_measurement_profiles(
                calibration_data=calibration_result,
                measurement_result=measurement_result,
                output_path=measurement_result['output_dir'] / "profile_comparison.png"
            )

            if comparison_result:
                print(f"\n{'=' * 60}")
                print("PROFIL-VERGLEICH ERFOLGREICH! (MIT LICHTBRECHUNG)")
                print(f"{'=' * 60}")
                print(f"Höhenprofil-Statistiken:")
                print(f"  Min: {comparison_result['stats']['min']:.3f} cm")
                print(f"  Max: {comparison_result['stats']['max']:.3f} cm")
                print(f"  Mittelwert: {comparison_result['stats']['mean']:.3f} cm")
                print(f"  Standardabweichung: {comparison_result['stats']['std']:.3f} cm")
            else:
                print("\n[ERROR] Profil-Vergleich fehlgeschlagen")

            # SCHRITT 4: KOMBINIERTE 3D-VISUALISIERUNG
            print("\n" + "=" * 80)
            print("SCHRITT 4: KOMBINIERTE 3D-VISUALISIERUNG")
            print("=" * 80 + "\n")

            visualize_combined_3d_scene(
                calibration_data=calibration_result,
                measurement_data=measurement_result,
                output_path=measurement_result['output_dir'] / "combined_3d_scene.png",
                show_refraction_comparison=True
            )

            print(f"\n{'=' * 60}")
            print("KOMBINIERTE 3D-VISUALISIERUNG ERFOLGREICH!")
            print(f"{'=' * 60}")

        else:  # <-- KORREKT: 4 Spaces Einrückung (gleiche Ebene wie "if measurement_result:")
            print("\n[ERROR] Messbild-Verarbeitung fehlgeschlagen")


    else:
        print("\n[ERROR] Kalibrierung fehlgeschlagen")
