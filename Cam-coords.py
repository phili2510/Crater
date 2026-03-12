from __future__ import annotations

import cv2
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
import platform
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =======================
# EINGABEN
# =======================
IMAGE_PATH = '/Users/philippadelbrecht/PycharmProjects/Crater/GP011245.JPG'  # Pfad zum Kalibrierbild

# =======================
# REALE KOORDINATEN DER KALIBRIERPUNKTE (3D-Weltkoordinaten)
# =======================
# Reihenfolge beim Klicken: UL (oben links), UR (oben rechts), LR (unten rechts), LL (unten links)
REAL_POINTS = np.array([
    [0.0, 60.8, 36.0],  # UL: (x, y, z) in cm - oben links an Wasseroberfläche
    [60.8, 60.8, 36.0],  # UR: oben rechts an Wasseroberfläche
    [60.8, 60.8, 0.0],  # LR: unten rechts am Boden
    [0.0, 60.8, 0.0],  # LL: unten links am Boden
], dtype=float)

# =======================
# AQUARIUM-GEOMETRIE
# =======================
TANK_WIDTH_X = 60.8  # cm
TANK_DEPTH_Y = 60.8  # cm
TANK_HEIGHT_Z = 36.0  # cm (Füllhöhe)

# =======================
# ROI-KONFIGURATION
# =======================
# Gewünschtes ROI-Seitenverhältnis (Breite / Höhe)
DESIRED_RATIO = TANK_WIDTH_X / TANK_HEIGHT_Z  # 60.8 / 36.0 = 1.689

# =======================
# BLINDSPOT-EINSTELLUNGEN
# =======================
AUTO_BLINDSPOT_ENABLED = True
AUTO_BLINDSPOT_TOP_PERCENT = 5.0
AUTO_EDGE_BLINDSPOT_ENABLED = True
AUTO_EDGE_BLINDSPOT_PERCENT = 1.0

# =======================
# LASER-GRID-TRACKING EINSTELLUNGEN
# =======================
LASER_EDGE_MARGIN = 5
LASER_GAP_TOLERANCE = 5

# HSV-Bereiche für Rot-Erkennung [H_min, H_max, S_min, V_min]
# Rot liegt bei 0° und 180° im HSV-Farbraum, daher zwei Bereiche
LASER_RED_LOWER_1 = [0, 10, 100, 100]      # [H_min, H_max, S_min, V_min]
LASER_RED_LOWER_2 = [170, 180, 100, 100]   # Zweiter Rot-Bereich

# System-Erkennung für macOS-Fixes
import platform
IS_MACOS = platform.system() == 'Darwin'
WINDOW_DELAY = 1.5 if IS_MACOS else 0.5



# ============================================================
# HILFSFUNKTIONEN
# ============================================================
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


def select_quad(frame, window_name="ROI-Quad"):
    """Wähle 4 Punkte für ROI-Quad aus (macOS-optimiert)."""
    print(f"[INFO] Starte {window_name} - Klicke 4 Punkte (ESC=abbrechen)")

    unique_name = f"{window_name}_{int(time.time() * 1000) % 10000}"

    # Längere Wartezeit auf macOS
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
        # Fenster erstellen mit Fehlerbehandlung
        try:
            cv2.namedWindow(unique_name, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow(unique_name, 1280, 720)

            if IS_MACOS:
                # Mehrere waitKey-Aufrufe für macOS
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

                # Längeres waitKey auf macOS
                wait_time = 50 if IS_MACOS else 1
                key = cv2.waitKey(wait_time) & 0xFF

                if key == 27:  # ESC
                    print("[INFO] Abgebrochen")
                    break
                elif key == ord('r'):
                    pts = []
                    print("[INFO] Punkte zurückgesetzt")
                elif key in (13, 32) and len(pts) == 4:  # ENTER/SPACE
                    print("[INFO] 4 Punkte ausgewählt")

                    # Sicheres Schließen
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

        # Cleanup
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

            # Automatische Blindspots (grün)
            for auto_poly in auto_blindspots:
                auto_pts_scaled = [(int(x * scale), int(y * scale)) for x, y in auto_poly]
                overlay = disp.copy()
                cv2.fillPoly(overlay, [np.array(auto_pts_scaled, dtype=np.int32)], color=(0, 255, 0))
                cv2.addWeighted(overlay, 0.3, disp, 0.7, 0, disp)
                cv2.polylines(disp, [np.array(auto_pts_scaled, dtype=np.int32)],
                              isClosed=True, color=(0, 255, 0), thickness=2)

            # Fertige manuelle Polygone (blau)
            for poly_idx, polygon in enumerate(finished_polygons):
                poly_scaled = [(int(x * scale), int(y * scale)) for x, y in polygon]
                overlay = disp.copy()
                cv2.fillPoly(overlay, [np.array(poly_scaled, dtype=np.int32)], color=(255, 100, 0))
                cv2.addWeighted(overlay, 0.3, disp, 0.7, 0, disp)
                cv2.polylines(disp, [np.array(poly_scaled, dtype=np.int32)],
                              isClosed=True, color=(255, 100, 0), thickness=2)

            # Aktuelles Polygon (rot)
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

            if key == 27:  # ESC
                if len(current_polygon) >= 3:
                    finished_polygons.append(current_polygon[:])
                safe_destroy_window(unique_name)
                return auto_blindspots + finished_polygons

            elif key == ord('r'):
                if current_polygon:
                    print(f"[INFO] Polygon zurückgesetzt")
                    current_polygon = []

            elif key in (13, 32):  # ENTER
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

            if key == 27:  # ESC
                break
            elif key == ord('r'):
                pts = []
            elif key in (13, 32) and len(pts) == n_points:
                safe_destroy_window(unique_name)

                # Konvertiere zu 3D-Punkten (Frontkamera: y=konstant)
                virtual = []
                for (u, v) in pts:
                    u_cm = float(u) * cm_per_px_x
                    z_from_top = float(v) * cm_per_px_z
                    z_cm = TANK_HEIGHT_Z - z_from_top
                    virtual.append([u_cm, 0.0, z_cm])  # y=0 (Frontkamera)

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


def plot_calibration_scene(save_path: Path, real_points: np.ndarray, virtual_points: np.ndarray,
                           K: np.ndarray, title: str):
    """Erstellt 3D-Visualisierung der Kalibrierung."""
    LX = TANK_WIDTH_X
    LY = TANK_DEPTH_Y
    LZ = TANK_HEIGHT_Z

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Aquarium-Kanten
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

    # Punkte
    ax.scatter(real_points[:, 0], real_points[:, 1], real_points[:, 2],
               s=100, c='red', marker='o', label='Reale Punkte')
    ax.scatter(virtual_points[:, 0], virtual_points[:, 1], virtual_points[:, 2],
               s=100, c='green', marker='^', label='Virtuelle Punkte')

    # Sichtlinien
    for vp, rp in zip(virtual_points, real_points):
        d = rp - vp
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d_unit = d / n
        t_star = np.dot(K - vp, d_unit)
        P_close = vp + t_star * d_unit

        ax.plot([rp[0], P_close[0]], [rp[1], P_close[1]], [rp[2], P_close[2]],
                'r--', linewidth=2.0, alpha=0.7)

    # Kamera
    ax.scatter([K[0]], [K[1]], [K[2]], s=200, c='purple', marker='*',
               label=f'Kamera\n({K[0]:.1f}, {K[1]:.1f}, {K[2]:.1f})')

    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    ax.set_title(title)
    ax.legend()

    # Gleiche Achsenskalierung
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


def detect_red_pixels(roi_img):
    """Erkennt rote Pixel im Bild mittels HSV-Filterung."""
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    # Zwei Masken für Rot (da Rot bei 0° und 180° liegt)
    # LASER_RED_LOWER_1 = [H_min, H_max, S_min, V_min]
    mask1 = cv2.inRange(hsv,
                        (LASER_RED_LOWER_1[0], LASER_RED_LOWER_1[2], LASER_RED_LOWER_1[3]),
                        (LASER_RED_LOWER_1[1], 255, 255))

    # LASER_RED_LOWER_2 = [H_min, H_max, S_min, V_min]
    mask2 = cv2.inRange(hsv,
                        (LASER_RED_LOWER_2[0], LASER_RED_LOWER_2[2], LASER_RED_LOWER_2[3]),
                        (LASER_RED_LOWER_2[1], 255, 255))

    mask = cv2.bitwise_or(mask1, mask2)
    return mask


def detect_laser_lines(floor_roi):
    """Erkennt Laserlinien am linken und unteren Rand des Boden-ROI."""
    h, w = floor_roi.shape[:2]

    # Rote Pixel erkennen
    red_mask = detect_red_pixels(floor_roi)

    # Linke Kante (x-Achse): von x=0 bis x=LASER_EDGE_MARGIN
    left_edge_mask = red_mask[:, :LASER_EDGE_MARGIN]

    # Untere Kante (y-Achse): von y=(h-LASER_EDGE_MARGIN) bis y=h
    bottom_edge_mask = red_mask[h - LASER_EDGE_MARGIN:, :]

    # Finde vertikale Linien am linken Rand (konstantes x, verschiedene y)
    vertical_lines = []
    for x in range(LASER_EDGE_MARGIN):
        y_coords = np.where(left_edge_mask[:, x] > 0)[0]
        if len(y_coords) > 0:
            lines = group_points_into_lines(y_coords.tolist(), LASER_GAP_TOLERANCE)
            for line in lines:
                center_y = int(np.mean(line))
                vertical_lines.append((x, center_y))

    # Finde horizontale Linien am unteren Rand (konstantes y, verschiedene x)
    horizontal_lines = []
    for y_offset in range(LASER_EDGE_MARGIN):
        y = h - LASER_EDGE_MARGIN + y_offset
        x_coords = np.where(bottom_edge_mask[y_offset, :] > 0)[0]
        if len(x_coords) > 0:
            lines = group_points_into_lines(x_coords.tolist(), LASER_GAP_TOLERANCE)
            for line in lines:
                center_x = int(np.mean(line))
                horizontal_lines.append((center_x, y))

    print(f"[INFO] Gefunden: {len(vertical_lines)} vertikale Linien, {len(horizontal_lines)} horizontale Linien")

    return vertical_lines, horizontal_lines, red_mask


def visualize_laser_lines(floor_roi, vertical_lines, horizontal_lines, save_path):
    """Visualisiert gefundene Laserlinien mit großen Punkten."""
    vis_img = floor_roi.copy()

    # Zeichne vertikale Linien (grün)
    for x, y in vertical_lines:
        cv2.circle(vis_img, (x, y), 10, (0, 255, 0), -1)
        cv2.putText(vis_img, f"V", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Zeichne horizontale Linien (blau)
    for x, y in horizontal_lines:
        cv2.circle(vis_img, (x, y), 10, (255, 0, 0), -1)
        cv2.putText(vis_img, f"H", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Info-Text
    info_text = [
        f"Vertikale Linien (grün): {len(vertical_lines)}",
        f"Horizontale Linien (blau): {len(horizontal_lines)}",
        f"Rand-Margin: {LASER_EDGE_MARGIN}px",
        f"Gap-Toleranz: {LASER_GAP_TOLERANCE}px"
    ]

    for i, text in enumerate(info_text):
        cv2.putText(vis_img, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Speichern
    cv2.imwrite(str(save_path), vis_img)
    print(f"[OK] Laser-Visualisierung gespeichert: {save_path}")

    return vis_img


def calibrate_camera(image_path: str, real_points: np.ndarray, roi_name: str = "Kamera"):
    """Hauptfunktion für Kamerakalibrierung aus Einzelbild."""

    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[ERROR] Bild nicht gefunden: {image_path}")
        return None

    # Lade Bild
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[ERROR] Bild konnte nicht geladen werden: {image_path}")
        return None

    print(f"\n{'=' * 60}")
    print(f"KAMERA-KALIBRIERUNG: {img_path.name}")
    print(f"{'=' * 60}\n")

    # 1) ROI-Quad auswählen
    quad = select_quad(frame, window_name=f"{roi_name} - Quad auswählen")
    if quad is None or len(quad) != 4:
        print("[ERROR] ROI-Auswahl abgebrochen")
        return None

    # 2) Berechne ROI-Größe mit gewünschtem Seitenverhältnis
    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    w1 = dist(quad[0], quad[1])
    w2 = dist(quad[2], quad[3])
    h1 = dist(quad[1], quad[2])
    h2 = dist(quad[3], quad[0])

    W_est = (w1 + w2) / 2.0
    H_est = (h1 + h2) / 2.0

    W_rect = max(W_est, 10.0)
    H_rect = max(H_est, 10.0)

    current_ratio = W_rect / H_rect

    if current_ratio > DESIRED_RATIO:
        W_rect = H_rect * DESIRED_RATIO
    else:
        H_rect = W_rect / DESIRED_RATIO

    W_rect = int(round(W_rect))
    H_rect = int(round(H_rect))
    W_rect = max(W_rect, 10)
    H_rect = max(H_rect, 10)

    print(f"[INFO] ROI-Größe: {W_rect} x {H_rect} px (Verhältnis: {W_rect / H_rect:.3f})")

    # 3) Perspektivische Transformation
    dst_pts = np.array([
        [0, 0],
        [W_rect - 1, 0],
        [W_rect - 1, H_rect - 1],
        [0, H_rect - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst_pts)
    roi_preview = cv2.warpPerspective(frame, M, (W_rect, H_rect))

    # 4) Blindspot-Auswahl (optional)
    blind_poly = select_blindspot(
        roi_preview,
        window_name=f"{roi_name} - Blindspot (optional)",
        roi_size=(W_rect, H_rect),
        auto_top_percent=AUTO_BLINDSPOT_TOP_PERCENT if AUTO_BLINDSPOT_ENABLED else 0.0
    )

    # 5) Pixel-zu-cm Umrechnung
    cm_per_px_x = TANK_WIDTH_X / float(W_rect)
    cm_per_px_z = TANK_HEIGHT_Z / float(H_rect)

    print(f"[INFO] Umrechnung: {cm_per_px_x:.4f} cm/px (X), {cm_per_px_z:.4f} cm/px (Z)")

    # 6) Kalibrierpunkte auswählen
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

    # 7) Kameraposition berechnen
    print("\n--- KALIBRIERUNGS-ERGEBNISSE ---")
    print(f"Virtuelle Punkte (cm):\n{virtual_points}")
    print(f"Reale Punkte (cm):\n{real_points}")

    try:
        K = best_point_to_lines(virtual_points, real_points)

        print(f"\n>>> KAMERAPOSITION: ({K[0]:.3f}, {K[1]:.3f}, {K[2]:.3f}) cm\n")

        # Fehleranalyse
        dists = []
        for i, (P0, P1) in enumerate(zip(virtual_points, real_points), start=1):
            d = distance_point_to_line(K, P0, P1)
            dists.append(d)
            print(f"    Gerade {i}: Abstand = {d:.6f} cm")
        print(f"    Mittelwert: {float(np.mean(dists)):.6f} cm")

        # 8) Visualisierung
        output_dir = img_path.parent
        plot_path = output_dir / f"calibration_{img_path.stem}.png"

        plot_calibration_scene(
            save_path=plot_path,
            real_points=real_points,
            virtual_points=virtual_points,
            K=K,
            title=f"Kamera-Kalibrierung: {img_path.name}"
        )

        # 9) LASER-GRID-TRACKING
        print(f"\n{'=' * 60}")
        print("LASER-GRID-TRACKING")
        print(f"{'=' * 60}\n")

        # Boden-ROI auswählen (60.8cm x 60.8cm)
        print("[INFO] Wähle Boden-ROI (4 Punkte: UL, UR, LR, LL)")
        floor_quad = select_quad(roi_preview, window_name="Boden-ROI auswählen")

        if floor_quad is not None and len(floor_quad) == 4:
            # Berechne Boden-ROI-Größe (quadratisch: 60.8cm x 60.8cm)
            floor_w1 = dist(floor_quad[0], floor_quad[1])
            floor_w2 = dist(floor_quad[2], floor_quad[3])
            floor_h1 = dist(floor_quad[1], floor_quad[2])
            floor_h2 = dist(floor_quad[3], floor_quad[0])

            floor_size = int(round((floor_w1 + floor_w2 + floor_h1 + floor_h2) / 4.0))
            floor_size = max(floor_size, 10)

            print(f"[INFO] Boden-ROI-Größe: {floor_size} x {floor_size} px")

            # Perspektivische Transformation für Boden
            floor_dst_pts = np.array([
                [0, 0],
                [floor_size - 1, 0],
                [floor_size - 1, floor_size - 1],
                [0, floor_size - 1]
            ], dtype=np.float32)

            M_floor = cv2.getPerspectiveTransform(floor_quad, floor_dst_pts)
            floor_roi = cv2.warpPerspective(roi_preview, M_floor, (floor_size, floor_size))

            # Laser-Linien erkennen
            vertical_lines, horizontal_lines, red_mask = detect_laser_lines(floor_roi)

            # Visualisierung speichern
            laser_vis_path = output_dir / f"laser_lines_{img_path.stem}.png"
            visualize_laser_lines(floor_roi, vertical_lines, horizontal_lines, laser_vis_path)

            # Debug: Rote Maske speichern
            red_mask_path = output_dir / f"laser_red_mask_{img_path.stem}.png"
            cv2.imwrite(str(red_mask_path), red_mask)
            print(f"[OK] Rote Maske gespeichert: {red_mask_path}")

        else:
            print("[INFO] Boden-ROI-Auswahl übersprungen")
            vertical_lines = []
            horizontal_lines = []

        # 10) Ergebnisse speichern
        result_txt = output_dir / f"calibration_{img_path.stem}.txt"
        with open(result_txt, 'w') as f:
            f.write(f"KAMERA-KALIBRIERUNG\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Bild: {img_path.name}\n")
            f.write(f"ROI-Größe: {W_rect} x {H_rect} px\n\n")
            f.write(f"KAMERAPOSITION (cm):\n")
            f.write(f"  X = {K[0]:.3f}\n")
            f.write(f"  Y = {K[1]:.3f}\n")
            f.write(f"  Z = {K[2]:.3f}\n\n")
            f.write(f"FEHLERANALYSE:\n")
            for i, d in enumerate(dists, 1):
                f.write(f"  Gerade {i}: {d:.6f} cm\n")
            f.write(f"  Mittelwert: {np.mean(dists):.6f} cm\n\n")
            f.write(f"VIRTUELLE PUNKTE (cm):\n{virtual_points}\n\n")
            f.write(f"REALE PUNKTE (cm):\n{real_points}\n\n")
            f.write(f"LASER-GRID-TRACKING:\n")
            f.write(f"  Vertikale Linien: {len(vertical_lines)}\n")
            f.write(f"  Horizontale Linien: {len(horizontal_lines)}\n")
            f.write(f"  Rand-Margin: {LASER_EDGE_MARGIN} px\n")
            f.write(f"  Gap-Toleranz: {LASER_GAP_TOLERANCE} px\n")


        print(f"\n[OK] Ergebnisse gespeichert: {result_txt}")

        # macOS: Alle Fenster sauber schließen
        if IS_MACOS:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
            time.sleep(0.5)

        return {
            'camera_position': K,
            'virtual_points': virtual_points,
            'real_points': real_points,
            'roi_size': (W_rect, H_rect),
            'transform_matrix': M,
            'blindspots': blind_poly,
            'mean_error': np.mean(dists),
            'laser_vertical_lines': vertical_lines,
            'laser_horizontal_lines': horizontal_lines
        }


    except Exception as e:
        print(f"[ERROR] Kalibrierung fehlgeschlagen: {e}")
        return None


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    result = calibrate_camera(
        image_path=IMAGE_PATH,
        real_points=REAL_POINTS,
        roi_name="Frontkamera"
    )

    if result:
        print(f"\n{'=' * 60}")
        print("KALIBRIERUNG ERFOLGREICH!")
        print(f"{'=' * 60}")
        print(f"Kameraposition: {result['camera_position']}")
        print(f"Mittlerer Fehler: {result['mean_error']:.6f} cm")
        print(f"Vertikale Laserlinien: {len(result['laser_vertical_lines'])}")
        print(f"Horizontale Laserlinien: {len(result['laser_horizontal_lines'])}")
    else:
        print("\n[ERROR] Kalibrierung fehlgeschlagen")
