import numpy as np
import matplotlib.pyplot as plt

# Parameter für die Gauss-Verteilungen
# Linke Verteilung
mu_left = -3.0        # Position (Mittelwert)
width_left = 3.0      # Breite (bestimmt wie breit die Kurve ist)
height_left = 1.0     # Höhe (maximale Amplitude)

# Mittlere Verteilung (bei 0)
mu_center = 0.0       # Position (Mittelwert)
width_center = 1.5    # Breite
height_center = 0.1   # Höhe

# Rechte Verteilung (symmetrisch zur linken)
mu_right = 3.0        # Position (Mittelwert)
width_right = 3.0     # Breite
height_right = 1.0    # Höhe

# X-Achse definieren
x = np.linspace(-6, 6, 1000)

# Gauss-Funktion mit unabhängiger Breite und Höhe
def gaussian(x, mu, width, height):
    # sigma wird aus der gewünschten Breite berechnet
    # width entspricht etwa 4*sigma (±2σ deckt ~95% ab)
    sigma = width / 4.0
    # Normalisierte Gauss-Funktion multipliziert mit gewünschter Höhe
    return height * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Gauss-Verteilungen berechnen
y_left = gaussian(x, mu_left, width_left, height_left)
y_center = gaussian(x, mu_center, width_center, height_center)
y_right = gaussian(x, mu_right, width_right, height_right)

# Plot erstellen
fig, ax = plt.subplots(figsize=(5, 3))

# Verteilungen plotten
ax.plot(x, y_left, linewidth=2, label=f'Links (Breite={width_left}, Höhe={height_left})', color='blue')
#ax.plot(x, y_center, linewidth=2, label=f'Mitte (Breite={width_center}, Höhe={height_center})', color='green')
ax.plot(x, y_right, linewidth=2, label=f'Rechts (Breite={width_right}, Höhe={height_right})', color='blue')

# Achsen-Ticks entfernen
ax.set_xticks([])
ax.set_yticks([])

# Grid ausschalten
ax.grid(False)

# Legende hinzufügen
#ax.legend()

# Achsenbeschriftungen (optional)
ax.set_xlabel('x')
ax.set_ylabel('z')

plt.tight_layout()
plt.savefig('Exp-Crater.svg')
plt.show()
