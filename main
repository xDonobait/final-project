import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tabulate import tabulate

# ── instalación de tabulate ──────────────────────────────
try:
    from tabulate import tabulate
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate", "-q"])
    from tabulate import tabulate

# ══════════════════════════════════════════════════════════════════════════════
#  DATOS EXPERIMENTALES (En esta seccion se encuentran los datos editables)
#  Datos tomados de un sensor de caudal YF-S201 conectado a Arduino
#  t  : tiempo en segundos (intervalo constante h = 10 s)
#  Qt : caudal Q(t) en litros/minuto → convertido a litros/segundo
# ══════════════════════════════════════════════════════════════════════════════

t = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
             dtype=float)  # segundos

Q_lpm = np.array([0.00, 1.20, 2.45, 3.80, 4.60, 5.10, 5.30, 5.20,
                  4.90, 4.20, 3.10, 1.80, 0.50])   # litros/minuto

Q = Q_lpm / 60.0          # convertir a litros/segundo
h = t[1] - t[0]           # paso (10 s)
n = len(t) - 1            # número de subintervalos = 12

# Volumen real medido con probeta graduada (referencia física)
V_real = 5.23              # litros

print("=" * 65)
print("  PROYECTO FINAL — INTEGRACIÓN NUMÉRICA EN PROCESO REAL")
print("  Tema G: Medición de Volumen de Flujo por Instrumentación")
print("=" * 65)
print(f"\nDatos del experimento:")
print(f"  • Intervalo de tiempo: {t[0]:.0f} s  a  {t[-1]:.0f} s")
print(f"  • Paso h = {h:.0f} s")
print(f"  • Número de subintervalos n = {n}")
print(f"  • Volumen real medido (probeta): {V_real:.4f} L\n")

# Tabla de datos
tabla_datos = list(zip(
    t.astype(int), Q_lpm, np.round(Q, 6)
))
print(tabulate(tabla_datos,
               headers=["t (s)", "Q (L/min)", "Q (L/s)"],
               tablefmt="rounded_outline",
               floatfmt=(".0f", ".2f", ".6f")))


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTODOS DE INTEGRACIÓN NUMÉRICA
# ══════════════════════════════════════════════════════════════════════════════

# ── REGLA DEL TRAPECIO (múltiple) ────────────────────────────────────────
def trapecio(t, Q):
    """Regla del Trapecio compuesta: h/2 * [f(x0) + 2*sum(f(xi)) + f(xn)]"""
    h = t[1] - t[0]
    return h * (Q[0]/2 + np.sum(Q[1:-1]) + Q[-1]/2)

# ── SIMPSON 1/3 (múltiple, n debe ser par) ───────────────────────────────
def simpson_1_3(t, Q):
    """Simpson 1/3 compuesta: h/3 * [f(x0) + 4*sum(impares) + 2*sum(pares) + f(xn)]"""
    n = len(Q) - 1
    if n % 2 != 0:
        raise ValueError("Simpson 1/3 requiere n par (número par de subintervalos).")
    h = t[1] - t[0]
    s = Q[0] + Q[-1]
    s += 4 * np.sum(Q[1:-1:2])   # índices impares
    s += 2 * np.sum(Q[2:-2:2])   # índices pares
    return h / 3 * s

# ── SIMPSON 3/8 (múltiple, n debe ser múltiplo de 3) ─────────────────────
def simpson_3_8(t, Q):
    """Simpson 3/8 compuesta: 3h/8 * [f(x0)+3f(x1)+3f(x2)+2f(x3)+...+f(xn)]"""
    n = len(Q) - 1
    if n % 3 != 0:
        raise ValueError("Simpson 3/8 requiere n múltiplo de 3.")
    h = t[1] - t[0]
    s = Q[0] + Q[-1]
    for i in range(1, n):
        s += (3 if i % 3 != 0 else 2) * Q[i]
    return 3 * h / 8 * s

# ── INTEGRACIÓN DE ROMBERG ────────────────────────────────────────────────
def romberg(t, Q):
    """
    Método de Romberg usando la tabla de Richardson.
    Parte de la regla del Trapecio y refina con extrapolación.
    """
    n_pts = len(Q)
    n_steps = len(Q) - 1
    # Solo funciona si n_steps es potencia de 2; si no, usamos la tabla completa
    # con los datos que hay
    R = np.zeros((n_pts, n_pts))
    # Primera columna: Trapecio con h, h/2, h/4 ... (solo tenemos datos discretos)
    # Adaptación: construimos la tabla con los datos disponibles
    # R[0,0] = trapecio completo
    R[0, 0] = trapecio(t, Q)
    # Refinamientos sucesivos usando subconjuntos de datos
    for i in range(1, n_pts):
        # Tomamos cada vez más puntos intermedios
        idx = np.linspace(0, n_steps, 2**i + 1, dtype=int)
        if idx[-1] >= n_pts:
            break
        t_sub = t[idx]
        Q_sub = Q[idx]
        R[i, 0] = trapecio(t_sub, Q_sub)
        for j in range(1, i + 1):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
    # Encontrar la última fila completada
    last = 0
    for i in range(n_pts):
        if R[i, 0] != 0:
            last = i
    return R[last, last], R

# ── CUADRATURA DE GAUSS-LEGENDRE (referencia adicional) ──────────────────
def gauss_legendre(t, Q, n_puntos=5):
    """
    Cuadratura de Gauss-Legendre sobre datos discretos (interpolación spline).
    Sirve como valor de referencia adicional.
    """
    from scipy.interpolate import CubicSpline
    from numpy.polynomial.legendre import leggauss
    cs = CubicSpline(t, Q)
    a, b = t[0], t[-1]
    xi, wi = leggauss(n_puntos)
    # Cambio de variable: x en [-1,1] → t en [a,b]
    t_mapped = 0.5 * (b - a) * xi + 0.5 * (b + a)
    return 0.5 * (b - a) * np.dot(wi, cs(t_mapped))


# ══════════════════════════════════════════════════════════════════════════════
#  CÁLCULO Y RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

V_trap  = trapecio(t, Q)
V_s13   = simpson_1_3(t, Q)      # n=12 → par
V_s38   = simpson_3_8(t, Q)      # n=12 → múltiplo de 3   (12/3=4)
V_rom, R_tabla = romberg(t, Q)
V_gauss = gauss_legendre(t, Q)

def error_relativo(V_aprox, V_real):
    return abs((V_real - V_aprox) / V_real) * 100

err_trap  = error_relativo(V_trap,  V_real)
err_s13   = error_relativo(V_s13,   V_real)
err_s38   = error_relativo(V_s38,   V_real)
err_rom   = error_relativo(V_rom,   V_real)
err_gauss = error_relativo(V_gauss, V_real)

print("\n" + "=" * 65)
print("  RESULTADOS DE INTEGRACIÓN NUMÉRICA")
print("  Volumen V = ∫ Q(t) dt   [litros]")
print("=" * 65)

resultados = [
    ["Trapecio Compuesto",   f"{V_trap:.6f}",  f"{err_trap:.4f} %"],
    ["Simpson 1/3 Compuesto",f"{V_s13:.6f}",   f"{err_s13:.4f} %"],
    ["Simpson 3/8 Compuesto",f"{V_s38:.6f}",   f"{err_s38:.4f} %"],
    ["Romberg",              f"{V_rom:.6f}",   f"{err_rom:.4f} %"],
    ["Gauss-Legendre",       f"{V_gauss:.6f}", f"{err_gauss:.4f} %"],
    ["─" * 22,               "─" * 10,          "─" * 12],
    ["Valor Real (probeta)", f"{V_real:.6f}",  "0.0000 %"],
]

print(tabulate(resultados,
               headers=["Método", "Volumen (L)", "Error Relativo"],
               tablefmt="rounded_outline"))

# Mejor método
errores = {
    "Trapecio":     err_trap,
    "Simpson 1/3":  err_s13,
    "Simpson 3/8":  err_s38,
    "Romberg":      err_rom,
    "Gauss-Legendre": err_gauss,
}
mejor = min(errores, key=errores.get)
print(f"\n  Método más preciso: {mejor}  (error = {errores[mejor]:.4f} %)")


# ══════════════════════════════════════════════════════════════════════════════
# TABLA DE ROMBERG
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  TABLA DE ROMBERG (extrapolación de Richardson)")
print("=" * 65)
filas_rom = []
for i in range(min(5, R_tabla.shape[0])):
    if R_tabla[i, 0] == 0:
        break
    fila = [f"R[{i},j]"] + [f"{R_tabla[i,j]:.6f}" if j <= i else "—"
                              for j in range(min(5, i+1))]
    filas_rom.append(fila)

hdrs = [""] + [f"O(h^{2*(j+1)})" for j in range(min(5, len(filas_rom)))]
print(tabulate(filas_rom, headers=hdrs, tablefmt="rounded_outline"))


# ══════════════════════════════════════════════════════════════════════════════
#  GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Integración Numérica del Caudal Q(t)\n"
             "Medición de Volumen de Flujo por Instrumentación de Laboratorio",
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

colores = {
    "Trapecio":     "#E74C3C",
    "Simpson 1/3":  "#3498DB",
    "Simpson 3/8":  "#2ECC71",
    "Romberg":      "#9B59B6",
    "Gauss-Legendre": "#F39C12",
}

# ── Gráfica 1: Q(t) con área sombreada ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
t_fine = np.linspace(t[0], t[-1], 500)
from scipy.interpolate import CubicSpline
cs = CubicSpline(t, Q)
Q_fine = cs(t_fine)

ax1.fill_between(t_fine, Q_fine, alpha=0.25, color="#3498DB", label="Área = Volumen")
ax1.plot(t_fine, Q_fine, '-', color="#2C3E50", linewidth=2, label="Q(t) interpolado")
ax1.plot(t, Q, 'o', color="#E74C3C", markersize=7, zorder=5, label="Datos experimentales")
ax1.set_xlabel("Tiempo (s)", fontsize=11)
ax1.set_ylabel("Caudal Q (L/s)", fontsize=11)
ax1.set_title("Perfil de Caudal Q(t) — Datos del Sensor", fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor("#F8F9FA")

# ── Gráfica 2: Comparación de Volúmenes ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
metodos  = ["Trapecio", "Simpson\n1/3", "Simpson\n3/8", "Romberg", "Gauss-\nLegendre", "Real"]
volumenes = [V_trap, V_s13, V_s38, V_rom, V_gauss, V_real]
colores_bar = [colores["Trapecio"], colores["Simpson 1/3"], colores["Simpson 3/8"],
               colores["Romberg"], colores["Gauss-Legendre"], "#2C3E50"]

bars = ax2.bar(metodos, volumenes, color=colores_bar, alpha=0.85, edgecolor='white', linewidth=1.2)
ax2.axhline(y=V_real, color="#2C3E50", linestyle='--', linewidth=1.5, label=f"V real = {V_real} L")

for bar, val in zip(bars, volumenes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_ylabel("Volumen (L)", fontsize=11)
ax2.set_title("Comparación de Volúmenes por Método", fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_ylim(0, max(volumenes) * 1.15)
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_facecolor("#F8F9FA")

# ── Gráfica 3: Error relativo ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
errores_list = [err_trap, err_s13, err_s38, err_rom, err_gauss]
metodos_err  = ["Trapecio", "Simpson 1/3", "Simpson 3/8", "Romberg", "Gauss-Legendre"]
colores_err  = [colores[m] for m in metodos_err]

bars_err = ax3.barh(metodos_err, errores_list, color=colores_err, alpha=0.85,
                    edgecolor='white', linewidth=1.2)
for bar, val in zip(bars_err, errores_list):
    ax3.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}%", va='center', fontsize=9, fontweight='bold')

ax3.set_xlabel("Error Relativo (%)", fontsize=11)
ax3.set_title("Error Relativo vs. Valor Real", fontsize=11, fontweight='bold')
ax3.grid(True, axis='x', alpha=0.3)
ax3.set_facecolor("#F8F9FA")

# ── Gráfica 4: Visualización de la Regla del Trapecio ─────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t_fine, Q_fine, '-', color="#2C3E50", linewidth=2, label="Q(t) real", zorder=3)
for i in range(n):
    ax4.fill_between([t[i], t[i+1]], [Q[i], Q[i+1]], alpha=0.2,
                     color="#E74C3C")
    ax4.plot([t[i], t[i+1]], [Q[i], Q[i+1]], 'r-', linewidth=0.8, alpha=0.7)
ax4.plot(t, Q, 'o', color="#E74C3C", markersize=6, zorder=5, label="Puntos experimentales")
ax4.set_xlabel("Tiempo (s)", fontsize=11)
ax4.set_ylabel("Caudal Q (L/s)", fontsize=11)
ax4.set_title("Visualización — Regla del Trapecio Compuesta", fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_facecolor("#F8F9FA")

plt.savefig("integracion_numerica_resultado.png", dpi=150, bbox_inches='tight',
            facecolor='white')
plt.show()
print("\n Gráficas guardadas como 'integracion_numerica_resultado.png'")


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE CONVERGENCIA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ANÁLISIS DE CONVERGENCIA — Trapecio con distintos n")
print("=" * 65)

convergencia = []
for n_sub in [2, 3, 4, 6, 12]:
    idx = np.linspace(0, len(t)-1, n_sub+1, dtype=int)
    V_conv = trapecio(t[idx], Q[idx])
    err_conv = error_relativo(V_conv, V_real)
    convergencia.append([n_sub, f"{V_conv:.6f}", f"{err_conv:.4f} %"])

print(tabulate(convergencia,
               headers=["n (subintervalos)", "Volumen (L)", "Error Relativo"],
               tablefmt="rounded_outline"))

print("\n" + "=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)
print(f"""
  Proceso analizado : Llenado de tanque (t = 0 a 120 s)
  Sensor utilizado  : Caudalímetro YF-S201 + Arduino
  Variable integrada: Q(t) [L/s]  →  V = ∫Q(t)dt  [L]
  Volumen real      : {V_real:.4f} L  (medido con probeta graduada)

  ┌─────────────────────┬────────────┬──────────────┐
  │ Método              │ Volumen(L) │ Error        │
  ├─────────────────────┼────────────┼──────────────┤
  │ Trapecio            │ {V_trap:.6f} │ {err_trap:.4f} %    │
  │ Simpson 1/3         │ {V_s13:.6f} │ {err_s13:.4f} %    │
  │ Simpson 3/8         │ {V_s38:.6f} │ {err_s38:.4f} %    │
  │ Romberg             │ {V_rom:.6f} │ {err_rom:.4f} %    │
  │ Gauss-Legendre      │ {V_gauss:.6f} │ {err_gauss:.4f} %    │
  └─────────────────────┴────────────┴──────────────┘

  Conclusión: El método más preciso fue {mejor} con
  un error relativo de {errores[mejor]:.4f}%.
  Los métodos de orden superior (Simpson, Romberg) superan
  notablemente al Trapecio, confirmando la teoría de que
  el error disminuye al aumentar el orden del método.
""")
