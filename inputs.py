import numpy as np

# ========== PARAMÈTRES GÉOMÉTRIQUES ==========
R1 = 0.038
R2 = 0.06
R3 = 0.061
R4 = 0.085
Rg = (R2 + R3) / 2
R_EXT = 0.1
G = 0.001
E = 0.001

# ========== PARAMÈTRES ÉLECTRIQUES ==========
slip = 0.0001
ss = 1 + 5 * (1 - slip)
FREQ = 50.0
OMEGA_s = 2 * np.pi * FREQ
wrh = slip * OMEGA_s
fg = ss * FREQ
MU_0 = 1.2566370614359173e-06
MU_R = 1.0
SIGMA = 35000000.0
SIGMA_DEFECT = 3500000.0
SIGMA_BROKEN = 1e-08
P = 2

# ========== CONFIGURATION DES BARRES ROTORIQUES ==========
Q_R = 28
bar_status = np.full(Q_R, SIGMA)
bar_status[[0]] = SIGMA_BROKEN
bar_status[[5, 12]] = SIGMA_DEFECT
def get_sigma(j: int) -> float:
    assert 0 <= j < Q_R, f"Indice de barre invalide : {j}"
    return bar_status[j]

# ========== PARAMÈTRES DE SIMULATION ==========
g = np.linspace(0.1, 1, 50)
Ph = 3
Q_S = 36
N_HARM = 100
M_HARM = 3
L_HARM = 3

# ========== AUTRES PARAMÈTRES ==========
N_C = 15
L_U = 0.2
V_m = 220.0
V1 = 220.0
I_M = 20.0

# ========== PARAMÈTRES ANGULAIRES DÉDUITS ==========
B = 0.1121997376282069
C = 0.08726646259971647
