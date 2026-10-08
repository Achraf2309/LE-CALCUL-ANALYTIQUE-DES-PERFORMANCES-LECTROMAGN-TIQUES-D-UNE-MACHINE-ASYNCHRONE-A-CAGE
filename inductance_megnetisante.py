
import numpy as np
import pandas as pd
from scipy.special import jv, yv, jvp, yvp
import importlib.util
import sys
import os
import matplotlib.pyplot as plt


def calculate_Lm():
        # === Dynamically load inputs.py ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inputs_path = os.path.join(script_dir, "inputs.py")

    spec = importlib.util.spec_from_file_location("inputs", inputs_path)
    inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inputs)

    # ========== PARAMÈTRES GÉOMÉTRIQUES ==========
    R1 = inputs.R1
    R2 = inputs.R2
    R3 = inputs.R3
    R4 = inputs.R4
    Rg = inputs.Rg
    R_EXT = inputs.R_EXT
    G = inputs.G
    E = inputs.E

    # ========== PARAMÈTRES ÉLECTRIQUES ==========
    slip = 0.0001

    # Recompute dependent values manually
    FREQ = inputs.FREQ
    OMEGA_s = 2 * np.pi * FREQ
    ss = 1 + 5 * (1 - slip)
    wrh = slip * OMEGA_s
    fg = ss * FREQ

    # Other parameters as-is
    MU_0 = inputs.MU_0
    MU_R = inputs.MU_R
    SIGMA = inputs.SIGMA
    SIGMA_DEFECT = inputs.SIGMA_DEFECT
    SIGMA_BROKEN = inputs.SIGMA_BROKEN
    P = inputs.P

    # ========== CONFIGURATION DES BARRES ROTORIQUES ==========
    Q_R = inputs.Q_R
    bar_status = inputs.bar_status
    get_sigma = inputs.get_sigma

    # ========== PARAMÈTRES DE SIMULATION ==========

    Ph = inputs.Ph
    Q_S = inputs.Q_S
    N_HARM = inputs.N_HARM
    M_HARM = inputs.M_HARM
    L_HARM = inputs.L_HARM

    # ========== AUTRES PARAMÈTRES ==========
    N_C = inputs.N_C
    L_U = inputs.L_U
    V_m = inputs.V_m
    V1 = inputs.V1
    I_M = inputs.I_M

    # ========== PARAMÈTRES ANGULAIRES DÉDUITS ==========
    B = inputs.B
    C = inputs.C

    ###################################################################################################################################################
    # MATRICE DE COUPLAGE
    ###################################################################################################################################################

    pattern = [
        [1, 0, 0],#1
        [1, 0, 0],#2
        [1, 0, 0],#3
        [0, 0,-1],#4
        [0, 0,-1],#5
        [0, 0,-1],#6
        [0, 1, 0],#7
        [0, 1, 0],#8
        [0, 1, 0],#9
        [-1,0, 0],#10
        [-1,0, 0],#11
        [-1,0, 0],#12
        [0, 0, 1],#13
        [0, 0, 1],#14
        [0, 0, 1],#15
        [0,-1, 0],#16
        [0,-1, 0],#17
        [0,-1, 0],#18
        [1, 0, 0],#19
        [1, 0, 0],#20
        [1, 0, 0],#21
        [0, 0,-1],#22
        [0, 0,-1],#23
        [0, 0,-1],#24
        [0, 1, 0],#25
        [0, 1, 0],#26
        [0, 1, 0],#27
        [-1,0, 0],#28
        [-1,0, 0],#29
        [-1,0, 0],#30
        [0, 0, 1],#31
        [0, 0, 1],#32
        [0, 0, 1],#33
        [0,-1, 0],#34
        [0,-1, 0],#35
        [0,-1, 0],#36
    ]
    C_MATRIX= np.array(pattern)

    ###################################################################################################################################################
    # MATRICE DE COUPLAGE
    ###################################################################################################################################################
    # Définition du courant de phase
    I_PHASE = I_M * np.array([
        [np.cos(0) + 1j * np.sin(0)],
        [np.cos(-2 * np.pi / 3) + 1j * np.sin(-2 * np.pi / 3)],
        [np.cos(2 * np.pi / 3) + 1j * np.sin(2 * np.pi / 3)]
    ])
    # Surface des encoches du stator
    S_ENCOCHE = C * (R4**2 - R3**2) / 2  
    # Densité de courant du stator dans les encoches statoriques
    J_I = np.conjugate((N_C / S_ENCOCHE) * (C_MATRIX @ I_PHASE))

    ###################################################################################################################################################
    # Beta and g vectors calculations
    ###################################################################################################################################################

    def calculate_g_vector(Qr):
        Pdr = 360 / Qr  # Pole pitch in degrees
        rad = np.pi / 180  # Conversion factor to radians
        g = np.array([(Pdr * i - Pdr / 2) * rad for i in range(1, Qr + 1)])  # Maple-style sequence
        return g
    g_vector = calculate_g_vector(Q_R)

    def calculate_beta(Qs):
        Pds = 360 / Qs  # Slot pitch in degrees
        rad = np.pi / 180  # Conversion factor to radians
        beta = np.array([(Pds * i - Pds / 2) * rad for i in range(1, Qs + 1)])  # Maple-style sequence
        return beta
    beta_vector = calculate_beta(Q_S)
    ###################################################################################################################################################
    # fonctions 
    ###################################################################################################################################################
    # 📌 Fonction gamma(glissement) dépendante de glissment CORRECT
    def gamma(glissement):
        return np.sqrt(-1j *glissement* OMEGA_s * SIGMA * MU_0)


    # 📌 Fonction f(r) dépendante de gamma CORRECT
    def f(r, glissement, R1):
        gamma_val = gamma(glissement)
        jv0_r = jv(0, gamma_val * r)
        jv1_R1 = jv(1, gamma_val * R1)
        yv1_R1 = yv(1, gamma_val * R1)

        # Avoid division by zero or near-zero values
        if np.isclose(yv1_R1, 0):
            raise ValueError(f"yv(1, gamma * R1) is near zero: {yv1_R1}")

        return jv0_r - (jv1_R1 * yv(0, gamma_val * r)) / yv1_R1


    # 📌 Fonction d(r) dépendante de gamma CORRECT

    def d(r, glissement, R1):
        alpha = gamma(glissement)
        return jv(1, alpha * r) * alpha - (jv(1, alpha * R1) * yv(1, alpha * r) * alpha) / yv(1, alpha * R1)

    # 📌 Fonction INT1 dépendant de N_HARM et M_HARM CORRECT
    def compute_INT1(n, m, j, g_vector, B):
        g = g_vector[j]
        denom = (B * n)**2 - (np.pi * m)**2

        if np.isclose(denom, 0):
            term1 = 0.5 * B * np.cos(-n * g + 0.5 * n * B)
            term2 = 0.25 * np.sin(-n * g + 0.5 * n * B) / n
            term3 = 0.25 * np.sin(n * g + 1.5 * n * B) / n
            return (term1 + term2 + term3)
        else:
            term1 = n * B**2  * (np.sin(0.5 * n * (-2 * g + B)) + (-1)**m * np.sin(0.5 * n * (2 * g + B)))
            return term1 / denom


    # 📌 Fonction INT1 dépendant de N_HARM et M_HARM CORRECT
    def compute_INT2(n, m, j, g_vector, B):
        g = g_vector[j]
        denom = -(B * n)**2 + (np.pi * m)**2

        if np.isclose(denom, 0):
            term1 = 0.25 * np.cos(-n * g + 0.5 * n * B) / n  
            term2 = - 0.25 * np.cos(n * g + 1.5 * n * B) / n
            term3 = - 0.5 * B * np.sin(-n * g + 0.5 * n * B)
            return (term1 + term2 + term3)
        else:
            term1 = n * B**2  * (-np.cos(0.5 * n * (-2 * g + B)) + (-1)**m * np.cos(0.5 * n * (2 * g + B)))
            return term1  / denom


    # 📌 Fonction INT3 dépendant de N_HARM et L_HARM CORRECT
    def compute_INT3(n, l, ib_vect , beta_vector, C):
        beta = beta_vector[ib_vect]
        denom = (C * n)**2 - (np.pi * l)**2

        if np.isclose(denom, 0):
            term1 = 0.5 * C * np.cos(-n * beta + 0.5 * n * C)
            term2 = 0.25 * np.sin(-n * beta + 0.5 * n * C) / n
            term3 = 0.25 * np.sin(n * beta + 1.5 * n * C) / n
            return (term1 + term2 + term3)
        else:
            term1 = n * C**2  * (np.sin(0.5 * n * (-2 * beta + C)) + (-1)**l * np.sin(0.5 * n * (2 * beta + C)))
            return term1 / denom


    # 📌 Fonction INT dépendant de N_HARM et L_HARM CORRECT
    def compute_INT4( n, l, ib_vect, beta_vector, C):
        beta = beta_vector[ib_vect]
        denom = -(C * n)**2 + (np.pi * l)**2

        if np.isclose(denom, 0):
            term1 = -0.5 * C * np.sin(-n * beta + 0.5 * n * C)
            term2 = -0.25 * np.cos(n * beta + 1.5 * n * C) / n
            term3 = 0.25 * np.cos(-n * beta + 0.5 * n * C) / n
            return (term1 + term2 + term3)
        else:
            term1 = n * C**2  * (-np.cos(0.5 * n * (-2 * beta + C)) + (-1)**l * np.cos(0.5 * n * (2 * beta + C)))
            return term1 / denom


    # 📌 Fonction h_l(r)  CORRECT ✅✅✅✅✅
    def h_l(r, l, C, R4):
        exponent = l * np.pi / C
        return r**(-exponent) + R4**(-2 * exponent) * r**(exponent)

    # 📌 Fonction h_l_prime(r)
    def h_l_prime(r, l, C, R4):
        alpha = l * np.pi / C
        term1 = -alpha * r**(-alpha - 1)
        term2 = alpha * R4**(-2 * alpha) * r**(alpha - 1)
        return term1 + term2

    # 📌 Fonction h_l_int(r)
    def integrate_hl_times_r(l, C, R3, R4):
        exponent = l * np.pi / C
        result = R4**(-exponent+2)/(-exponent+2) + (R4**(-2 * exponent))/(exponent+2) * R4**(exponent+2) - (R3**(-exponent+2))/(-exponent+2) + (R4**(-2 * exponent))/(exponent+2) * R3**(exponent+2)
        return result


    # 📌 Fonction g_m(r) CORRECT
    def g_m (r, m, B, glissement, R1):
        gamma_val = gamma(glissement)
        J_num=    (-jv((m * np.pi + B ) / B , gamma_val * R1 ) * B * gamma_val * R1 + m * np.pi * jv( m * np.pi / B ,gamma_val * R1 )) * yv( m * np.pi / B , gamma_val * r ) 
        J_denum= - yv((m * np.pi + B ) / B , gamma_val * R1 ) * B * gamma_val * R1 + m * np.pi * yv( m * np.pi / B ,gamma_val * R1 )
        coeff= J_num/J_denum    
        if np.isclose(J_denum, 0):
            raise ValueError(f"Dénominateur nul : yv(1, gamma * R1) is near zero.")
        coeff= J_num/J_denum
        return jv ( m * np.pi / B , gamma_val * r ) -  coeff


    # 📌 Fonction v_m(r) CORRECT
    def v_m (r, m, B, glissement, R1):
        gamma_val = gamma(glissement)
        J_num=   (- jv((m * np.pi + B ) / B , gamma_val * R1 ) * B * gamma_val * R1 + m * np.pi * jv( m * np.pi / B ,gamma_val * R1 )) * yv( m * np.pi / B , gamma_val * r ) 
        J_denum= - yv((m * np.pi + B ) / B , gamma_val * R1 ) * B * gamma_val * R1 + m * np.pi * yv( m * np.pi / B ,gamma_val * R1 )
        coeff= J_num/J_denum    
        if np.isclose(J_denum, 0):
            raise ValueError(f"Dénominateur nul : yv(1, gamma * R1) is near zero.")
        coeff= J_num/J_denum
        return jv ( m * np.pi / B , gamma_val * r ) -  coeff

    # 📌 Fonction g_prime_m(r) CORRECT
    def g_prime_m(r , m, B, glissement, R1):
        alpha = gamma(glissement)
        order = m * np.pi / B
        nu = (m * np.pi + B) / B

        # First term
        term1 = (-jv(nu, alpha * r) + (m * np.pi * jv(order, alpha * r)) / (B * alpha * r)) * alpha

        # Second term numerator
        num2 = (-jv(nu, alpha * R1) * B * alpha * R1 + m * np.pi * jv(order, alpha * R1))

        # Coefficient numerator
        coeff1 = (-yv(nu, alpha * r) + (m * np.pi * yv(order, alpha * r)) / (B * alpha * r)) * alpha

        # Coefficient denominator
        coeff2 = (-yv(nu, alpha * R1) * B * alpha * R1 + m * np.pi * yv(order, alpha * R1))

        if np.isclose(coeff2, 0):
            raise ValueError("Denominator is zero or nearly zero.")

        return (term1 - num2 * coeff1 / coeff2)

    def v_prime_m(r , m, B, glissement, R1):
        alpha = gamma(glissement)
        order = m * np.pi / B
        nu = (m * np.pi + B) / B

        # First term
        term1 = (-jv(nu, alpha * r) + (m * np.pi * jv(order, alpha * r)) / (B * alpha * r)) * alpha

        # Second term numerator
        num2 = (-jv(nu, alpha * R1) * B * alpha * R1 + m * np.pi * jv(order, alpha * R1))

        # Coefficient numerator
        coeff1 = (-yv(nu, alpha * r) + (m * np.pi * yv(order, alpha * r)) / (B * alpha * r)) * alpha

        # Coefficient denominator
        coeff2 = (-yv(nu, alpha * R1) * B * alpha * R1 + m * np.pi * yv(order, alpha * R1))

        if np.isclose(coeff2, 0):
            raise ValueError("Denominator is zero or nearly zero.")

        return (-term1 + num2 * coeff1 / coeff2)

    
    def j_m(r, glissement, R1,B):
        alpha = gamma(glissement)
        term1= (-jv( 1 , alpha * r ) * yv( 1 , alpha * R1 ) + jv( 1 , alpha * R1) * yv( 1 , alpha * r )) * B
        term2 = yv( 1 , alpha * R1 ) * alpha
        if np.isclose(term2, 0):
            raise ValueError(f"yv(1, gamma * R1) is near zero: {term2}")

        return term1 / term2

    ###################################################################################################################################################
    # b vecteur
    ###################################################################################################################################################
    b1 = np.zeros((Q_R, 1))  # Q_r x 1
    b2 = np.zeros((Q_R * M_HARM, 1))  # (Q_r x M) x 1
    b3 = np.zeros((Q_S, 1), dtype=complex)  # Q_s x 1
    b4 = np.zeros((Q_S * L_HARM, 1))  # (Q_s x L) x 1
    b5 = np.zeros((1, 1))  # 1 x 1
    b6 = np.zeros((N_HARM, 1))  # N x 1
    b7 = np.zeros((N_HARM, 1))  # N x 1
    b8 = np.zeros((N_HARM, 1), dtype=complex)  # 1 x 1
    b9 = np.zeros((N_HARM, 1), dtype=complex)  # N x 1


    for i in range(Q_S):  
        b3[i, 0] = (1/2) * MU_0 * J_I[i, 0] * (R4**2) * np.log(R3) - (1/4) * MU_0 * J_I[i, 0] * (R3**2)


    for n in range(1, N_HARM + 1):
        for i in range(Q_S):
            b8[n - 1] -= (2  *(-0.5 * MU_0 * J_I[i, 0] * (R4**2 / R3) + 0.5 * MU_0 * J_I[i, 0] * R3) * np.sin(0.5 * n * C) * np.sin(n * beta_vector[i]))/ (n * np.pi * MU_0)
            b9[n - 1] -= (2  *(-0.5 * MU_0 * J_I[i, 0] * (R4**2 / R3) + 0.5 * MU_0 * J_I[i, 0] * R3) * np.sin(0.5 * n * C) * np.cos(n * beta_vector[i]))/ (n * np.pi * MU_0)


    b1 = b1.astype(complex)
    b2 = b2.astype(complex)
    b3 = b3.astype(complex)
    b4 = b4.astype(complex)
    b5 = b5.astype(complex)
    b6 = b6.astype(complex)
    b7 = b7.astype(complex)
    b8 = b8.astype(complex)
    b9 = b9.astype(complex)

    # 🟢 Regroupement de tous les vecteurs b dans un seul vecteur

    b = np.concatenate([b1, b2, b3, b4, b5, b6, b7, b8, b9], axis=0)
    ###################################################################################################################################################
    # equation1
    ###################################################################################################################################################
    alpha_1 = np.ones((Q_R, 1),dtype=complex)  # Taille (Q_r x 1)
    gamma_1 = np.zeros((Q_R, N_HARM),dtype=complex)  
    theta_1 = np.zeros((Q_R, N_HARM),dtype=complex)  
    lambda_1 = np.zeros((Q_R, N_HARM),dtype=complex)  
    Delta_1 = np.zeros((Q_R, N_HARM),dtype=complex)  
    delta_1 = np.zeros((Q_R, Q_R), dtype=complex)  
    E_1 = np.zeros((Q_R, Q_R * M_HARM),dtype=complex)  
    mu_1 = np.zeros((Q_R, Q_S),dtype=complex)  
    rho_1 = np.zeros((Q_R, Q_S * L_HARM),dtype=complex)  


    # Remplir les matrices gamma_1, theta_1, lambda_1, Delta_1 pour chaque harmonique
    for q in range(Q_R):
        for n in range(1, N_HARM + 1):
            gamma_1[q, n - 1] = (1 / (n * B)) * R2**n    * 2 * np.sin( 0.5 * n * B) * np.cos( n *  g_vector[q])
            theta_1[q, n - 1] = (1 / (n * B)) * R2**(-n) * 2 * np.sin( 0.5 * n * B) * np.cos( n *  g_vector[q])
            lambda_1[q, n - 1]= (1 / (n * B)) * R2**n    * 2 * np.sin( 0.5 * n * B) * np.sin( n *  g_vector[q])
            Delta_1[q, n - 1] = (1 / (n * B)) * R2**(-n) * 2 * np.sin( 0.5 * n * B) * np.sin( n *  g_vector[q])

    # Mise à jour de la matrice diagonale delta_1
    for q in range(Q_R):
        delta_1[q, q] = -f(R2,slip,R1) # Matrice diagonale utilisant la fonction f(R2)

    # Valeurs fixes
    E_1[:, :] = 0  
    mu_1[:, :] = 0  
    rho_1[:, :] = 0
    ###################################################################################################################################################
    # equation2
    ###################################################################################################################################################
    # 🟢 Initialisation des matrices
    alpha_2 = np.zeros((Q_R * M_HARM, 1),dtype=complex) 
    gamma_2 = np.zeros((Q_R * M_HARM, N_HARM),dtype=complex)  
    theta_2 = np.zeros((Q_R * M_HARM, N_HARM),dtype=complex)  
    lambda_2 = np.zeros((Q_R * M_HARM, N_HARM),dtype=complex)  
    Delta_2 = np.zeros((Q_R * M_HARM, N_HARM),dtype=complex)  
    delta_2 = np.zeros((Q_R * M_HARM, Q_R),dtype=complex)  
    E_2 = np.zeros((Q_R * M_HARM, Q_R * M_HARM), dtype=complex)  
    mu_2 = np.zeros((Q_R * M_HARM, Q_S),dtype=complex)  
    rho_2 = np.zeros((Q_R * M_HARM, Q_S * L_HARM),dtype=complex)  

    # 🟢 Remplir les matrices
            # Calcul des éléments de gamma_2 et theta_2
    for j in range(Q_R):  # Boucle sur les barres rotoriques
        for n in range(1, N_HARM + 1):  # Boucle sur les harmoniques dans l’entrefer
                for m in range(1, M_HARM + 1):  # Boucle sur les harmoniques des barres rotoriques
                    row_idx = j * M_HARM + (m - 1)  # Index pour la ligne dans les matrices
                    # Calcul de l'intégrale (INT1 et INT2)
                    INT1 = compute_INT1(n, m, j,g_vector, B)
                    INT2 = compute_INT2(n, m, j,g_vector, B)
                    gamma_2[row_idx, n - 1] = INT1 * R2**(n) * (2 / B) 
                    theta_2[row_idx, n - 1] = INT1 * R2**(-n)* (2 / B) 
                    lambda_2[row_idx, n - 1]= INT2 * R2**(n) * (2 / B)
                    Delta_2[row_idx, n - 1] = INT2 * R2**(-n)* (2 / B)
                    # Mise à jour de la matrice diagonale  
                    E_2[row_idx, row_idx] = - v_m(R2, m, B, slip, R1)  # gamma est constant ici

    ###################################################################################################################################################
    # equation3
    ###################################################################################################################################################
    # 📌 Initialisation des matrices avec les tailles correctes
    alpha_3 = np.ones((Q_S, 1),dtype=complex)  # Taille (Q_s x 1)
    gamma_3 = np.zeros((Q_S, N_HARM),dtype=complex)  # Taille (Q_s x N)
    theta_3 = np.zeros((Q_S, N_HARM),dtype=complex)  # Taille (Q_s x N)
    lambda_3 = np.zeros((Q_S, N_HARM),dtype=complex)  # Taille (Q_s x N)
    Delta_3 = np.zeros((Q_S, N_HARM),dtype=complex)  # Taille (Q_s x N)
    delta_3 = np.zeros((Q_S, Q_R),dtype=complex)  # Taille (Q_s x Q_r)
    E_3 = np.zeros((Q_S, M_HARM * Q_R),dtype=complex)  # Taille (Q_s x (M x Q_r))
    mu_3 = -np.eye(Q_S)  # Matrice diagonale (Q_s x Q_s)
    rho_3 = np.zeros((Q_S, Q_S * L_HARM),dtype=complex)  # Taille (Q_s x (L x Q_s))


    # 📌 Remplissage des matrices
    for i in range(Q_S):  # Boucle sur les encoches statoriques
        for n in range(1, N_HARM + 1):  # Boucle sur les harmoniques
            gamma_3[i, n - 1] = (1 / (n * C)) * R3**n    * 2 * np.sin( 0.5 * n * C) * np.cos( n *  beta_vector[i])  
            theta_3[i, n - 1] = (1 / (n * C)) * R3**(-n) * 2 * np.sin( 0.5 * n * C) * np.cos( n *  beta_vector[i])
            lambda_3[i, n - 1]= (1 / (n * C)) * R3**n    * 2 * np.sin( 0.5 * n * C) * np.sin( n *  beta_vector[i])
            Delta_3[i, n - 1] = (1 / (n * C)) * R3**(-n) * 2 * np.sin( 0.5 * n * C) * np.sin( n *  beta_vector[i])


    ###################################################################################################################################################
    # equation4
    ###################################################################################################################################################
    # 📌 Initialisation des matrices avec les tailles correctes
    alpha_4 = np.zeros((Q_S * L_HARM, 1),dtype=complex)  # Taille ((Q_s x L) x 1)
    gamma_4 = np.zeros((Q_S * L_HARM, N_HARM),dtype=complex)  # Taille ((Q_s x L) x N)
    theta_4 = np.zeros((Q_S * L_HARM, N_HARM),dtype=complex)  # Taille ((Q_s x L) x N)
    lambda_4 = np.zeros((Q_S * L_HARM, N_HARM),dtype=complex)  # Taille ((Q_s x L) x N)
    Delta_4 = np.zeros((Q_S * L_HARM, N_HARM),dtype=complex)  # Taille ((Q_s x L) x N)
    delta_4 = np.zeros((Q_S * L_HARM, Q_R),dtype=complex)  # Taille ((Q_s x L) x Q_r)
    E_4 = np.zeros((Q_S * L_HARM, Q_R * M_HARM),dtype=complex)  # Taille ((Q_s x L) x (Q_r x M))
    mu_4 = np.zeros((Q_S * L_HARM, Q_S),dtype=complex)  # Taille ((Q_s x L) x Q_s)
    rho_4 = np.zeros((Q_S * L_HARM, Q_S * L_HARM),dtype=complex)  # Taille ((Q_s x L) x (Q_s x L))


    # 📌 Remplissage des matrices
    for i in range(Q_S): 
        for n in range(1, N_HARM + 1):  
                for l in range(1, L_HARM + 1):  
                    row_idx = i * L_HARM + (l - 1)  
                    INT3 = compute_INT3(n, l, i ,beta_vector, C) 
                    INT4 = compute_INT4(n, l, i ,beta_vector, C)  
                    gamma_4 [row_idx, n - 1] =  INT3 * (2 / C) * R3**n
                    theta_4 [row_idx, n - 1] =  INT3 * (2 / C) * R3**(-n) 
                    lambda_4[row_idx, n - 1]=   INT4 * (2 / C) * R3**n
                    Delta_4 [row_idx, n - 1] =  INT4 * (2 / C) * R3**(-n)

    rho_4 = -np.diag([h_l(R3, l, C, R4) for i in range(Q_S) for l in range(1, L_HARM + 1)])
    ###################################################################################################################################################
    # equation5
    ###################################################################################################################################################
    # 🟢 Initialisation des matrices
    alpha_5 = np.zeros((1, 1),dtype=complex)  
    gamma_5 = np.zeros((1, N_HARM),dtype=complex)  
    theta_5 = np.zeros((1, N_HARM),dtype=complex)  
    lambda_5 = np.zeros((1, N_HARM),dtype=complex)  
    Delta_5 = np.zeros((1, N_HARM),dtype=complex)  
    delta_5 = np.zeros((1, Q_R), dtype=complex)  
    E_5 = np.zeros((1, Q_R * M_HARM),dtype=complex)  
    mu_5 = np.zeros((1, Q_S),dtype=complex)  
    rho_5 = np.zeros((1, Q_S * L_HARM),dtype=complex)  


    # 🟢 Remplissage des matrices
    for n in range(1, N_HARM + 1):  
        gamma_5[0, n - 1] = 0  
        theta_5[0, n - 1] = 0  
        lambda_5[0, n - 1] = 0  
        Delta_5[0, n - 1] = 0  


    for j in range(1, Q_R + 1):  
        delta_5[0, j - 1] = 1
    ###################################################################################################################################################
    # equation6
    ###################################################################################################################################################
    # 🟢 Initialisation des matrices
    alpha_6 = np.zeros((N_HARM, 1),dtype=complex)  
    gamma_6 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    theta_6 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    lambda_6 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    Delta_6 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    delta_6 = np.zeros((N_HARM, Q_R), dtype=complex)  
    E_6 = np.zeros((N_HARM, Q_R * M_HARM), dtype=complex)  
    mu_6 = np.zeros((N_HARM, Q_S),dtype=complex)  
    rho_6 = np.zeros((N_HARM, Q_S * L_HARM),dtype=complex)  

    # 📌 Supposons que gamma est une constante scalaire
    # (elle doit être définie quelque part dans ton code ou tes imports)
    # Ex: gamma = gamma(1, OMEGA_M, SIGMA, MU_0) ← si tu veux une valeur fixe

    # 🟢 Remplissage des matrices
    for n in range(1, N_HARM + 1):  
        gamma_6[n - 1, n - 1] =  n * R2**(n - 1) / (MU_0)
        theta_6[n - 1, n - 1] = -n * R2**(-n - 1) / (MU_0)   
        # lambda_6 et Delta_6 sont laissées à zéro
        for j in range(Q_R):  
                delta_6[n - 1, j] = d(R2,slip,R1) * 2 * np.sin(0.5 * n * B ) * np.cos(n * g_vector[j]) / (MU_0 * np.pi * n)


    # 🟢 Calcul de E_6 avec INT1


    for n in range(1, N_HARM + 1):
        for j in range(Q_R):
            for m in range(1, M_HARM + 1): 
                INT1 = compute_INT1(n, m,j, g_vector, B)
                col_index = (j) * M_HARM + (m - 1)   
                E_6[n - 1, col_index] =  v_prime_m(R2, m, B, slip, R1) * INT1 * (1 / (np.pi * MU_0))  

    ###################################################################################################################################################
    # equation7
    ###################################################################################################################################################

    # 🟢 Initialisation des matrices
    alpha_7 = np.zeros((N_HARM, 1),dtype=complex)  
    gamma_7 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    theta_7 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    lambda_7 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    Delta_7 = np.zeros((N_HARM, N_HARM ),dtype=complex)  
    delta_7 = np.zeros((N_HARM, Q_R), dtype=complex)  
    E_7 = np.zeros((N_HARM, Q_R * M_HARM), dtype=complex)  
    mu_7 = np.zeros((N_HARM, Q_S),dtype=complex)  
    rho_7 = np.zeros((N_HARM, Q_S * L_HARM),dtype=complex)  

    # 🟢 Remplissage des matrices diagonales
    for n in range(1, N_HARM + 1):  
        lambda_7[n - 1, n - 1] =  n * R2**(n - 1) *(1/MU_0) 
        Delta_7[n - 1, n - 1] = - n * R2**(-n - 1)*(1/MU_0) 
        for j in range(Q_R):  
            delta_7[n - 1, j] = d(R2,slip,R1)  * 2 * np.sin(0.5 * n * B ) * np.sin( n * g_vector[j]) / (MU_0 * np.pi * n)


    for n in range(1, N_HARM + 1):
        for j in range(Q_R):
            for m in range(1, M_HARM + 1): 
                INT2 = compute_INT2(n, m,j, g_vector, B)
                col_index = (j) * M_HARM + (m - 1)   
                E_7[n - 1, col_index] = v_prime_m(R2, m, B, slip, R1) * INT2 * (1 / (np.pi * MU_0))  

    ###################################################################################################################################################
    # equation8
    ###################################################################################################################################################

    # 🟢 Initialisation des matrices
    alpha_8 = np.zeros((N_HARM, 1),dtype=complex)  
    gamma_8 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    theta_8 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    lambda_8 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    Delta_8 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    delta_8 = np.zeros((N_HARM, Q_R),dtype=complex)  
    E_8 = np.zeros((N_HARM, Q_R * M_HARM),dtype=complex)  
    mu_8 = np.zeros((N_HARM, Q_S),dtype=complex)  
    rho_8 = np.zeros((N_HARM, Q_S * L_HARM),dtype=complex)  

    # 🟢 Remplissage des matrices diagonales
    for n in range(1, N_HARM + 1):  
        lambda_8[n - 1, n - 1] =  n * R3**(n - 1)   / MU_0 
        Delta_8 [n - 1, n - 1] = -n * R3**(-n - 1) / MU_0 

    for i in range(1, Q_S + 1):  
        for n in range(1, N_HARM + 1):     
            for l in range(1, L_HARM + 1):  
                INT4 = compute_INT4(n, l,i-1, beta_vector, C)
                rho_8[n - 1, (i - 1) * L_HARM + (l - 1)] = - h_l_prime(R3, l, C, R4) * INT4 * (1 /(MU_0 * np.pi) )

    ###################################################################################################################################################
    # equation9
    ###################################################################################################################################################

    # 🟢 Initialisation des matrices
    alpha_9 = np.zeros((N_HARM, 1),dtype=complex)  
    beta_9 = np.zeros((N_HARM, 1),dtype=complex)  
    gamma_9 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    theta_9 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    lambda_9 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    Delta_9 = np.zeros((N_HARM, N_HARM),dtype=complex)  
    delta_9 = np.zeros((N_HARM, Q_R),dtype=complex)  
    E_9 = np.zeros((N_HARM, Q_R * M_HARM),dtype=complex)  
    mu_9 = np.zeros((N_HARM, Q_S),dtype=complex)  
    rho_9 = np.zeros((N_HARM, Q_S * L_HARM),dtype=complex)  

    # 🟢 Remplissage des matrices diagonales
    for n in range(1, N_HARM + 1):  
        gamma_9[n - 1, n - 1] =  n  * R3**(n - 1)  / MU_0
        theta_9[n - 1, n - 1] = -n  * R3**(-n - 1) / MU_0 


    for i in range(1, Q_S + 1):  
        for n in range(1, N_HARM + 1):     
            for l in range(1, L_HARM + 1):  
                INT3 = compute_INT3(n, l,i-1, beta_vector, C)
                rho_9[n - 1, (i - 1) * L_HARM + (l - 1)] = - h_l_prime(R3, l, C, R4) * INT3 * (1 /(MU_0 * np.pi) )


    ###################################################################################################################################################
    # equations
    ###################################################################################################################################################



    alpha = np.concatenate([
        alpha_1.astype(complex), alpha_2.astype(complex), alpha_3.astype(complex),
        alpha_4.astype(complex), alpha_5.astype(complex), alpha_6.astype(complex),
        alpha_7.astype(complex), alpha_8.astype(complex), alpha_9.astype(complex)
    ], axis=0)

    gama = np.concatenate([
        gamma_1.astype(complex), gamma_2.astype(complex), gamma_3.astype(complex),
        gamma_4.astype(complex), gamma_5.astype(complex), gamma_6.astype(complex),
        gamma_7.astype(complex), gamma_8.astype(complex), gamma_9.astype(complex)
    ], axis=0)

    theta = np.concatenate([
        theta_1.astype(complex), theta_2.astype(complex), theta_3.astype(complex),
        theta_4.astype(complex), theta_5.astype(complex), theta_6.astype(complex),
        theta_7.astype(complex), theta_8.astype(complex), theta_9.astype(complex)
    ], axis=0)

    lambda_ = np.concatenate([
        lambda_1.astype(complex), lambda_2.astype(complex), lambda_3.astype(complex),
        lambda_4.astype(complex), lambda_5.astype(complex), lambda_6.astype(complex),
        lambda_7.astype(complex), lambda_8.astype(complex), lambda_9.astype(complex)
    ], axis=0)

    Delta = np.concatenate([
        Delta_1.astype(complex), Delta_2.astype(complex), Delta_3.astype(complex),
        Delta_4.astype(complex), Delta_5.astype(complex), Delta_6.astype(complex),
        Delta_7.astype(complex), Delta_8.astype(complex), Delta_9.astype(complex)
    ], axis=0)

    delta = np.concatenate([
        delta_1.astype(complex), delta_2.astype(complex), delta_3.astype(complex),
        delta_4.astype(complex), delta_5.astype(complex), delta_6.astype(complex),
        delta_7.astype(complex), delta_8.astype(complex), delta_9.astype(complex)
    ], axis=0)

    En = np.concatenate([
        E_1.astype(complex), E_2.astype(complex), E_3.astype(complex),
        E_4.astype(complex), E_5.astype(complex), E_6.astype(complex),
        E_7.astype(complex), E_8.astype(complex), E_9.astype(complex)
    ], axis=0)

    mu = np.concatenate([
        mu_1.astype(complex), mu_2.astype(complex), mu_3.astype(complex),
        mu_4.astype(complex), mu_5.astype(complex), mu_6.astype(complex),
        mu_7.astype(complex), mu_8.astype(complex), mu_9.astype(complex)
    ], axis=0)

    rho = np.concatenate([
        rho_1.astype(complex), rho_2.astype(complex), rho_3.astype(complex),
        rho_4.astype(complex), rho_5.astype(complex), rho_6.astype(complex),
        rho_7.astype(complex), rho_8.astype(complex), rho_9.astype(complex)
    ], axis=0)

    # alpha = np.concatenate([alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7, alpha_8, alpha_9], axis=0)
    # gama = np.concatenate([gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, gamma_6, gamma_7, gamma_8, gamma_9], axis=0)
    # theta = np.concatenate([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8, theta_8], axis=0)
    # lambda_ = np.concatenate([lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9], axis=0)
    # Delta = np.concatenate([Delta_1, Delta_2, Delta_3, Delta_4, Delta_5, Delta_6, Delta_7, Delta_8, Delta_9], axis=0)
    # delta = np.concatenate([delta_1, delta_2, delta_3, delta_4, delta_5, delta_6, delta_7, delta_8, delta_9], axis=0)
    # En = np.concatenate([E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, E_9], axis=0)
    # mu = np.concatenate([mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7, mu_8, mu_9], axis=0)
    # rho = np.concatenate([rho_1, rho_2, rho_3, rho_4, rho_5, rho_6, rho_7, rho_8, rho_9], axis=0)

    ###################################################################################################################################################
    # Resoudre le systeme
    ###################################################################################################################################################
    A_matrix = np.column_stack([
        alpha, # A10
        gama,     # A1
        theta,    # A2
        lambda_,  # A3
        Delta,    # A4
        delta,    # Bj0
        En,       # Bjm
        mu,       # Ci0
        rho       # Cil
    ])

    X = np.linalg.inv(A_matrix) @ b

    Num = 4 * N_HARM + 1 + (1 + M_HARM) * Q_R + (1 + L_HARM) * Q_S

    # Définition des tailles des blocs
    size_A1 = 1
    size_A2 = 4 * N_HARM
    size_B = (1 + M_HARM) * Q_R
    size_C = (1 + L_HARM) * Q_S

    # Découpage en sous-vecteurs
    A1 = X[:size_A1]
    A2 = X[size_A1 : size_A1 + size_A2]
    Bj  = X[size_A1 + size_A2 : size_A1 + size_A2 + size_B]
    Ci  = X[size_A1 + size_A2 + size_B:]

    # 📌 Dictionnaire des noms des éléments
    nom_elements = {}

    # A1
    nom_elements[0] = "A1_{1,0}"

    # A2
    for i in range(1, 5):
        for j in range(N_HARM):
            index = size_A1 + (i - 1) * N_HARM + j
            nom_elements[index] = f"A2_{{{i},{j+1}}}"

    # B
    for q in range(Q_R):
        for m in range(M_HARM + 1):
            index = size_A1 + size_A2 + q * (M_HARM + 1) + m
            nom_elements[index] = f"Bj_{{{q+1},{m}}}"

    # C
    for l in range(L_HARM + 1):
        for q in range(Q_S):
            index = size_A1 + size_A2 + size_B + l * Q_S + q
            nom_elements[index] = f"Ci_{{{q+1},{l}}}"

    # ✅ Fonction d'accès générique
    def get_value_from_name(name, vecteur_X=None, dictionnaire=None):
        if vecteur_X is None:
            vecteur_X = X
        if dictionnaire is None:
            dictionnaire = nom_elements

        for idx, n in dictionnaire.items():
            if n == name:
                return vecteur_X[idx]
        raise ValueError(f"Nom '{name}' non trouvé dans le dictionnaire.")

    # print("Noms disponibles pour C :")
    for idx, name in nom_elements.items():
       if name.startswith("C_"):
           print(name)
    ###################################################################################################################################################
    # flux
    ###################################################################################################################################################

    def phi():
        S_ENCOCHE = C * (R4**2 - R3**2) / 2  
        phi_i = []
        for i in range(Q_S):
            nom_Ci0 = f"Ci_{{{i+1},0}}"
            Ci0 = get_value_from_name(nom_Ci0)
            # Constant terms outside the harmonic sum
            term1 = Ci0 * 0.5 * (R4**2 - R3**2)
            term2 = 0.5 * MU_0 * J_I[i, 0] * R4**2 * (0.5 * R4**2 * np.log(R4) - 0.25 * R4**2 - 0.5 * R3**2 * np.log(R3) + 0.25 * R3**2)
            term3 = 0.25 * MU_0 * J_I[i, 0] * 0.25 * (R4**4 - R3**4)
            termA = (term1 + term2 + term3) * C
            total = (L_U / S_ENCOCHE) * (termA)
            phi_i.append(total)

        return phi_i

    def psi(): 
        psi = N_C * np.transpose(C_MATRIX) @ phi() # Produit matriciel
        return psi

    # L’inductance de magnetisation
    L1 = abs(psi()[0]/ I_M)
    # print(L1*2*np.pi*50)
    return L1*2 * np.pi * FREQ


# # Generate the slip vector with 40 points from 0.0001 to 1
# slip = np.linspace(0.0001, 1, 40)

# # Initialize an empty list to store results
# Lm_values = []

# # Call the function for each slip value and store the result
# for s in slip:
#     Lm_values.append(calculate_Lm(s))

# # Create a DataFrame
# df = pd.DataFrame({
#     'slip': slip,
#     'Lm': Lm_values
# })

# # Save to CSV file
# df.to_csv('Lm_values.csv', index=False)
