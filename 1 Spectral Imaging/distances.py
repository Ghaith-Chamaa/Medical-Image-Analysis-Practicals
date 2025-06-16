import numpy as np


epsilon = 10**(-9)

# Minkowski Based Measurements ##################################################################
def dist_Minkowski(P, Q, k = 2):
    return ( np.sum( np.abs( P - Q )**k ) )**(1/k)

def dist_Manhattan(P, Q):
    return ( np.sum( np.abs( P - Q )**1 ) )**(1/1)

def dist_Euclidienne(P, Q):
    return ( np.sum( np.abs( P - Q )**2 ) )**(1/2)

def dist_Chebyshev(P,Q):
    return np.max( np.abs(P - Q) )

def dist_Sorensen(P, Q):
    return dist_Minkowski(P,Q, k=1) / np.sum(P + Q)

def dist_Soergel(P, Q):
    return dist_Minkowski(P,Q, k = 1) / np.sum( np.maximum(P,Q) )

def dist_Kulczynski_d(P,Q):
    return dist_Minkowski(P,Q, k = 1) / np.sum( np.minimum(P,Q) )

def dist_Canberra(P,Q):
    return np.sum( np.abs( P - Q ) / ( P + Q ) )

def dist_Lorentzian(P,Q):
    return np.sum( np.log( 1 + np.abs(P - Q) ) )

# Intersection ####################################################################################
def sim_Intersection(P,Q):
    min = np.minimum(P,Q)
    return np.sum( min )

def dist_Intersection(P,Q):
    return np.sum( np.abs(P - Q) ) / 2

# Wave Hedges #####################################################################################
def Wave_Hedges_1(P,Q):
    min = np.minimum(P,Q)
    max = np.maximum(P,Q)
    return np.sum( 1 - (min / max) )

def Wave_Hedges_2(P,Q):
    max = np.maximum(P,Q)
    return np.sum( ( np.abs( P - Q ) / max ) )

# Czekanowski ######################################################################################
def sim_Czekanowski(P,Q):
    min = np.minimum(P,Q)
    return np.sum( min ) / np.sum( P + Q )

# Motyka ###########################################################################################
def sim_Motyka(P,Q):
    min = np.minimum(P,Q)
    return np.sum( min ) / np.sum( np.abs( P + Q ) )

def dist_Motyka(P,Q):
    max = np.maximum(P,Q)
    return np.sum( max ) / np.sum( np.abs( P + Q ) )

# Kulczynski s #####################################################################################
def sim_Kulczynski_s(P,Q):
    min = np.minimum(P,Q)
    return np.sum( min ) / np.sum( np.abs( P - Q ) )

def dist_Kulczynski_s(P,Q):
    min = np.minimum(P,Q)
    return np.sum( np.abs( P - Q ) ) / np.sum( min )

# Ruzicka ##########################################################################################
def sim_Ruzicka(P,Q):
    min = np.minimum(P,Q)
    max = np.maximum(P,Q)
    return np.sum( min ) / np.sum( max )

# Tani-moto ########################################################################################
def Tani_moto_1(P,Q):
    min = np.minimum(P,Q)
    up = np.sum(P) + np.sum(Q) -2 * np.sum( min )
    bt = np.sum(P) + np.sum(Q) - np.sum( min )
    return up / bt

def Tani_moto_2(P,Q):
    min = np.minimum(P,Q)
    max = np.maximum(P,Q)
    return np.sum( max - min ) / np.sum( max )

# Inner Product #####################################################################################
def sim_Inner_product(P,Q):
    return np.sum(P * Q)

# Harmonic mean #####################################################################################
def sim_Harmonic_mean(P,Q):
    return 2 * np.sum( (P * Q) / (P + Q) )

# Cosine ############################################################################################
def sim_Cosine(P,Q):
    return np.sum(P * Q) / ( np.sqrt( np.sum( P**2 ) ) * np.sqrt( np.sum( Q**2 ) ) )

# Jaccard ###########################################################################################
def sim_Jaccard(P,Q):
    prod = sim_Inner_product(P,Q)
    return prod / ( np.sum( P**2 ) + np.sum(Q**2) - prod)

def dist_Jaccard(P,Q):
    prod = sim_Inner_product(P,Q)
    return np.sum( P - Q )**2 / ( np.sum(P**2) + np.sum(Q**2) - prod )

# Dice ###############################################################################################
def sim_Dice(P,Q):
    prod = np.sum( P * Q )
    return ( 2 * prod ) / ( np.sum(P**2) + np.sum(Q**2) )

def sim_Dice(P,Q):
    return np.sum( (P - Q)**2 ) / ( np.sum(P**2) + np.sum(Q**2) )

# Bhattacharyya ######################################################################################
def dist_Bhattacharyya(P,Q):
    return -np.log( np.sqrt(P * Q) )

# Hellinger ##########################################################################################
def dist_Hellinger_1(P, Q): 
    return np.sqrt( 2 * np.sum( ( np.sqrt(P) - np.sqrt(Q) )**2 ) )

def dist_Hellinger_2(P, Q): 
    return 2 * np.sqrt( 1 - np.sum( np.sqrt( P * Q ) ) )

# Matusita ###########################################################################################
def dist_Matusita_1(P,Q):
    return np.sqrt( np.sum( ( np.sqrt(P) - np.sqrt(Q) )**2 ) )

def dist_Matusita_2(P,Q):
    return np.sqrt( 2 - 2 * np.sum( np.sqrt(P * Q) ) )

# Squared-chord ######################################################################################
def dist_Squared_chord(P,Q):
    return np.sum( ( np.sqrt(P) - np.sqrt(Q) )**2 )

def sim_Squared_chord(P,Q):
    return 2 * np.sum( np.sqrt( P * Q ) - 1 )

# Squared Euclidean ##################################################################################
def dist_Squared_Euclidean(P,Q):
    return np.sum( (P-Q)**2 )

# Pearson chi square #################################################################################
def dist_Pearson_chi(P,Q):
    return np.sum( (P-Q)**2 / Q )

# Neyman chi #########################################################################################
def dist_Neyman_chi(P,Q):
    return np.sum( (P-Q)**2 / P )

# chi Square #########################################################################################
def dist_Squared_chi(P,Q):
    return np.sum( (P-Q)**2 / (P+Q) )

# Divergence #########################################################################################
def Divergence(P,Q):
    return 2 * np.sum( (P-Q)**2 / (P+Q)**2 )

# Clark ##############################################################################################
def dist_Clark(P,Q):
    return np.sqrt( np.sum( ( np.abs(P-Q) / (P+Q) )**2 ) )

# Additive Symmetric chi square ######################################################################
def Additive_symmetric_chi(P,Q):
    return np.sum( ( (P-Q)**2 * (P+Q) ) / (P*Q) )

# Kullback-Leibler ###################################################################################
def dist_KL(P, Q):
    Qe = np.where(Q == 0, epsilon, Q)  # Eliminate the 0 values
    Pe = np.where(P == 0, epsilon, P)  # Eliminate the 0 values
    return np.sum( P * np.log( Pe / Qe ) )

# Jeffreys ###########################################################################################
def dist_Jeffreys(P, Q):
    return ( dist_KL(P,Q) + dist_KL(Q,P) ) / 2.0

# K divergence #######################################################################################
def K_divergence(P, Q):
    Qe = np.where(Q == 0, epsilon, Q)  # Eliminate the 0 values
    return np.sum( P * np.log( (2*P) / (P + Qe) ) )

# Topsoe #############################################################################################
def dist_Topsoe(P, Q):
    Qe = np.where(Q == 0, epsilon, Q)  # Eliminate the 0 values
    return np.sum( P * np.log( (2*P) / (Qe + P) ) + Q * np.log( (2*Q) / (Qe + P) ) )

# Jensen-Shannon #####################################################################################
def dist_Jensen_Shannon(P,Q):
    return (K_divergence(P,Q) + K_divergence(Q,P)) / 2

# Jensen difference #####################################################################################
def delta_Jensen(P,Q):
    return np.sum( ( P + np.log(P) + Q + np.log(Q) ) / 2 - ( (P + Q) / 2 ) * np.log( (P+Q) / 2 ) )

################ SPECTRAL DISTANCES #####################################################################
# KLPD
def dist_KLPD(S1, S2):
    
    # Replace zeros to avoid division by zero
    S1 = np.where(S1 == 0, 1e-6, S1)
    S2 = np.where(S2 == 0, 1e-6, S2)

    # Calculate integrals (trapz) for both inputs
    k1 = np.trapz(S1, axis=-1)
    k2 = np.trapz(S2, axis=-1)

    # Normalize spectra
    N1 = S1 / k1[:, np.newaxis] if S1.ndim > 1 else S1 / k1
    N2 = S2 / k2[:, np.newaxis] if S2.ndim > 1 else S2 / k2

    # Calculate G and W
    G = k1 * np.trapz(N1 * np.log( (N1 / N2) + 1e-9), axis=-1) + k2 * np.trapz(N2 * np.log( (N2 / N1) + 1e-9), axis=-1)
    W = (k1 - k2) * np.log(k1 / k2)

##============================================================================

def dist_SAM(H1, H2, resolution=1.0):
    """
    Spectral Angle Mapper distance function (SAM) with trapezoidal integration.
    """
    epsilon = 1e-12  # petite valeur pour éviter les divisions par zéro

    dot_product = np.trapz(H1 * H2, dx=resolution, axis=0)
    norm_H1 = np.sqrt(np.trapz(H1**2, dx=resolution, axis=0)) + epsilon
    norm_H2 = np.sqrt(np.trapz(H2**2, dx=resolution, axis=0)) + epsilon

    A = dot_product / (norm_H1 * norm_H2)
    A = np.clip(A, -1.0, 1.0)  # assure que arccos est bien défini

    return np.arccos(A)

#============================================================================

def dist_SID(S1, S2):
    """
    Spectral Information Divergence (SID).
    """
    epsilon = 1e-12  # éviter log(0) et division par zéro

    # Évite les zéros et valeurs négatives
    S1 = np.clip(S1, epsilon, None)
    S2 = np.clip(S2, epsilon, None)

    k1 = np.trapz(S1) + epsilon
    k2 = np.trapz(S2) + epsilon

    H1 = S1 / k1
    H2 = S2 / k2

    # Normaliser
    H1 /= H1.sum() + epsilon
    H2 /= H2.sum() + epsilon

    H1 = np.clip(H1, epsilon, None)
    H2 = np.clip(H2, epsilon, None)

    return np.sum((H1 - H2) * (np.log(H1) - np.log(H2)))

#============================================================================

def dist_SGA(H1, H2):
    """
    Spectral Gradient Angle (SGA).
    """
    H1_grad = np.abs(np.diff(H1))
    H2_grad = np.abs(np.diff(H2))

    return dist_SAM(H1_grad, H2_grad)

#============================================================================

def dist_SCA(H1, H2):
    """
    Spectral Correlation Angle (SCA).
    """
    epsilon = 1e-12

    H1n = H1 - np.mean(H1)
    H2n = H2 - np.mean(H2)

    numerator = np.sum(H1n * H2n)
    denominator = np.sqrt(np.sum(H1n**2)) * np.sqrt(np.sum(H2n**2)) + epsilon

    c = numerator / denominator
    c = np.clip(c, -1.0, 1.0)

    return np.arccos((c + 1.0) / 2.0)