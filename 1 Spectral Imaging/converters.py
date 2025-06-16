import numpy as np
import cv2

# Référence blanche (D65)
Xn = 0.95047
Yn = 1.0
Zn = 1.08883

# ================================================================
# Conversion RGB <-> HSV
# ================================================================

def rgb_to_hsv(image):
    """
    Convertit une image RGB (sRGB, valeurs dans [0,255]) en HSV.
    On convertit d'abord l'image en float dans [0,1].
    Le résultat est un tableau float de forme (H,W,3) avec :
      H : teinte en degrés [0, 360)
      S : saturation dans [0, 1]
      V : valeur dans [0, 1]
    """
    # Normaliser en [0,1]
    img = image.astype('float32') / 255.0
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    Cmax = np.max(img, axis=-1)
    Cmin = np.min(img, axis=-1)
    delta = Cmax - Cmin

    # Initialisation des canaux
    H = np.zeros_like(Cmax)
    S = np.zeros_like(Cmax)
    V = Cmax.copy()

    # Calcul de H
    # Pour delta = 0, H reste à 0
    mask = delta > 1e-6  # éviter division par zéro

    # Lorsque Cmax == R
    mask_r = (mask) & (Cmax == R)
    H[mask_r] = 60 * (((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6)
    # Lorsque Cmax == G
    mask_g = (mask) & (Cmax == G)
    H[mask_g] = 60 * (((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2)
    # Lorsque Cmax == B
    mask_b = (mask) & (Cmax == B)
    H[mask_b] = 60 * (((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4)

    # Calcul de S
    S[mask] = delta[mask] / Cmax[mask]
    S[~mask] = 0

    # On retourne H, S, V empilés
    hsv = np.stack([H, S, V], axis=-1)
    return hsv

def hsv_to_rgb(hsv):
    """
    Convertit une image HSV (H en degrés [0,360), S et V dans [0,1])
    en image RGB (sRGB, valeurs dans [0,255], type uint8).
    """
    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    C = V * S
    H_prime = H / 60.0  # division en secteurs
    X = C * (1 - np.abs((H_prime % 2) - 1))
    m = V - C

    # Initialisation des canaux primaires
    R1 = np.zeros_like(H)
    G1 = np.zeros_like(H)
    B1 = np.zeros_like(H)

    # Conditions par secteur de H_prime
    cond0 = (H_prime >= 0) & (H_prime < 1)
    R1[cond0] = C[cond0]
    G1[cond0] = X[cond0]
    B1[cond0] = 0

    cond1 = (H_prime >= 1) & (H_prime < 2)
    R1[cond1] = X[cond1]
    G1[cond1] = C[cond1]
    B1[cond1] = 0

    cond2 = (H_prime >= 2) & (H_prime < 3)
    R1[cond2] = 0
    G1[cond2] = C[cond2]
    B1[cond2] = X[cond2]

    cond3 = (H_prime >= 3) & (H_prime < 4)
    R1[cond3] = 0
    G1[cond3] = X[cond3]
    B1[cond3] = C[cond3]

    cond4 = (H_prime >= 4) & (H_prime < 5)
    R1[cond4] = X[cond4]
    G1[cond4] = 0
    B1[cond4] = C[cond4]

    cond5 = (H_prime >= 5) & (H_prime < 6)
    R1[cond5] = C[cond5]
    G1[cond5] = 0
    B1[cond5] = X[cond5]

    R = (R1 + m)
    G = (G1 + m)
    B = (B1 + m)

    rgb = np.stack([R, G, B], axis=-1)
    # Convertir en [0,255] et en uint8
    rgb = np.clip(rgb * 255, 0, 255).astype('uint8')
    return rgb

# ================================================================
# Conversion RGB <-> XYZ
# ================================================================

def srgb_to_linear(rgb):
    """
    Convertit sRGB [0,1] en RGB linéaire [0,1] selon la loi gamma inverse.
    """
    rgb_linear = np.where(rgb > 0.04045,
                            ((rgb + 0.055) / 1.055) ** 2.4,
                            rgb / 12.92)
    return rgb_linear

def linear_to_srgb(rgb_linear):
    """
    Convertit RGB linéaire [0,1] en sRGB [0,1] (application de la courbe gamma).
    """
    srgb = np.where(rgb_linear > 0.0031308,
                    1.055 * (rgb_linear ** (1/2.4)) - 0.055,
                    12.92 * rgb_linear)
    return srgb

def rgb_to_xyz(image):
    """
    Convertit une image sRGB (valeurs dans [0,255]) en espace XYZ.
    On normalise l'image en [0,1], on passe en RGB linéaire, puis on applique
    la matrice de transformation (pour D65).
    Le résultat est en XYZ dans l'intervalle [0,1] (normalisé par Yn=1).
    """
    # Normaliser en [0,1]
    img = image.astype('float32') / 255.0
    # Convertir en RGB linéaire
    img_linear = srgb_to_linear(img)

    # Matrice de transformation sRGB -> XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    
    # Appliquer la transformation (vectorisée)
    # Reshape image en (N, 3)
    H, W, _ = img_linear.shape
    img_flat = img_linear.reshape((-1, 3))
    xyz_flat = np.dot(img_flat, M.T)
    xyz = xyz_flat.reshape((H, W, 3))
    return xyz

def xyz_to_rgb(xyz):
    """
    Convertit une image en espace XYZ (supposé normalisé, Yn=1)
    en sRGB (valeurs dans [0,255], type uint8).
    """
    # Matrice inverse de transformation XYZ -> sRGB (D65)
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]])
    
    H, W, _ = xyz.shape
    xyz_flat = xyz.reshape((-1, 3))
    rgb_linear_flat = np.dot(xyz_flat, M_inv.T)
    rgb_linear = rgb_linear_flat.reshape((H, W, 3))
    
    # Appliquer la correction gamma (linéaire -> sRGB)
    srgb = linear_to_srgb(rgb_linear)
    srgb = np.clip(srgb, 0, 1)
    srgb = (srgb * 255).astype('uint8')
    return srgb

# ================================================================
# Conversion RGB <-> Lab
# ================================================================

def xyz_to_lab(xyz):
    """
    Convertit une image en espace XYZ (supposé normalisé, Yn=1) en Lab.
    On utilise le blanc de référence D65 (Xn=0.95047, Yn=1.0, Zn=1.08883).
    Le résultat est Lab avec L dans [0,100] et a,b approximativement dans [-128,127].
    """

    X = xyz[..., 0] / Xn
    Y = xyz[..., 1] / Yn
    Z = xyz[..., 2] / Zn

    # Fonction de transformation f(t)
    epsilon = 0.008856  # (6/29)^3
    kappa   = 903.3

    def f(t):
        return np.where(t > epsilon, np.cbrt(t), (kappa * t + 16) / 116)
    
    fX = f(X)
    fY = f(Y)
    fZ = f(Z)

    L = 116 * fY - 16
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    lab = np.stack([L, a, b], axis=-1)
    return lab

def lab_to_xyz(lab):
    """
    Convertit une image en espace Lab en espace XYZ.
    On utilise le blanc de référence D65 (Xn=0.95047, Yn=1.0, Zn=1.08883).
    """

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    # Calcul de fY, fX, fZ
    fY = (L + 16) / 116
    fX = fY + (a / 500)
    fZ = fY - (b / 200)

    epsilon = 0.008856  # (6/29)^3
    kappa   = 903.3

    def f_inv(t):
        return np.where(t ** 3 > epsilon, t ** 3, (116 * t - 16) / kappa)
    
    X = Xn * f_inv(fX)
    Y = Yn * f_inv(fY)
    Z = Zn * f_inv(fZ)

    xyz = np.stack([X, Y, Z], axis=-1)
    return xyz

def rgb_to_lab(image):
    """
    Convertit une image sRGB (valeurs dans [0,255]) en Lab.
    La conversion se fait par une première conversion en XYZ puis en Lab.
    """
    xyz = rgb_to_xyz(image)
    lab = xyz_to_lab(xyz)
    return lab

def lab_to_rgb(lab):
    """
    Convertit une image en Lab en sRGB (valeurs dans [0,255], type uint8).
    La conversion se fait par une conversion de Lab vers XYZ puis XYZ vers RGB.
    """
    xyz = lab_to_xyz(lab)
    rgb = xyz_to_rgb(xyz)
    return rgb

# ================================================================
# Script de test
# ================================================================

if __name__ == '__main__':

    from tkinter import filedialog
    import matplotlib.pyplot as plt

    # Charger une image avec cv2 (attention : cv2 lit en BGR)
    img_bgr = cv2.imread(filedialog.askopenfilename(title="Open RGB image") )
    if img_bgr is None:
        raise IOError("Impossible de charger l'image.")
    
    # Convertir BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Conversion RGB -> HSV et retour
    hsv = rgb_to_hsv(img_rgb)

    # Conversion RGB -> XYZ et retour
    xyz = rgb_to_xyz(img_rgb)

    # Conversion RGB -> Lab et retour
    lab = rgb_to_lab(img_rgb)

    # Création d'une figure avec 6 sous-graphes
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()  # pour itérer facilement sur un tableau 1D de sous-graphes

    # Affichage de rgb
    axes[0].imshow(img_rgb[:,:,0], cmap='Reds')
    axes[0].set_title("Red")

    axes[1].imshow(img_rgb[:,:,1], cmap='Greens')
    axes[1].set_title("Green")

    axes[2].imshow(img_rgb[:,:,2], cmap='Blues')
    axes[2].set_title("Blue")

    # Affichage de hsv
    axes[3].imshow(hsv[:,:,0])
    axes[3].set_title("Hue")

    axes[4].imshow(hsv[:,:,1])
    axes[4].set_title("Saturation")

    axes[5].imshow(hsv[:,:,2])
    axes[5].set_title("Value")

    # Supprimer les axes pour une meilleure lisibilité
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Création d'une figure avec 6 sous-graphes
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()  # pour itérer facilement sur un tableau 1D de sous-graphes

    # Affichage de xyz
    axes[0].imshow(xyz[:,:,0], cmap='Reds')
    axes[0].set_title("x")

    axes[1].imshow(xyz[:,:,1], cmap='Greens')
    axes[1].set_title("y")

    axes[2].imshow(xyz[:,:,2], cmap='Blues')
    axes[2].set_title("z")

    # Affichage de lab
    axes[3].imshow(lab[:,:,0])
    axes[3].set_title("L")

    axes[4].imshow(lab[:,:,1])
    axes[4].set_title("a")

    axes[5].imshow(lab[:,:,2])
    axes[5].set_title("b")

    # Supprimer les axes pour une meilleure lisibilité
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()