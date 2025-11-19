"""
Coordonnées des 60 sphères pour le kissing number en dimension 6
Réseau D6

Note: K(6) n'est pas connu exactement
Bornes: 72 ≤ K(6) ≤ 78
Cette configuration D6 donne 60 sphères (valide mais sous-optimale)
Le réseau E6 donnerait 72 sphères (probablement optimal)
"""

import numpy as np
from itertools import combinations

sqrt2 = np.sqrt(2)

def generate_d6_centers():
    """Génère les 60 centres de sphères pour D6"""
    centers = []
    
    # 4 patterns de signes × C(6,2) = 4 × 15 = 60 vecteurs
    patterns = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    for pattern in patterns:
        for positions in combinations(range(6), 2):
            vector = [0.0] * 6
            vector[positions[0]] = sqrt2 * pattern[0] / np.sqrt(2)  # Normalisation
            vector[positions[1]] = sqrt2 * pattern[1] / np.sqrt(2)
            centers.append(vector)
    
    return np.array(centers)

# Centres des 60 sphères (configuration D6)
sphere_centers = generate_d6_centers()


if __name__ == "__main__":
    print("Centres des 60 sphères en dimension 6 (Réseau D6):")
    print(f"Shape: {sphere_centers.shape}")
    print(f"\nNombre de sphères: {len(sphere_centers)}")
    print(f"\n⚠️  Note: K(6) exact inconnu, bornes: 72 ≤ K(6) ≤ 78")
    print("   Cette configuration D6 donne 60 sphères")
    print("   Le réseau E6 donnerait 72 sphères (plus optimal)")
    
    print(f"\nPremières 10 coordonnées:")
    for i, c in enumerate(sphere_centers[:10]):
        print(f"  s{i+1:2d}: {c}")
    
    # Vérification
    norms = np.linalg.norm(sphere_centers, axis=1)
    print(f"\nNormes (doivent toutes être égales à 2):")
    print(f"  Min: {norms.min():.6f}")
    print(f"  Max: {norms.max():.6f}")
    print(f"  Toutes égales: {np.allclose(norms, 2.0)}")
    
    # Vérifier la contrainte du kissing number
    print(f"\nVérification de la contrainte:")
    min_dist = float('inf')
    for i in range(len(sphere_centers)):
        for j in range(i+1, len(sphere_centers)):
            dist = np.linalg.norm(sphere_centers[i] - sphere_centers[j])
            min_dist = min(min_dist, dist)
    
    max_norm = norms.max()
    print(f"  Distance minimale entre sphères: {min_dist:.6f}")
    print(f"  Norme maximale: {max_norm:.6f}")
    print(f"  Ratio (min_dist/max_norm): {min_dist/max_norm:.6f}")
    print(f"  Contrainte satisfaite: {min_dist >= max_norm}")
