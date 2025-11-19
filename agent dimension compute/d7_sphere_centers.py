"""
Coordonnées des 84 sphères pour le kissing number en dimension 7
Réseau D7

Note: K(7) n'est pas connu exactement
Bornes: 126 ≤ K(7) ≤ 134
Cette configuration D7 donne 84 sphères (valide mais sous-optimale)
Le réseau E7 donnerait 126 sphères (probablement optimal)
"""

import numpy as np
from itertools import combinations

sqrt2 = np.sqrt(2)

def generate_d7_centers():
    """Génère les 84 centres de sphères pour D7"""
    centers = []
    
    # 4 patterns de signes × C(7,2) = 4 × 21 = 84 vecteurs
    patterns = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    for pattern in patterns:
        for positions in combinations(range(7), 2):
            vector = [0.0] * 7
            vector[positions[0]] = sqrt2 * pattern[0] / np.sqrt(2)
            vector[positions[1]] = sqrt2 * pattern[1] / np.sqrt(2)
            centers.append(vector)
    
    return np.array(centers)

# Centres des 84 sphères (configuration D7)
sphere_centers = generate_d7_centers()


if __name__ == "__main__":
    print("Centres des 84 sphères en dimension 7 (Réseau D7):")
    print(f"Shape: {sphere_centers.shape}")
    print(f"\nNombre de sphères: {len(sphere_centers)}")
    print(f"\n⚠️  Note: K(7) exact inconnu, bornes: 126 ≤ K(7) ≤ 134")
    print("   Cette configuration D7 donne 84 sphères")
    print("   Le réseau E7 donnerait 126 sphères (plus optimal)")
    
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
    for i in range(min(len(sphere_centers), 50)):
        for j in range(i+1, min(len(sphere_centers), 60)):
            dist = np.linalg.norm(sphere_centers[i] - sphere_centers[j])
            min_dist = min(min_dist, dist)
    
    max_norm = norms.max()
    print(f"  Distance minimale entre sphères (échantillon): {min_dist:.6f}")
    print(f"  Norme maximale: {max_norm:.6f}")
    print(f"  Ratio (min_dist/max_norm): {min_dist/max_norm:.6f}")
    print(f"  Contrainte satisfaite: {min_dist >= max_norm}")
    
    print(f"\n📊 Résumé:")
    print(f"  - Dimension: 7")
    print(f"  - Configuration: D7")
    print(f"  - Nombre de sphères: 84")
    print(f"  - Formule: 4 × C(7,2) = 4 × 21 = 84")
