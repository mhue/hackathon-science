"""
Coordonnées des 40 sphères pour le kissing number en dimension 5
Réseau D5

Note: K(5) n'est pas connu exactement
Bornes: 40 ≤ K(5) ≤ 44
Cette configuration donne 40 sphères (probablement optimal)
"""

import numpy as np

sqrt2 = np.sqrt(2)

# Centres des 40 sphères (configuration optimale connue)
sphere_centers = np.array([
    # Pattern (1, 1, 0, 0, 0) et permutations
    [sqrt2, sqrt2, 0.0, 0.0, 0.0],
    [sqrt2, 0.0, sqrt2, 0.0, 0.0],
    [sqrt2, 0.0, 0.0, sqrt2, 0.0],
    [sqrt2, 0.0, 0.0, 0.0, sqrt2],
    [0.0, sqrt2, sqrt2, 0.0, 0.0],
    [0.0, sqrt2, 0.0, sqrt2, 0.0],
    [0.0, sqrt2, 0.0, 0.0, sqrt2],
    [0.0, 0.0, sqrt2, sqrt2, 0.0],
    [0.0, 0.0, sqrt2, 0.0, sqrt2],
    [0.0, 0.0, 0.0, sqrt2, sqrt2],
    
    # Pattern (1, -1, 0, 0, 0) et permutations
    [sqrt2, -sqrt2, 0.0, 0.0, 0.0],
    [sqrt2, 0.0, -sqrt2, 0.0, 0.0],
    [sqrt2, 0.0, 0.0, -sqrt2, 0.0],
    [sqrt2, 0.0, 0.0, 0.0, -sqrt2],
    [0.0, sqrt2, -sqrt2, 0.0, 0.0],
    [0.0, sqrt2, 0.0, -sqrt2, 0.0],
    [0.0, sqrt2, 0.0, 0.0, -sqrt2],
    [0.0, 0.0, sqrt2, -sqrt2, 0.0],
    [0.0, 0.0, sqrt2, 0.0, -sqrt2],
    [0.0, 0.0, 0.0, sqrt2, -sqrt2],
    
    # Pattern (-1, 1, 0, 0, 0) et permutations
    [-sqrt2, sqrt2, 0.0, 0.0, 0.0],
    [-sqrt2, 0.0, sqrt2, 0.0, 0.0],
    [-sqrt2, 0.0, 0.0, sqrt2, 0.0],
    [-sqrt2, 0.0, 0.0, 0.0, sqrt2],
    [0.0, -sqrt2, sqrt2, 0.0, 0.0],
    [0.0, -sqrt2, 0.0, sqrt2, 0.0],
    [0.0, -sqrt2, 0.0, 0.0, sqrt2],
    [0.0, 0.0, -sqrt2, sqrt2, 0.0],
    [0.0, 0.0, -sqrt2, 0.0, sqrt2],
    [0.0, 0.0, 0.0, -sqrt2, sqrt2],
    
    # Pattern (-1, -1, 0, 0, 0) et permutations
    [-sqrt2, -sqrt2, 0.0, 0.0, 0.0],
    [-sqrt2, 0.0, -sqrt2, 0.0, 0.0],
    [-sqrt2, 0.0, 0.0, -sqrt2, 0.0],
    [-sqrt2, 0.0, 0.0, 0.0, -sqrt2],
    [0.0, -sqrt2, -sqrt2, 0.0, 0.0],
    [0.0, -sqrt2, 0.0, -sqrt2, 0.0],
    [0.0, -sqrt2, 0.0, 0.0, -sqrt2],
    [0.0, 0.0, -sqrt2, -sqrt2, 0.0],
    [0.0, 0.0, -sqrt2, 0.0, -sqrt2],
    [0.0, 0.0, 0.0, -sqrt2, -sqrt2]
])


if __name__ == "__main__":
    print("Centres des 40 sphères en dimension 5 (Réseau D5):")
    print(f"Shape: {sphere_centers.shape}")
    print(f"\nNombre de sphères: {len(sphere_centers)}")
    print(f"\n⚠️  Note: K(5) exact inconnu, bornes: 40 ≤ K(5) ≤ 44")
    
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
