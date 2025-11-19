"""
Kissing Number Solution for Dimension 4
Calcule les coordonnées exactes des 24 sphères en utilisant le réseau D4
"""

import numpy as np
from itertools import combinations, permutations
import json


def generate_d4_lattice_vectors():
    """
    Génère les vecteurs du réseau D4 qui forment une configuration de kissing optimale.
    
    Le réseau D4 est défini par tous les vecteurs (x1, x2, x3, x4) où:
    - Tous les xi sont des entiers
    - La somme x1 + x2 + x3 + x4 est paire
    
    Pour le kissing number, on prend les vecteurs de norme minimale non-nulle.
    Ces vecteurs sont de la forme (±1, ±1, 0, 0) avec toutes les permutations.
    
    Returns:
        np.array de shape (24, 4) contenant les 24 vecteurs
    """
    vectors = []
    
    # Génération de tous les vecteurs (±1, ±1, 0, 0) et leurs permutations
    # On commence avec la base (1, 1, 0, 0)
    base_patterns = [
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1)
    ]
    
    # Pour chaque pattern de signes
    for pattern in base_patterns:
        # Pour chaque paire de positions où placer les ±1
        for positions in combinations(range(4), 2):
            vector = [0, 0, 0, 0]
            vector[positions[0]] = pattern[0]
            vector[positions[1]] = pattern[1]
            vectors.append(vector)
    
    return np.array(vectors, dtype=float)


def validate_d4_configuration(vectors):
    """
    Valide que les vecteurs D4 satisfont la contrainte du kissing number.
    
    Contrainte: min{||x-y|| : x≠y ∈ C} ≥ max{||x|| : x ∈ C}
    
    Args:
        vectors: np.array de shape (N, 4)
        
    Returns:
        Dict avec les métriques de validation
    """
    n = len(vectors)
    
    # Vérifier que 0 n'est pas dans la configuration
    has_zero = np.any(np.all(vectors == 0, axis=1))
    
    # Calcul des normes
    norms = np.linalg.norm(vectors, axis=1)
    max_norm = np.max(norms)
    
    # Calcul des distances pairwise
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(vectors[i] - vectors[j])
            distances.append(dist)
    
    min_distance = np.min(distances)
    avg_distance = np.mean(distances)
    
    # Vérification de la contrainte
    is_valid = min_distance >= max_norm and not has_zero
    
    return {
        "valid": is_valid,
        "num_vectors": n,
        "min_distance": float(min_distance),
        "max_norm": float(max_norm),
        "avg_distance": float(avg_distance),
        "ratio": float(min_distance / max_norm) if max_norm > 0 else 0,
        "all_norms_equal": bool(np.allclose(norms, norms[0])),
        "norm_value": float(norms[0]) if len(norms) > 0 else 0
    }


def compute_kissing_spheres_centers(vectors):
    """
    Convertit les vecteurs D4 en centres de sphères pour la configuration de kissing.
    
    Si C satisfait la contrainte, alors les sphères unitaires centrées en 
    {2x/||x|| : x ∈ C} forment une configuration de kissing valide.
    
    Args:
        vectors: np.array de shape (N, 4)
        
    Returns:
        np.array de shape (N, 4) des centres normalisés
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = 2 * vectors / norms
    return normalized


def visualize_properties(vectors):
    """
    Analyse et affiche les propriétés géométriques de la configuration
    """
    print("\n" + "="*70)
    print("  PROPRIÉTÉS GÉOMÉTRIQUES DE LA CONFIGURATION D4")
    print("="*70)
    
    # Statistiques de base
    n = len(vectors)
    print(f"\n📊 Nombre de vecteurs: {n}")
    
    # Normes
    norms = np.linalg.norm(vectors, axis=1)
    print(f"\n📏 Normes des vecteurs:")
    print(f"  - Toutes égales: {np.allclose(norms, norms[0])}")
    print(f"  - Valeur: {norms[0]:.6f}")
    print(f"  - Norme au carré: {norms[0]**2:.6f}")
    
    # Distances
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(vectors[i] - vectors[j])
            distances.append(dist)
    
    print(f"\n📐 Distances entre vecteurs:")
    print(f"  - Minimum: {min(distances):.6f}")
    print(f"  - Maximum: {max(distances):.6f}")
    print(f"  - Moyenne: {np.mean(distances):.6f}")
    print(f"  - Écart-type: {np.std(distances):.6f}")
    
    # Distribution des distances
    unique_distances = np.unique(np.round(distances, 6))
    print(f"\n  Distances uniques ({len(unique_distances)}):")
    for d in unique_distances[:5]:  # Afficher les 5 premières
        count = sum(1 for dist in distances if abs(dist - d) < 1e-6)
        print(f"    {d:.6f}: {count} paires")
    
    # Produits scalaires
    dot_products = []
    for i in range(n):
        for j in range(i+1, n):
            dot = np.dot(vectors[i], vectors[j])
            dot_products.append(dot)
    
    print(f"\n🔗 Produits scalaires:")
    print(f"  - Minimum: {min(dot_products):.6f}")
    print(f"  - Maximum: {max(dot_products):.6f}")
    
    unique_dots = np.unique(np.round(dot_products, 6))
    print(f"  - Valeurs uniques: {unique_dots}")


def save_configuration(vectors, sphere_centers, validation, filename="d4_kissing_config.json"):
    """
    Sauvegarde la configuration dans un fichier JSON
    """
    data = {
        "dimension": 4,
        "kissing_number": 24,
        "lattice_type": "D4",
        "d4_vectors": vectors.tolist(),
        "sphere_centers": sphere_centers.tolist(),
        "validation": validation,
        "description": "Configuration optimale du kissing number en dimension 4 via le réseau D4"
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Configuration sauvegardée dans: {filename}")


def main():
    """
    Fonction principale
    """
    print("\n" + "="*70)
    print("  KISSING NUMBER EN DIMENSION 4 - RÉSEAU D4")
    print("="*70)
    
    # Générer les vecteurs D4
    print("\n🔄 Génération des vecteurs du réseau D4...")
    vectors = generate_d4_lattice_vectors()
    
    print(f"✓ {len(vectors)} vecteurs générés")
    
    # Valider la configuration
    print("\n🔍 Validation de la contrainte du kissing number...")
    validation = validate_d4_configuration(vectors)
    
    print("\n" + "─"*70)
    print("  RÉSULTATS DE LA VALIDATION")
    print("─"*70)
    print(f"\n{'✅' if validation['valid'] else '❌'} Configuration valide: {validation['valid']}")
    print(f"  - Nombre de vecteurs: {validation['num_vectors']}")
    print(f"  - Distance minimale: {validation['min_distance']:.6f}")
    print(f"  - Norme maximale: {validation['max_norm']:.6f}")
    print(f"  - Ratio (min_dist/max_norm): {validation['ratio']:.6f}")
    print(f"  - Toutes les normes égales: {validation['all_norms_equal']}")
    
    # Afficher quelques vecteurs
    print("\n" + "─"*70)
    print("  EXEMPLES DE VECTEURS D4")
    print("─"*70)
    print("\nPremiers 6 vecteurs:")
    for i, v in enumerate(vectors[:6]):
        print(f"  v{i+1}: {v}")
    
    # Calculer les centres des sphères
    print("\n🎯 Calcul des centres des sphères de kissing...")
    sphere_centers = compute_kissing_spheres_centers(vectors)
    
    print("\nPremiers 6 centres de sphères (normalisés):")
    for i, c in enumerate(sphere_centers[:6]):
        norm = np.linalg.norm(c)
        print(f"  s{i+1}: {c}  (norme: {norm:.6f})")
    
    # Analyser les propriétés
    visualize_properties(vectors)
    
    # Vérifier les propriétés du réseau D4
    print("\n" + "─"*70)
    print("  PROPRIÉTÉS DU RÉSEAU D4")
    print("─"*70)
    print("\n✓ Tous les vecteurs ont la forme (±1, ±1, 0, 0) et permutations")
    print("✓ Norme carrée constante: 2")
    print(f"✓ Norme constante: √2 = {np.sqrt(2):.6f}")
    print("✓ Configuration géométrique: 24-cell (polychore régulier)")
    
    # Sauvegarder
    save_configuration(vectors, sphere_centers, validation)
    
    # Afficher la configuration complète
    print("\n" + "="*70)
    print("  CONFIGURATION COMPLÈTE DES 24 VECTEURS D4")
    print("="*70)
    print("\nVecteurs du réseau D4:")
    for i, v in enumerate(vectors):
        print(f"  v{i+1:2d}: [{v[0]:2.0f}, {v[1]:2.0f}, {v[2]:2.0f}, {v[3]:2.0f}]")
    
    print("\n" + "="*70)
    print("  CENTRES DES 24 SPHÈRES DE KISSING")
    print("="*70)
    print("\nCentres normalisés (sphères unitaires):")
    for i, c in enumerate(sphere_centers):
        print(f"  s{i+1:2d}: [{c[0]:7.4f}, {c[1]:7.4f}, {c[2]:7.4f}, {c[3]:7.4f}]")
    
    return vectors, sphere_centers, validation


if __name__ == "__main__":
    vectors, sphere_centers, validation = main()
    
    print("\n" + "="*70)
    print("  ✅ SOLUTION EXACTE DU KISSING NUMBER EN DIMENSION 4")
    print("="*70)
    print(f"\n🎯 Kissing number K(4) = {len(vectors)}")
    print("📐 Configuration: Réseau D4 (24-cell)")
    print(f"✓ Contrainte satisfaite: ratio = {validation['ratio']:.6f} ≥ 1.0")
    print("\n")
