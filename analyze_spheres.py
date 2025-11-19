"""
Analyze Sphere Information from Evolution History
Extracts and displays the number of spheres and their centers at each iteration
"""

import json
from typing import List, Dict, Any
from sphere_solver import extract_sphere_info_from_genome, compute_kissing_number_spheres


def analyze_evolution_history(history_file: str = "kissing_number_results.json"):
    """
    Analyze evolution history and extract sphere information
    
    Args:
        history_file: Path to JSON file with evolution results
    """
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {history_file} not found.")
        print("Please run the evolution first to generate results.")
        return
    
    history = data.get('history', [])
    top_individuals = data.get('top_individuals', [])
    
    if not history:
        print("No evolution history found in the results file.")
        return
    
    print("=" * 80)
    print("SPHERE ANALYSIS - EVOLUTION ITERATIONS")
    print("=" * 80)
    print()
    
    # Create a mapping of generation to best genome
    gen_to_genome = {}
    best_individual = data.get('best_individual', {})
    if best_individual.get('genome'):
        gen_to_genome[best_individual.get('generation', -1)] = best_individual['genome']
    
    # Also map from top individuals
    for ind in top_individuals:
        gen = ind.get('generation', -1)
        if gen not in gen_to_genome:
            gen_to_genome[gen] = ind.get('genome', '')
    
    # Analyze each generation
    for gen_data in history:
        generation = gen_data.get('generation', -1)
        best_fitness = gen_data.get('best_fitness', 0.0)
        
        # Get sphere information if available
        sphere_count = gen_data.get('sphere_count')
        sphere_centers = gen_data.get('sphere_centers', [])
        central_center = gen_data.get('central_center', [0.0, 0.0])
        dimension = gen_data.get('dimension', 2)
        
        # If sphere info not in history, try to compute from genome
        if sphere_count is None and generation in gen_to_genome:
            genome = gen_to_genome[generation]
            sphere_info = extract_sphere_info_from_genome(
                genome,
                context={'problem': 'kissing_number_dimension_2', 'expected_answer': 6}
            )
            sphere_count = sphere_info.get('sphere_count')
            sphere_centers = sphere_info.get('centers', [])
            central_center = sphere_info.get('central_center', [0.0, 0.0])
            dimension = sphere_info.get('dimension', 2)
        
        print(f"Generation {generation}:")
        print(f"  Best Fitness: {best_fitness:.4f}")
        
        if sphere_count is not None:
            print(f"  Number of Spheres: {sphere_count}")
            print(f"  Dimension: {dimension}D")
            print(f"  Central Center: {central_center}")
            print(f"  Sphere Centers ({len(sphere_centers)}):")
            for i, center in enumerate(sphere_centers, 1):
                print(f"    Sphere {i}: {center}")
        else:
            print("  Sphere information not available for this generation")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    sphere_counts = [gen.get('sphere_count') for gen in history if gen.get('sphere_count') is not None]
    if sphere_counts:
        unique_counts = sorted(set(sphere_counts))
        print(f"Unique sphere counts found: {unique_counts}")
        print(f"Most common sphere count: {max(set(sphere_counts), key=sphere_counts.count)}")
    
    # Best configuration
    best_gen = max(history, key=lambda x: x.get('best_fitness', 0))
    print(f"\nBest Generation: {best_gen.get('generation')}")
    print(f"Best Fitness: {best_gen.get('best_fitness', 0):.4f}")
    if best_gen.get('sphere_count') is not None:
        print(f"Number of Spheres: {best_gen.get('sphere_count')}")
        print(f"Sphere Centers:")
        for i, center in enumerate(best_gen.get('sphere_centers', []), 1):
            print(f"  {i}: {center}")


def extract_spheres_from_genome(genome: str, context: Dict[str, Any] = None):
    """
    Extract sphere information from a single genome
    
    Args:
        genome: The genome/prompt string
        context: Optional context
    """
    sphere_info = extract_sphere_info_from_genome(genome, context)
    
    print("=" * 80)
    print("SPHERE INFORMATION FROM GENOME")
    print("=" * 80)
    print(f"Genome: {genome}")
    print()
    print(f"Number of Spheres: {sphere_info['sphere_count']}")
    print(f"Dimension: {sphere_info['dimension']}D")
    print(f"Central Center: {sphere_info['central_center']}")
    print(f"Sphere Centers:")
    for i, center in enumerate(sphere_info['centers'], 1):
        print(f"  Sphere {i}: {center}")


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        history_file = sys.argv[1]
    else:
        history_file = "kissing_number_results.json"
    
    analyze_evolution_history(history_file)
    
    # Also try to extract from best individual if available
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
        
        best_individual = data.get('best_individual')
        if best_individual and best_individual.get('genome'):
            print("\n" + "=" * 80)
            print("BEST INDIVIDUAL SPHERE ANALYSIS")
            print("=" * 80)
            extract_spheres_from_genome(
                best_individual['genome'],
                context={'problem': 'kissing_number_dimension_2', 'expected_answer': 6}
            )
    except Exception as e:
        print(f"\nNote: Could not analyze best individual: {e}")


if __name__ == "__main__":
    main()

