"""
Kissing Number Problem Solver Agent
Résout le problème du kissing number en dimension 3 via un agent LLM itératif
"""

import anthropic
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import os
from dotenv import load_dotenv


class KissingNumberAgent:
    """
    Agent LLM qui génère itérativement des algorithmes d'optimisation
    pour résoudre le kissing number problem
    """
    
    def __init__(
        self, 
        api_key: str, 
        dimension: int = 3,
        max_iterations: int = 15,
        target_spheres: int = 12
    ):
        """
        Args:
            api_key: Clé API Anthropic
            dimension: Dimension du problème (default: 3)
            max_iterations: Nombre maximum d'itérations
            target_spheres: Nombre cible de sphères à trouver
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.dimension = dimension
        self.attempts_history: List[Dict] = []
        self.max_iterations = max_iterations
        self.target_spheres = target_spheres
        
    def validate_configuration(self, centers: np.ndarray) -> Dict:
        """
        Valide qu'une configuration satisfait le Lemme du kissing number
        
        Contrainte: min{||x-y|| : x≠y ∈ C} ≥ max{||x|| : x ∈ C}
        
        Args:
            centers: Array numpy de shape (N, dimension)
            
        Returns:
            Dict avec validation, métriques et diagnostics
        """
        if len(centers) == 0:
            return {
                "valid": False, 
                "reason": "Empty configuration",
                "num_spheres": 0
            }
        
        if centers.shape[1] != self.dimension:
            return {
                "valid": False,
                "reason": f"Wrong dimension: {centers.shape[1]} instead of {self.dimension}",
                "num_spheres": len(centers)
            }
        
        # Vérifier que 0 n'est pas dans C
        if np.any(np.all(centers == 0, axis=1)):
            return {
                "valid": False,
                "reason": "Configuration contains origin (0 vector)",
                "num_spheres": len(centers)
            }
        
        # Calcul des distances pairwise
        n = len(centers)
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append(dist)
        
        # Calcul des normes
        norms = np.linalg.norm(centers, axis=1)
        
        if len(distances) == 0:
            min_dist = float('inf')
        else:
            min_dist = min(distances)
            
        max_norm = max(norms)
        
        # Vérification de la contrainte
        is_valid = min_dist >= max_norm
        
        # Calcul de statistiques supplémentaires
        avg_dist = np.mean(distances) if distances else 0
        std_dist = np.std(distances) if distances else 0
        
        return {
            "valid": is_valid,
            "num_spheres": len(centers),
            "min_distance": float(min_dist),
            "max_norm": float(max_norm),
            "avg_distance": float(avg_dist),
            "std_distance": float(std_dist),
            "ratio": float(min_dist / max_norm) if max_norm > 0 else 0,
            "distances": distances[:10] if len(distances) <= 10 else None  # Sample pour debug
        }
    
    def execute_code(self, code: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Exécute le code généré par l'agent dans un environnement contrôlé
        
        Args:
            code: Code Python à exécuter
            
        Returns:
            (centers, error) où centers est un np.array ou None si erreur
        """
        try:
            # Environnement d'exécution avec bibliothèques autorisées
            allowed_globals = {
                "np": np,
                "numpy": np,
                "__builtins__": __builtins__
            }
            
            # Essayer d'importer scipy si disponible
            try:
                import scipy
                from scipy import optimize, spatial
                allowed_globals["scipy"] = scipy
                allowed_globals["optimize"] = optimize
                allowed_globals["spatial"] = spatial
            except ImportError:
                pass
            
            local_scope = {}
            exec(code, allowed_globals, local_scope)
            
            # Vérifier que la fonction requise existe
            if "generate_configuration" not in local_scope:
                return None, "Function 'generate_configuration' not found in code"
            
            # Exécuter la fonction
            centers = local_scope["generate_configuration"]()
            
            # Vérifications de type et forme
            if not isinstance(centers, np.ndarray):
                return None, f"Function must return np.ndarray, got {type(centers)}"
            
            if len(centers.shape) != 2:
                return None, f"Array must be 2D, got shape {centers.shape}"
            
            if centers.shape[1] != self.dimension:
                return None, f"Wrong dimension: expected {self.dimension}, got {centers.shape[1]}"
            
            if len(centers) == 0:
                return None, "Generated empty configuration"
            
            return centers, None
            
        except Exception as e:
            return None, f"Execution error: {type(e).__name__}: {str(e)}"
    
    def generate_feedback(self, validation: Optional[Dict], error: Optional[str] = None) -> str:
        """
        Génère un feedback détaillé pour guider l'agent
        
        Args:
            validation: Résultat de la validation (ou None si erreur d'exécution)
            error: Message d'erreur d'exécution (ou None si succès)
            
        Returns:
            Feedback formaté en texte
        """
        if error:
            return f"""❌ ERREUR D'EXÉCUTION
{error}

SUGGESTIONS:
- Vérifie que ta fonction s'appelle bien 'generate_configuration'
- Vérifie qu'elle retourne un np.array de shape (N, {self.dimension})
- Vérifie les imports (numpy doit être importé comme 'np')
- Évite les divisions par zéro ou valeurs infinies
"""
        
        if not validation["valid"]:
            reason = validation.get("reason", "Contrainte non satisfaite")
            
            feedback = f"""❌ CONFIGURATION INVALIDE

RAISON: {reason}

MÉTRIQUES:
- Nombre de sphères: {validation['num_spheres']}
- Distance minimale entre points: {validation.get('min_distance', 'N/A'):.6f}
- Norme maximale des points: {validation.get('max_norm', 'N/A'):.6f}
- Ratio min_dist/max_norm: {validation.get('ratio', 'N/A'):.6f} (doit être ≥ 1.0)

"""
            
            if validation.get('ratio', 0) < 1.0:
                gap = 1.0 - validation.get('ratio', 0)
                feedback += f"""PROBLÈME PRINCIPAL:
La contrainte min(distances) ≥ max(normes) n'est PAS satisfaite.
Gap: {gap:.6f}

Les points sont trop PROCHES les uns des autres par rapport à leur distance au centre.

SOLUTIONS POSSIBLES:
1. Augmente l'espacement entre les points (force de répulsion plus forte)
2. Réduis la distance des points au centre (place-les plus près de l'origine)
3. Utilise une approche en deux phases: génère d'abord, puis optimise les distances
"""
            
            return feedback
        
        # Configuration valide
        feedback = f"""✓ CONFIGURATION VALIDE

MÉTRIQUES:
- Nombre de sphères: {validation['num_spheres']} / {self.target_spheres}
- Distance minimale: {validation['min_distance']:.6f}
- Norme maximale: {validation['max_norm']:.6f}
- Ratio: {validation['ratio']:.6f}
- Distance moyenne: {validation.get('avg_distance', 0):.6f}
- Écart-type distances: {validation.get('std_distance', 0):.6f}

"""
        
        if validation['num_spheres'] >= self.target_spheres:
            feedback += f"""🎉 SUCCÈS! 
Tu as atteint l'objectif de {self.target_spheres} sphères!
Configuration finale validée.
"""
        else:
            missing = self.target_spheres - validation['num_spheres']
            feedback += f"""⚠️ OBJECTIF NON ATTEINT
Il manque encore {missing} sphères pour atteindre {self.target_spheres}.

SUGGESTIONS:
1. Essaie une densité de placement plus élevée
2. Utilise des structures géométriques régulières (icosaèdre, dodécaèdre)
3. Explore des algorithmes évolutionnaires avec population plus grande
4. Augmente le nombre d'itérations de ton optimiseur
"""
        
        return feedback
    
    def get_system_prompt(self) -> str:
        """Construit le prompt système pour l'agent"""
        return f"""Tu es un expert en optimisation géométrique et en algorithmes numériques.

# MISSION
Résoudre le kissing number problem en dimension {self.dimension}: trouver le maximum de sphères 
pouvant toucher une sphère centrale sans se chevaucher.

# CONTRAINTE MATHÉMATIQUE (Le Lemme)
Soit C ⊂ ℝ^{self.dimension} un ensemble de points satisfaisant:
1. 0 ∉ C (l'origine n'est pas dans C)
2. min{{||x-y|| : x≠y ∈ C}} ≥ max{{||x|| : x ∈ C}}

Si cette contrainte est satisfaite, alors les sphères unitaires centrées en {{2x/||x|| : x ∈ C}}
forment une configuration de kissing valide.

# OBJECTIF
Maximiser |C| (le nombre de points). Pour dimension {self.dimension}, l'objectif est {self.target_spheres}.

# FORMAT DE CODE STRICT
Tu DOIS générer un code Python avec:

1. Imports en haut (numpy obligatoire, scipy optionnel)
2. Une fonction nommée EXACTEMENT `generate_configuration` qui:
   - Ne prend AUCUN paramètre
   - Retourne un np.array de shape (N, {self.dimension}) où N est le nombre de points
   - Les points NE DOIVENT PAS être normalisés (la normalisation se fait après validation)

3. Pas de code en dehors de la fonction (pas de if __name__ == "__main__")

# EXEMPLE DE STRUCTURE
```python
import numpy as np

def generate_configuration():
    # Ton algorithme ici
    centers = ...  # np.array de shape (N, {self.dimension})
    return centers
```

# APPROCHES SUGGÉRÉES
- Placement sur des structures géométriques régulières (polyèdres)
- Optimisation par forces (répulsion/attraction)
- Simulated annealing sur variété
- Algorithmes évolutionnaires
- Optimisation sous contraintes avec scipy.optimize

# RÈGLES IMPORTANTES
- NE PAS hardcoder de solutions connues (comme les coordonnées de l'icosaèdre)
- DÉVELOPPER un algorithme d'optimisation générique
- TESTER différentes approches si les précédentes échouent

Génère UNIQUEMENT le code Python, sans explication, sans markdown, sans backticks.
"""
    
    def run(self, verbose: bool = True) -> Dict:
        """
        Lance la boucle d'optimisation itérative
        
        Args:
            verbose: Afficher les détails pendant l'exécution
            
        Returns:
            Dict contenant le meilleur résultat trouvé
        """
        system_prompt = self.get_system_prompt()
        conversation_history = []
        best_result = {"num_spheres": 0}
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"  KISSING NUMBER PROBLEM SOLVER - DIMENSION {self.dimension}")
            print(f"  Objectif: {self.target_spheres} sphères")
            print(f"  Itérations max: {self.max_iterations}")
            print(f"{'='*70}\n")
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n{'─'*70}")
                print(f"  ITÉRATION {iteration + 1}/{self.max_iterations}")
                print(f"{'─'*70}")
            
            # Construire le message utilisateur
            if iteration == 0:
                user_message = "Génère un premier algorithme pour trouver une configuration de sphères optimale."
            else:
                last_attempt = self.attempts_history[-1]
                user_message = f"""FEEDBACK DE L'ITÉRATION PRÉCÉDENTE:

{last_attempt['feedback']}

MEILLEUR RÉSULTAT ACTUEL: {best_result['num_spheres']} sphères (objectif: {self.target_spheres})

Analyse ce qui n'a pas fonctionné et génère une NOUVELLE approche différente pour améliorer.
"""
            
            # Appel à Claude
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[
                        *conversation_history,
                        {"role": "user", "content": user_message}
                    ]
                )
                
                code = response.content[0].text.strip()
                
                # Nettoyer le code (enlever markdown si présent)
                if code.startswith("```"):
                    lines = code.split("\n")
                    code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                
                if verbose:
                    print(f"\n📝 Code généré ({len(code)} caractères)")
                    print(f"{'─'*70}")
                    print(code[:800] + ("..." if len(code) > 800 else ""))
                    print(f"{'─'*70}")
                
            except Exception as e:
                print(f"\n❌ Erreur API: {e}")
                continue
            
            # Exécution du code
            centers, error = self.execute_code(code)
            
            # Validation
            if error:
                validation = None
                feedback = self.generate_feedback(None, error)
            else:
                validation = self.validate_configuration(centers)
                feedback = self.generate_feedback(validation)
            
            if verbose:
                print(f"\n{feedback}")
            
            # Sauvegarder l'historique
            attempt = {
                "iteration": iteration + 1,
                "code": code,
                "validation": validation,
                "feedback": feedback,
                "error": error,
                "timestamp": datetime.now().isoformat()
            }
            self.attempts_history.append(attempt)
            
            # Mettre à jour le meilleur résultat
            if validation and validation["valid"]:
                if validation["num_spheres"] > best_result["num_spheres"]:
                    best_result = validation.copy()
                    best_result["code"] = code
                    best_result["centers"] = centers
                    best_result["iteration"] = iteration + 1
                    
                    if verbose:
                        print(f"\n🌟 NOUVEAU RECORD: {validation['num_spheres']} sphères!")
            
            # Condition d'arrêt (succès)
            if validation and validation["valid"] and validation["num_spheres"] >= self.target_spheres:
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"  🎉 SUCCÈS! Objectif atteint en {iteration + 1} itérations!")
                    print(f"{'='*70}\n")
                return best_result
            
            # Ajouter à l'historique de conversation
            conversation_history.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": code}
            ])
        
        # Fin des itérations sans succès complet
        if verbose:
            print(f"\n{'='*70}")
            if best_result["num_spheres"] > 0:
                print(f"  ⚠️  Objectif non atteint après {self.max_iterations} itérations")
                print(f"  Meilleur résultat: {best_result['num_spheres']}/{self.target_spheres} sphères")
            else:
                print(f"  ❌ Aucune configuration valide trouvée")
            print(f"{'='*70}\n")
        
        return best_result
    
    def save_results(self, filename: str = "kissing_number_results.json"):
        """Sauvegarde tous les résultats dans un fichier JSON"""
        results = {
            "dimension": self.dimension,
            "target_spheres": self.target_spheres,
            "max_iterations": self.max_iterations,
            "attempts": self.attempts_history
        }
        
        # Convertir les np.arrays en listes pour JSON
        for attempt in results["attempts"]:
            if attempt["validation"] and "distances" in attempt["validation"]:
                if attempt["validation"]["distances"]:
                    attempt["validation"]["distances"] = [float(d) for d in attempt["validation"]["distances"]]
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés dans: {filename}")


def main():
    """Fonction principale pour exécuter l'agent"""
    
    # Charger les variables d'environnement
    load_dotenv()
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    if not API_KEY:
        print("❌ ERREUR: Variable d'environnement ANTHROPIC_API_KEY non trouvée!")
        print("Crée un fichier .env avec: ANTHROPIC_API_KEY=ta_clé_api")
        print("Obtiens une clé sur: https://console.anthropic.com/")
        return
    
    # Créer l'agent
    agent = KissingNumberAgent(
        api_key=API_KEY,
        dimension=3,
        max_iterations=15,
        target_spheres=12
    )
    
    # Lancer la résolution
    result = agent.run(verbose=True)
    
    # Afficher le résultat final
    if result.get("num_spheres", 0) >= 12:
        print("\n" + "="*70)
        print("  ✅ CONFIGURATION FINALE TROUVÉE")
        print("="*70)
        print(f"\nNombre de sphères: {result['num_spheres']}")
        print(f"Trouvé à l'itération: {result.get('iteration', 'N/A')}")
        print(f"Ratio min_dist/max_norm: {result['ratio']:.6f}")
        
        print("\n📊 Centres des sphères:")
        print(result['centers'])
        
        print("\n💻 Code de l'algorithme gagnant:")
        print("─"*70)
        print(result['code'])
        print("─"*70)
        
        # Normaliser et afficher la configuration finale
        centers_normalized = 2 * result['centers'] / np.linalg.norm(result['centers'], axis=1, keepdims=True)
        print("\n🎯 Configuration finale (centres normalisés pour kissing):")
        print(centers_normalized)
        
    else:
        print("\n" + "="*70)
        print("  ⚠️  OBJECTIF NON ATTEINT")
        print("="*70)
        if result.get("num_spheres", 0) > 0:
            print(f"\nMeilleur résultat: {result['num_spheres']} sphères")
            print("\nSuggestions:")
            print("- Augmente max_iterations (actuellement 15)")
            print("- Modifie le prompt système pour guider différemment")
            print("- Ajoute des contraintes plus spécifiques")
        else:
            print("\nAucune configuration valide trouvée.")
            print("Vérifie les erreurs dans l'historique.")
    
    # Sauvegarder les résultats
    agent.save_results()
    
    return result


if __name__ == "__main__":
    result = main()
