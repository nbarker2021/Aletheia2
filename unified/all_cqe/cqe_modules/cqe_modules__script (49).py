# 5. MORSR Explorer
morsr_explorer_code = '''"""
MORSR (Multi-Objective Random Search and Repair) Explorer

Implements the core MORSR algorithm with parity-preserving moves,
triadic repair mechanisms, and geometric constraint satisfaction.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import random
from .objective_function import CQEObjectiveFunction
from .parity_channels import ParityChannels

class MORSRExplorer:
    """MORSR exploration algorithm for CQE optimization."""
    
    def __init__(self, 
                 objective_function: CQEObjectiveFunction,
                 parity_channels: ParityChannels,
                 random_seed: Optional[int] = None):
        
        self.objective_function = objective_function
        self.parity_channels = parity_channels
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # MORSR parameters
        self.pulse_size = 0.1
        self.repair_threshold = 0.05
        self.exploration_decay = 0.95
        self.parity_enforcement_strength = 0.8
    
    def explore(self, 
               initial_vector: np.ndarray,
               reference_channels: Dict[str, float],
               max_iterations: int = 50,
               domain_context: Optional[Dict] = None,
               convergence_threshold: float = 1e-4) -> Tuple[np.ndarray, Dict[str, float], float]:
        """
        Execute MORSR exploration starting from initial vector.
        
        Returns:
            best_vector: Optimal vector found
            best_channels: Parity channels of optimal vector  
            best_score: Objective function value
        """
        
        current_vector = initial_vector.copy()
        current_score = self.objective_function.evaluate(
            current_vector, reference_channels, domain_context
        )["phi_total"]
        
        best_vector = current_vector.copy()
        best_score = current_score
        best_channels = self.parity_channels.extract_channels(best_vector)
        
        # Exploration history
        history = {
            "scores": [current_score],
            "vectors": [current_vector.copy()],
            "improvements": 0,
            "repairs": 0
        }
        
        current_pulse_size = self.pulse_size
        
        for iteration in range(max_iterations):
            # Generate candidate moves
            candidates = self._generate_candidates(
                current_vector, current_pulse_size, reference_channels
            )
            
            # Evaluate candidates
            best_candidate = None
            best_candidate_score = current_score
            
            for candidate in candidates:
                # Apply triadic repair if needed
                repaired_candidate = self._triadic_repair(candidate, reference_channels)
                
                # Evaluate candidate
                candidate_scores = self.objective_function.evaluate(
                    repaired_candidate, reference_channels, domain_context
                )
                candidate_score = candidate_scores["phi_total"]
                
                if candidate_score > best_candidate_score:
                    best_candidate = repaired_candidate
                    best_candidate_score = candidate_score
            
            # Accept or reject move
            if best_candidate is not None:
                current_vector = best_candidate
                current_score = best_candidate_score
                history["improvements"] += 1
                
                # Update global best
                if current_score > best_score:
                    best_vector = current_vector.copy()
                    best_score = current_score
                    best_channels = self.parity_channels.extract_channels(best_vector)
            
            # Record history
            history["scores"].append(current_score)
            history["vectors"].append(current_vector.copy())
            
            # Convergence check
            if len(history["scores"]) > 10:
                recent_improvement = max(history["scores"][-10:]) - min(history["scores"][-10:])
                if recent_improvement < convergence_threshold:
                    print(f"MORSR converged at iteration {iteration}")
                    break
            
            # Adapt pulse size
            current_pulse_size *= self.exploration_decay
        
        print(f"MORSR completed: {history['improvements']} improvements, {history['repairs']} repairs")
        print(f"Final score: {best_score:.6f}")
        
        return best_vector, best_channels, best_score
    
    def _generate_candidates(self, 
                           current_vector: np.ndarray,
                           pulse_size: float,
                           reference_channels: Dict[str, float]) -> List[np.ndarray]:
        """Generate candidate moves for exploration."""
        candidates = []
        
        # Random perturbations
        for _ in range(5):
            perturbation = np.random.normal(0, pulse_size, 8)
            candidate = current_vector + perturbation
            candidates.append(candidate)
        
        # Gradient-based move
        try:
            direction, _ = self.objective_function.suggest_improvement_direction(
                current_vector, reference_channels
            )
            gradient_candidate = current_vector + pulse_size * direction
            candidates.append(gradient_candidate)
        except:
            pass  # Skip if gradient calculation fails
        
        # Parity-guided moves
        current_channels = self.parity_channels.extract_channels(current_vector)
        parity_candidate = self.parity_channels.enforce_parity(
            current_vector, reference_channels
        )
        candidates.append(parity_candidate)
        
        # Chamber-aware moves
        try:
            chamber_candidate = self._chamber_guided_move(current_vector, pulse_size)
            candidates.append(chamber_candidate)
        except:
            pass
        
        return candidates
    
    def _chamber_guided_move(self, vector: np.ndarray, pulse_size: float) -> np.ndarray:
        """Generate move that respects Weyl chamber structure."""
        # Move toward fundamental chamber
        projected = self.objective_function.e8_lattice.project_to_chamber(vector)
        
        # Add small random perturbation
        perturbation = np.random.normal(0, pulse_size * 0.5, 8)
        
        return projected + perturbation
    
    def _triadic_repair(self, 
                       vector: np.ndarray,
                       reference_channels: Dict[str, float],
                       max_repair_iterations: int = 3) -> np.ndarray:
        """Apply triadic repair mechanism to maintain parity constraints."""
        repaired = vector.copy()
        
        for repair_iteration in range(max_repair_iterations):
            # Check parity violations
            current_channels = self.parity_channels.extract_channels(repaired)
            
            violation_score = 0
            for channel_name, ref_value in reference_channels.items():
                if channel_name in current_channels:
                    violation = abs(current_channels[channel_name] - ref_value)
                    violation_score += violation
            
            if violation_score < self.repair_threshold:
                break  # Repair successful
            
            # Apply repair
            repair_strength = self.parity_enforcement_strength / (repair_iteration + 1)
            repaired = self.parity_channels.enforce_parity(
                repaired, reference_channels
            )
            
            # Add small stabilization
            repaired = 0.9 * repaired + 0.1 * vector  # Maintain connection to original
        
        return repaired
    
    def pulse_exploration(self,
                         vector: np.ndarray,
                         reference_channels: Dict[str, float],
                         pulse_count: int = 10,
                         domain_context: Optional[Dict] = None) -> List[Tuple[np.ndarray, float]]:
        """Execute multiple pulse explorations and return ranked results."""
        
        results = []
        
        for pulse in range(pulse_count):
            # Vary pulse size for each exploration
            pulse_size = self.pulse_size * (0.5 + random.random())
            
            # Generate candidate
            perturbation = np.random.normal(0, pulse_size, 8)
            candidate = vector + perturbation
            
            # Apply repair
            repaired_candidate = self._triadic_repair(candidate, reference_channels)
            
            # Evaluate
            score = self.objective_function.evaluate(
                repaired_candidate, reference_channels, domain_context
            )["phi_total"]
            
            results.append((repaired_candidate, score))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def set_parameters(self, 
                      pulse_size: Optional[float] = None,
                      repair_threshold: Optional[float] = None,
                      exploration_decay: Optional[float] = None,
                      parity_enforcement_strength: Optional[float] = None):
        """Update MORSR parameters."""
        
        if pulse_size is not None:
            self.pulse_size = pulse_size
        if repair_threshold is not None:
            self.repair_threshold = repair_threshold
        if exploration_decay is not None:
            self.exploration_decay = exploration_decay
        if parity_enforcement_strength is not None:
            self.parity_enforcement_strength = parity_enforcement_strength
    
    def exploration_statistics(self, history: Dict) -> Dict[str, float]:
        """Calculate statistics from exploration history."""
        scores = history.get("scores", [])
        
        if not scores:
            return {}
        
        return {
            "initial_score": scores[0],
            "final_score": scores[-1],
            "max_score": max(scores),
            "improvement": scores[-1] - scores[0],
            "max_improvement": max(scores) - scores[0],
            "convergence_iterations": len(scores),
            "improvement_rate": history.get("improvements", 0) / len(scores)
        }
'''

with open("cqe_system/morsr_explorer.py", 'w') as f:
    f.write(morsr_explorer_code)

print("Created: cqe_system/morsr_explorer.py")