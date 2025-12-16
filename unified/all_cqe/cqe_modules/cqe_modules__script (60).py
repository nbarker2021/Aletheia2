# Enhanced MORSR Explorer with complete E8 lattice node traversal

enhanced_morsr_code = '''"""
Enhanced MORSR Explorer - Complete E₈ Lattice Node Traversal

Modified MORSR algorithm that systematically visits ALL 240 E₈ root nodes
exactly once per task, logging comprehensive overlay data and making
determinations based on complete lattice information.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Set, Any
import logging
import time
from pathlib import Path

from .objective_function import CQEObjectiveFunction
from .parity_channels import ParityChannels

class CompleteMORSRExplorer:
    """
    Enhanced MORSR with complete E₈ lattice traversal.
    
    Visits ALL 240 lattice nodes exactly once per exploration task,
    logging comprehensive overlay data for complete problem analysis.
    """
    
    def __init__(self, 
                 objective_function: CQEObjectiveFunction,
                 parity_channels: ParityChannels,
                 random_seed: Optional[int] = None,
                 enable_detailed_logging: bool = True):
        
        self.objective_function = objective_function
        self.parity_channels = parity_channels
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Enhanced parameters for complete traversal
        self.enable_detailed_logging = enable_detailed_logging
        self.setup_logging()
        
        # Complete lattice analysis state
        self.complete_traversal_data = {}
        self.node_visit_order = []
        self.overlay_analytics = {}
        
        # E₈ lattice access
        self.e8_lattice = objective_function.e8_lattice
        self.all_roots = self.e8_lattice.roots  # 240×8 array
        
        self.logger.info("CompleteMORSRExplorer initialized for full lattice traversal")
    
    def setup_logging(self):
        """Setup comprehensive logging for complete traversal."""
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("CompleteMORSR")
        self.logger.setLevel(logging.INFO if self.enable_detailed_logging else logging.WARNING)
        
        # File handler for detailed logs
        log_file = Path("logs") / f"complete_morsr_{int(time.time())}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler for key events
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized: {log_file}")
    
    def complete_lattice_exploration(self,
                                   initial_vector: np.ndarray,
                                   reference_channels: Dict[str, float],
                                   domain_context: Optional[Dict] = None,
                                   traversal_strategy: str = "systematic") -> Dict[str, Any]:
        """
        Execute complete E₈ lattice traversal touching all 240 nodes.
        
        Args:
            initial_vector: Starting 8D vector
            reference_channels: Target parity channels
            domain_context: Problem domain information
            traversal_strategy: "systematic", "distance_ordered", or "chamber_guided"
            
        Returns:
            Complete overlay analysis with all node data
        """
        
        self.logger.info("Starting complete E₈ lattice traversal")
        self.logger.info(f"Traversal strategy: {traversal_strategy}")
        self.logger.info(f"Initial vector: {initial_vector}")
        self.logger.info(f"Domain context: {domain_context}")
        
        start_time = time.time()
        
        # Initialize traversal data structures
        self.complete_traversal_data = {}
        self.node_visit_order = []
        self.overlay_analytics = {
            "node_scores": {},
            "chamber_distribution": {},
            "parity_variations": {},
            "geometric_properties": {},
            "domain_insights": {}
        }
        
        # Determine traversal order
        traversal_order = self._determine_traversal_order(
            initial_vector, traversal_strategy
        )
        
        # Execute complete traversal
        best_node_idx = -1
        best_score = -np.inf
        best_vector = initial_vector.copy()
        best_channels = reference_channels.copy()
        
        for step, node_idx in enumerate(traversal_order):
            node_data = self._analyze_lattice_node(
                node_idx, initial_vector, reference_channels, domain_context, step
            )
            
            # Update best solution
            if node_data["objective_score"] > best_score:
                best_score = node_data["objective_score"]
                best_node_idx = node_idx
                best_vector = node_data["projected_vector"]
                best_channels = node_data["channels"]
            
            # Log progress
            if step % 50 == 0:
                self.logger.info(f"Progress: {step}/240 nodes analyzed")
                self.logger.info(f"Current best score: {best_score:.6f} at node {best_node_idx}")
        
        # Generate comprehensive overlay analysis
        total_time = time.time() - start_time
        overlay_analysis = self._generate_complete_overlay_analysis(
            initial_vector, best_vector, best_channels, best_score, 
            best_node_idx, total_time, domain_context
        )
        
        self.logger.info("Complete lattice traversal finished")
        self.logger.info(f"Total time: {total_time:.3f}s")
        self.logger.info(f"Best solution: node {best_node_idx}, score {best_score:.6f}")
        
        # Save complete data
        self._save_complete_traversal_data(overlay_analysis)
        
        return overlay_analysis
    
    def _determine_traversal_order(self, 
                                 initial_vector: np.ndarray, 
                                 strategy: str) -> List[int]:
        """Determine order for visiting all 240 lattice nodes."""
        
        if strategy == "systematic":
            # Simple sequential order
            return list(range(240))
        
        elif strategy == "distance_ordered":
            # Order by distance from initial vector
            distances = []
            for i in range(240):
                dist = np.linalg.norm(self.all_roots[i] - initial_vector)
                distances.append((dist, i))
            
            distances.sort()  # Closest first
            return [idx for _, idx in distances]
        
        elif strategy == "chamber_guided":
            # Order by Weyl chamber, then by distance within chamber
            chamber_groups = {}
            
            for i in range(240):
                chamber_sig, _ = self.e8_lattice.determine_chamber(self.all_roots[i])
                if chamber_sig not in chamber_groups:
                    chamber_groups[chamber_sig] = []
                chamber_groups[chamber_sig].append(i)
            
            # Order chambers and nodes within chambers
            ordered_nodes = []
            for chamber_sig in sorted(chamber_groups.keys()):
                nodes_in_chamber = chamber_groups[chamber_sig]
                
                # Sort by distance from initial vector within chamber
                chamber_distances = []
                for node_idx in nodes_in_chamber:
                    dist = np.linalg.norm(self.all_roots[node_idx] - initial_vector)
                    chamber_distances.append((dist, node_idx))
                
                chamber_distances.sort()
                ordered_nodes.extend([idx for _, idx in chamber_distances])
            
            return ordered_nodes
        
        else:
            # Fallback to systematic
            return list(range(240))
    
    def _analyze_lattice_node(self,
                            node_idx: int,
                            initial_vector: np.ndarray,
                            reference_channels: Dict[str, float],
                            domain_context: Optional[Dict],
                            step: int) -> Dict[str, Any]:
        """Complete analysis of a single lattice node."""
        
        root_vector = self.all_roots[node_idx]
        
        # Project initial vector toward root
        projection_weight = 0.3  # Blend with root
        projected_vector = (1 - projection_weight) * initial_vector + projection_weight * root_vector
        
        # Extract channels from projected vector
        channels = self.parity_channels.extract_channels(projected_vector)
        
        # Evaluate objective function
        scores = self.objective_function.evaluate(
            projected_vector, reference_channels, domain_context
        )
        
        # Chamber analysis
        chamber_sig, inner_prods = self.e8_lattice.determine_chamber(projected_vector)
        
        # Geometric properties
        distance_to_initial = np.linalg.norm(projected_vector - initial_vector)
        distance_to_root = np.linalg.norm(projected_vector - root_vector)
        root_norm = np.linalg.norm(root_vector)
        
        # Node analysis data
        node_data = {
            "node_index": node_idx,
            "step": step,
            "root_vector": root_vector.tolist(),
            "projected_vector": projected_vector.tolist(),
            "channels": channels,
            "objective_score": scores["phi_total"],
            "score_breakdown": scores,
            "chamber_signature": chamber_sig,
            "chamber_inner_products": inner_prods.tolist(),
            "geometric_properties": {
                "distance_to_initial": distance_to_initial,
                "distance_to_root": distance_to_root,
                "root_norm": root_norm,
                "projection_quality": 1.0 / (1.0 + distance_to_root)
            }
        }
        
        # Store in complete traversal data
        self.complete_traversal_data[node_idx] = node_data
        self.node_visit_order.append(node_idx)
        
        # Update overlay analytics
        self._update_overlay_analytics(node_data, domain_context)
        
        # Detailed logging
        self.logger.debug(f"Node {node_idx}: score={scores['phi_total']:.4f}, "
                         f"chamber={chamber_sig}, dist_to_root={distance_to_root:.4f}")
        
        return node_data
    
    def _update_overlay_analytics(self, 
                                node_data: Dict[str, Any], 
                                domain_context: Optional[Dict]):
        """Update running analytics with node data."""
        
        node_idx = node_data["node_index"]
        score = node_data["objective_score"]
        chamber_sig = node_data["chamber_signature"]
        
        # Node scores
        self.overlay_analytics["node_scores"][node_idx] = score
        
        # Chamber distribution
        if chamber_sig not in self.overlay_analytics["chamber_distribution"]:
            self.overlay_analytics["chamber_distribution"][chamber_sig] = []
        self.overlay_analytics["chamber_distribution"][chamber_sig].append(node_idx)
        
        # Parity variations
        channels = node_data["channels"]
        for channel_name, value in channels.items():
            if channel_name not in self.overlay_analytics["parity_variations"]:
                self.overlay_analytics["parity_variations"][channel_name] = []
            self.overlay_analytics["parity_variations"][channel_name].append(value)
        
        # Geometric properties
        geom_props = node_data["geometric_properties"]
        for prop_name, value in geom_props.items():
            if prop_name not in self.overlay_analytics["geometric_properties"]:
                self.overlay_analytics["geometric_properties"][prop_name] = []
            self.overlay_analytics["geometric_properties"][prop_name].append(value)
        
        # Domain-specific insights
        if domain_context:
            domain_type = domain_context.get("domain_type", "unknown")
            if domain_type not in self.overlay_analytics["domain_insights"]:
                self.overlay_analytics["domain_insights"][domain_type] = {
                    "node_scores": [],
                    "best_nodes": [],
                    "chamber_preferences": {}
                }
            
            domain_data = self.overlay_analytics["domain_insights"][domain_type]
            domain_data["node_scores"].append(score)
            
            # Track best nodes for this domain
            if len(domain_data["best_nodes"]) < 10:
                domain_data["best_nodes"].append((score, node_idx))
                domain_data["best_nodes"].sort(reverse=True)  # Best first
            elif score > domain_data["best_nodes"][-1][0]:
                domain_data["best_nodes"][-1] = (score, node_idx)
                domain_data["best_nodes"].sort(reverse=True)
            
            # Chamber preferences by domain
            if chamber_sig not in domain_data["chamber_preferences"]:
                domain_data["chamber_preferences"][chamber_sig] = []
            domain_data["chamber_preferences"][chamber_sig].append(score)
    
    def _generate_complete_overlay_analysis(self,
                                          initial_vector: np.ndarray,
                                          best_vector: np.ndarray,
                                          best_channels: Dict[str, float],
                                          best_score: float,
                                          best_node_idx: int,
                                          total_time: float,
                                          domain_context: Optional[Dict]) -> Dict[str, Any]:
        """Generate comprehensive overlay analysis from complete traversal."""
        
        # Statistical summaries
        all_scores = list(self.overlay_analytics["node_scores"].values())
        
        score_stats = {
            "mean": np.mean(all_scores),
            "std": np.std(all_scores),
            "min": np.min(all_scores),
            "max": np.max(all_scores),
            "median": np.median(all_scores),
            "best_score": best_score,
            "best_node": best_node_idx
        }
        
        # Chamber analysis
        chamber_stats = {}
        for chamber_sig, node_list in self.overlay_analytics["chamber_distribution"].items():
            chamber_scores = [self.overlay_analytics["node_scores"][idx] for idx in node_list]
            chamber_stats[chamber_sig] = {
                "node_count": len(node_list),
                "mean_score": np.mean(chamber_scores),
                "best_score": np.max(chamber_scores),
                "best_node": node_list[np.argmax(chamber_scores)]
            }
        
        # Parity analysis
        parity_stats = {}
        for channel_name, values in self.overlay_analytics["parity_variations"].items():
            parity_stats[channel_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "range": [np.min(values), np.max(values)]
            }
        
        # Geometric analysis
        geometric_stats = {}
        for prop_name, values in self.overlay_analytics["geometric_properties"].items():
            geometric_stats[prop_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "range": [np.min(values), np.max(values)]
            }
        
        # Top performing nodes
        top_nodes = sorted(
            [(score, idx) for idx, score in self.overlay_analytics["node_scores"].items()],
            reverse=True
        )[:20]  # Top 20
        
        # Complete overlay analysis
        analysis = {
            "traversal_metadata": {
                "total_nodes_visited": 240,
                "traversal_time": total_time,
                "nodes_per_second": 240 / total_time,
                "traversal_order": self.node_visit_order,
                "domain_context": domain_context
            },
            "solution": {
                "initial_vector": initial_vector.tolist(),
                "best_vector": best_vector.tolist(),
                "best_channels": best_channels,
                "best_score": best_score,
                "best_node_index": best_node_idx,
                "improvement": best_score - self.objective_function.evaluate(
                    initial_vector, best_channels, domain_context
                )["phi_total"]
            },
            "statistical_analysis": {
                "score_distribution": score_stats,
                "chamber_analysis": chamber_stats,
                "parity_analysis": parity_stats,
                "geometric_analysis": geometric_stats
            },
            "top_performing_nodes": [
                {
                    "rank": i + 1,
                    "node_index": idx,
                    "score": score,
                    "root_vector": self.all_roots[idx].tolist(),
                    "chamber": self.e8_lattice.determine_chamber(self.all_roots[idx])[0]
                }
                for i, (score, idx) in enumerate(top_nodes)
            ],
            "domain_insights": self.overlay_analytics["domain_insights"],
            "complete_node_data": self.complete_traversal_data,
            "recommendations": self._generate_recommendations_from_complete_data(
                score_stats, chamber_stats, domain_context
            )
        }
        
        return analysis
    
    def _generate_recommendations_from_complete_data(self,
                                                   score_stats: Dict,
                                                   chamber_stats: Dict,
                                                   domain_context: Optional[Dict]) -> List[str]:
        """Generate actionable recommendations based on complete traversal data."""
        
        recommendations = []
        
        # Score-based recommendations
        if score_stats["std"] > 0.2:
            recommendations.append(
                f"High score variance ({score_stats['std']:.3f}) suggests problem has "
                "distinct optimal regions - consider multi-modal optimization"
            )
        
        if score_stats["best_score"] - score_stats["mean"] > 2 * score_stats["std"]:
            recommendations.append(
                f"Best solution significantly outperforms average - "
                f"node {score_stats['best_node']} may represent optimal embedding"
            )
        
        # Chamber-based recommendations
        best_chamber = max(chamber_stats.items(), key=lambda x: x[1]["best_score"])
        recommendations.append(
            f"Chamber {best_chamber[0]} shows highest performance with "
            f"{best_chamber[1]['node_count']} nodes and best score {best_chamber[1]['best_score']:.4f}"
        )
        
        if len(chamber_stats) > 10:
            recommendations.append(
                f"Problem spans {len(chamber_stats)} chambers - "
                "consider chamber-specific optimization strategies"
            )
        
        # Domain-specific recommendations
        if domain_context:
            domain_type = domain_context.get("domain_type", "unknown")
            if domain_type in self.overlay_analytics["domain_insights"]:
                domain_data = self.overlay_analytics["domain_insights"][domain_type]
                best_domain_score = max(domain_data["node_scores"])
                
                if best_domain_score > 0.8:
                    recommendations.append(
                        f"Excellent {domain_type} problem embedding achieved "
                        f"(score: {best_domain_score:.4f})"
                    )
                elif best_domain_score < 0.5:
                    recommendations.append(
                        f"Poor {domain_type} problem embedding - "
                        "consider alternative domain adaptation strategies"
                    )
        
        return recommendations
    
    def _save_complete_traversal_data(self, analysis: Dict[str, Any]):
        """Save complete traversal data to file."""
        
        # Create data directory
        Path("data/generated").mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"complete_morsr_analysis_{timestamp}.json"
        filepath = Path("data/generated") / filename
        
        # Save analysis
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.logger.info(f"Complete analysis saved to: {filepath}")
        
        # Also save summary
        summary = {
            "timestamp": timestamp,
            "nodes_visited": 240,
            "best_score": analysis["solution"]["best_score"],
            "best_node": analysis["solution"]["best_node_index"],
            "traversal_time": analysis["traversal_metadata"]["traversal_time"],
            "recommendations": analysis["recommendations"]
        }
        
        summary_file = Path("data/generated") / f"morsr_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to: {summary_file}")

# Legacy compatibility wrapper
class MORSRExplorer:
    """Legacy compatibility wrapper for the enhanced complete traversal MORSR."""
    
    def __init__(self, objective_function, parity_channels, random_seed=None):
        self.complete_explorer = CompleteMORSRExplorer(
            objective_function, parity_channels, random_seed
        )
    
    def explore(self, 
               initial_vector: np.ndarray,
               reference_channels: Dict[str, float],
               max_iterations: int = 50,
               domain_context: Optional[Dict] = None,
               convergence_threshold: float = 1e-4) -> Tuple[np.ndarray, Dict[str, float], float]:
        """
        Legacy explore method - now performs complete lattice traversal.
        
        NOTE: max_iterations and convergence_threshold are ignored in favor of
        complete 240-node traversal for comprehensive analysis.
        """
        
        analysis = self.complete_explorer.complete_lattice_exploration(
            initial_vector, reference_channels, domain_context, "distance_ordered"
        )
        
        # Extract legacy format results
        best_vector = np.array(analysis["solution"]["best_vector"])
        best_channels = analysis["solution"]["best_channels"]
        best_score = analysis["solution"]["best_score"]
        
        return best_vector, best_channels, best_score
    
    # Delegate other methods to complete explorer
    def __getattr__(self, name):
        return getattr(self.complete_explorer, name)
'''

# Update the existing MORSR file
with open("cqe_system/morsr_explorer.py", 'w') as f:
    f.write(enhanced_morsr_code)

print("Enhanced MORSR Explorer with complete E₈ lattice traversal created")
print("✓ Visits ALL 240 lattice nodes exactly once per task")  
print("✓ Comprehensive overlay data logging and analysis")
print("✓ Makes determinations based on complete lattice information")
print("✓ Legacy compatibility maintained")