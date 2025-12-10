class EnhancedCQESystem:
    """Enhanced CQE system integrating all legacy variations."""
    
    def __init__(self, 
                 e8_embedding_path: Optional[str] = None,
                 governance_type: GovernanceType = GovernanceType.HYBRID,
                 tqf_config: Optional[TQFConfig] = None,
                 uvibs_config: Optional[UVIBSConfig] = None,
                 scene_config: Optional[SceneConfig] = None):
        
        self.governance_type = governance_type
        
        # Initialize base CQE components
        if e8_embedding_path and Path(e8_embedding_path).exists():
            self.e8_lattice = E8Lattice(e8_embedding_path)
        else:
            self.e8_lattice = None
        
        self.parity_channels = ParityChannels()
        self.domain_adapter = DomainAdapter()
        self.validation_framework = ValidationFramework()
        
        # Initialize enhanced components
        self.tqf_encoder = TQFEncoder(tqf_config or TQFConfig())
        self.uvibs_projector = UVIBSProjector(uvibs_config or UVIBSConfig())
        self.scene_debugger = SceneDebugger(scene_config or SceneConfig())
        
        # Initialize objective function if E8 lattice is available
        if self.e8_lattice:
            self.objective_function = CQEObjectiveFunction(self.e8_lattice, self.parity_channels)
            self.morsr_explorer = MORSRExplorer(self.objective_function, self.parity_channels)
        else:
            self.objective_function = None
            self.morsr_explorer = None
    
    def solve_problem_enhanced(self, problem: Dict[str, Any], 
                              domain_type: str = "computational",
                              governance_type: Optional[GovernanceType] = None) -> Dict[str, Any]:
        """Solve problem using enhanced CQE system with multiple governance options."""
        
        governance = governance_type or self.governance_type
        
        # Step 1: Domain embedding with governance
        if governance == GovernanceType.TQF:
            vector = self._embed_with_tqf_governance(problem, domain_type)
        elif governance == GovernanceType.UVIBS:
            vector = self._embed_with_uvibs_governance(problem, domain_type)
        elif governance == GovernanceType.HYBRID:
            vector = self._embed_with_hybrid_governance(problem, domain_type)
        else:
            vector = self.domain_adapter.embed_problem(problem, domain_type)
        
        # Step 2: Multi-window validation
        window_results = self._validate_multiple_windows(vector)
        
        # Step 3: Enhanced exploration
        if self.morsr_explorer:
            exploration_results = self._enhanced_exploration(vector, governance)
        else:
            exploration_results = {"optimal_vector": vector, "optimal_score": 0.5}
        
        # Step 4: Scene-based debugging
        scene_analysis = self._scene_based_analysis(exploration_results["optimal_vector"])
        
        # Step 5: Comprehensive validation
        validation_results = self._enhanced_validation(
            problem, exploration_results["optimal_vector"], scene_analysis
        )
        
        return {
            "problem": problem,
            "domain_type": domain_type,
            "governance_type": governance.value,
            "initial_vector": vector,
            "optimal_vector": exploration_results["optimal_vector"],
            "objective_score": exploration_results["optimal_score"],
            "window_validation": window_results,
            "scene_analysis": scene_analysis,
            "validation": validation_results,
            "recommendations": self._generate_enhanced_recommendations(validation_results)
        }
    
    def _embed_with_tqf_governance(self, problem: Dict[str, Any], domain_type: str) -> np.ndarray:
        """Embed problem with TQF governance."""
        base_vector = self.domain_adapter.embed_problem(problem, domain_type)
        
        # Apply TQF encoding
        quaternary = self.tqf_encoder.encode_quaternary(base_vector)
        orbit = self.tqf_encoder.orbit4_closure(quaternary[:4])  # Use first 4 elements
        
        # Find best lawful variant
        best_variant = None
        best_score = -1
        
        for variant_name, variant in orbit.items():
            if self.tqf_encoder.check_alt_lawful(variant):
                e_scalars = self.tqf_encoder.compute_e_scalars(variant, orbit)
                score = e_scalars["E8"]
                if score > best_score:
                    best_score = score
                    best_variant = variant
        
        if best_variant is not None:
            # Decode back to 8D
            extended = np.pad(best_variant, (0, 4), mode='constant')
            return self.tqf_encoder.decode_quaternary(extended)
        
        return base_vector
    
    def _embed_with_uvibs_governance(self, problem: Dict[str, Any], domain_type: str) -> np.ndarray:
        """Embed problem with UVIBS governance."""
        base_vector = self.domain_adapter.embed_problem(problem, domain_type)
        
        # Project to 80D
        vector_80d = self.uvibs_projector.project_80d(base_vector)
        
        # Check windows and apply corrections
        if not self.uvibs_projector.check_w80(vector_80d):
            # Simple correction: adjust sum to satisfy octadic neutrality
            current_sum = np.sum(vector_80d)
            target_adjustment = -(current_sum % 8)
            vector_80d[0] += target_adjustment / 8  # Distribute adjustment
        
        # Return to 8D (take first 8 components)
        return vector_80d[:8]
    
    def _embed_with_hybrid_governance(self, problem: Dict[str, Any], domain_type: str) -> np.ndarray:
        """Embed problem with hybrid governance combining multiple approaches."""
        base_vector = self.domain_adapter.embed_problem(problem, domain_type)
        
        # Try TQF first
        tqf_vector = self._embed_with_tqf_governance(problem, domain_type)
        
        # Try UVIBS
        uvibs_vector = self._embed_with_uvibs_governance(problem, domain_type)
        
        # Combine using weighted average
        alpha = 0.6  # Weight for TQF
        beta = 0.4   # Weight for UVIBS
        
        hybrid_vector = alpha * tqf_vector + beta * uvibs_vector
        
        return hybrid_vector
    
    def _validate_multiple_windows(self, vector: np.ndarray) -> Dict[str, bool]:
        """Validate vector against multiple window types."""
        results = {}
        
        # W4 window (parity)
        results["W4"] = (np.sum(vector) % 4) == 0
        
        # TQF lawful check
        quaternary = np.clip(vector * 3 + 1, 1, 4).astype(int)
        results["TQF_LAWFUL"] = self.tqf_encoder.check_alt_lawful(quaternary)
        
        # UVIBS W80 check (simplified for 8D)
        quad_form = np.sum(vector * vector)
        results["W80_SIMPLIFIED"] = (quad_form % 4) == 0 and (np.sum(vector) % 8) == 0
        
        return results
    
    def _enhanced_exploration(self, vector: np.ndarray, governance: GovernanceType) -> Dict[str, Any]:
        """Enhanced exploration using multiple strategies."""
        if not self.morsr_explorer:
            return {"optimal_vector": vector, "optimal_score": 0.5}
        
        # Standard MORSR exploration
        reference_channels = {"channel_1": 0.5, "channel_2": 0.3}
        optimal_vector, optimal_channels, optimal_score = self.morsr_explorer.explore(
            vector, reference_channels, max_iterations=50
        )
        
        # Apply governance-specific enhancements
        if governance == GovernanceType.TQF:
            # Apply TQF resonant gates
            orbit = self.tqf_encoder.orbit4_closure(np.clip(optimal_vector * 3 + 1, 1, 4).astype(int))
            e_scalars = self.tqf_encoder.compute_e_scalars(optimal_vector, orbit)
            optimal_score *= e_scalars["E8"]
        
        elif governance == GovernanceType.UVIBS:
            # Apply UVIBS governance check
            if self.uvibs_projector.monster_governance_check(optimal_vector):
                optimal_score *= 1.2  # Bonus for governance compliance
        
        return {
            "optimal_vector": optimal_vector,
            "optimal_channels": optimal_channels,
            "optimal_score": optimal_score
        }
    
    def _scene_based_analysis(self, vector: np.ndarray) -> Dict[str, Any]:
        """Perform scene-based debugging analysis."""
        # Create 8×8 viewer
        viewer = self.scene_debugger.create_8x8_viewer(vector)
        
        # Shell analysis
        shell_analysis = self.scene_debugger.create_shell_analysis(vector, viewer["hot_zones"])
        
        # Parity twin check (if hot zones exist)
        parity_results = {}
        if viewer["hot_zones"]:
            # Create a modified grid (simple perturbation)
            modified_grid = viewer["grid"] + np.random.normal(0, 0.01, viewer["grid"].shape)
            parity_results = self.scene_debugger.parity_twin_check(viewer["grid"], modified_grid)
        
        return {
            "viewer": viewer,
            "shell_analysis": shell_analysis,
            "parity_twin": parity_results
        }
    
    def _enhanced_validation(self, problem: Dict[str, Any], vector: np.ndarray, 
                           scene_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation incorporating scene analysis."""
        # Base validation
        mock_analysis = {
            "embedding_quality": {"optimal": {"nearest_root_distance": 0.5}},
            "objective_breakdown": {"phi_total": 0.7},
            "chamber_analysis": {"optimal_chamber": "11111111"},
            "geometric_metrics": {"convergence_quality": "good"}
        }
        
        base_validation = self.validation_framework.validate_solution(problem, vector, mock_analysis)
        
        # Enhanced validation with scene analysis
        scene_score = 1.0
        if scene_analysis["viewer"]["hot_zones"]:
            scene_score *= 0.8  # Penalty for hot zones
        
        if scene_analysis["parity_twin"] and scene_analysis["parity_twin"].get("hinged", False):
            scene_score *= 1.1  # Bonus for hinged repairs
        
        base_validation["scene_score"] = scene_score
        base_validation["overall_score"] *= scene_score
        
        return base_validation
    
    def _generate_enhanced_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations based on validation results."""
        recommendations = []
        
        if validation_results["overall_score"] < 0.7:
            recommendations.append("Consider using hybrid governance for better performance")
        
        if validation_results.get("scene_score", 1.0) < 0.9:
            recommendations.append("Apply scene-based debugging to identify hot zones")
        
        if "TQF_LAWFUL" in validation_results and not validation_results["TQF_LAWFUL"]:
            recommendations.append("Use TQF governance to ensure lawful state transitions")
        
        recommendations.append("Monitor E-scalar metrics for continuous improvement")
        
        return recommendations

# Factory function for easy instantiation