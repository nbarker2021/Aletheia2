class FireChainDemonstration:
    """Demonstration of iterative fire chain exploration."""
    
    def __init__(self):
        self.results = {}
        self.setup_complete = False
    
    def setup_systems(self):
        """Set up the fire chain demonstration."""
        print("Fire Chain Demonstration System")
        print("=" * 40)
        
        # Create mock components for demonstration
        self.mock_components = self._create_demo_components()
        
        # Initialize complete MORSR
        self.complete_morsr = CompleteMORSRExplorer(
            self.mock_components["objective_function"],
            self.mock_components["parity_channels"],
            random_seed=42
        )
        
        # Initialize fire chain explorer
        self.fire_chain_explorer = IterativeFireChainExplorer(
            self.complete_morsr,
            enable_emergent_discovery=True,
            max_fire_chains=3,  # Shorter for demo
            improvement_threshold=0.08,
            outlier_margin=2.5
        )
        
        self.setup_complete = True
        print("✓ Fire chain systems initialized\\n")
    
    def _create_demo_components(self):
        """Create demo components with realistic behavior."""
        
        class DemoE8Lattice:
            def __init__(self):
                # Create deterministic "E8" roots for consistent demo
                np.random.seed(42)
                self.roots = np.random.randn(240, 8)
                for i in range(240):
                    self.roots[i] = self.roots[i] / np.linalg.norm(self.roots[i]) * 1.4
            
            def determine_chamber(self, vector):
                chamber_sig = ''.join(['1' if v > 0 else '0' for v in vector])
                inner_prods = np.dot(vector, self.roots[:8].T)  # Use first 8 roots as simple roots
                return chamber_sig, inner_prods
        
        class DemoParityChannels:
            def extract_channels(self, vector):
                # Realistic channel extraction with some structure
                channels = {}
                for i in range(8):
                    # Add some correlation structure
                    base_val = (np.sin(vector[i] * np.pi) + 1) / 2
                    if i > 0:
                        correlation = 0.2 * channels[f"channel_{i}"]  # Correlate with previous
                        base_val = 0.8 * base_val + 0.2 * correlation
                    channels[f"channel_{i+1}"] = np.clip(base_val, 0, 1)
                return channels
        
        class DemoObjectiveFunction:
            def __init__(self):
                self.e8_lattice = DemoE8Lattice()
                np.random.seed(42)  # Consistent evaluation
                
            def evaluate(self, vector, reference_channels, domain_context=None):
                # Create realistic objective with multiple components
                
                # Base score from vector properties
                norm_penalty = abs(np.linalg.norm(vector) - 1.0) * 0.2
                base_score = 0.4 + 0.3 * np.sin(np.sum(vector)) ** 2 - norm_penalty
                
                # Parity consistency component
                current_channels = self.e8_lattice.__class__.__bases__[0].__dict__.get(
                    'parity_channels', DemoParityChannels()
                ).extract_channels(vector) if hasattr(self, 'parity_channels') else {}
                if not current_channels:
                    current_channels = DemoParityChannels().extract_channels(vector)
                
                parity_penalty = 0
                for ch_name, ref_val in reference_channels.items():
                    if ch_name in current_channels:
                        parity_penalty += abs(current_channels[ch_name] - ref_val) * 0.1
                
                parity_score = max(0, 1.0 - parity_penalty)
                
                # Domain context bonus
                domain_bonus = 0
                if domain_context:
                    complexity_class = domain_context.get("complexity_class", "unknown")
                    if complexity_class == "P":
                        domain_bonus = 0.05 if base_score > 0.6 else 0
                    elif complexity_class == "NP":
                        domain_bonus = 0.03 if base_score > 0.5 else 0
                
                # Chamber stability (prefer positive chambers)
                chamber_sig, _ = self.e8_lattice.determine_chamber(vector)
                chamber_bonus = 0.02 if chamber_sig.count('1') > 4 else 0
                
                final_score = np.clip(base_score + domain_bonus + chamber_bonus, 0.0, 1.0)
                
                return {
                    "phi_total": final_score,
                    "lattice_quality": base_score,
                    "parity_consistency": parity_score,
                    "chamber_stability": 0.5 + chamber_bonus * 10,
                    "geometric_separation": final_score * 1.1,
                    "domain_coherence": 0.5 + domain_bonus * 10
                }
        
        return {
            "objective_function": DemoObjectiveFunction(),
            "parity_channels": DemoParityChannels()
        }
    
    def demonstrate_fire_chains(self):
        """Demonstrate complete fire chain exploration."""
        print("🔥 FIRE CHAIN EXPLORATION DEMONSTRATION")
        print("=" * 50)
        
        if not self.setup_complete:
            self.setup_systems()
        
        # Create a challenging test case
        test_vector = np.array([0.8, -0.4, 0.6, -0.2, 0.3, -0.7, 0.5, -0.1])
        reference_channels = {f"channel_{i+1}": 0.4 + 0.2 * np.sin(i) for i in range(8)}
        domain_context = {
            "domain_type": "computational",
            "complexity_class": "NP",
            "problem_size": 200,
            "requires_breakthrough": True
        }
        
        print(f"Test vector: {test_vector}")
        print(f"Domain context: {domain_context}")
        print("Reference channels:", {k: f"{v:.3f}" for k, v in reference_channels.items()})
        
        # Execute fire chain exploration
        print("\\n🚀 Starting iterative fire chain exploration...")
        start_time = time.time()
        
        analysis = self.fire_chain_explorer.iterative_fire_chain_exploration(
            test_vector, reference_channels, domain_context
        )
        
        elapsed_time = time.time() - start_time
        
        # Display results
        self._display_fire_chain_results(analysis, elapsed_time)
        
        self.results["fire_chain_demo"] = analysis
        return analysis
    
    def _display_fire_chain_results(self, analysis: dict, elapsed_time: float):
        """Display fire chain exploration results."""
        
        print("\\n" + "=" * 60)
        print("🔥 FIRE CHAIN EXPLORATION RESULTS")
        print("=" * 60)
        
        # Summary
        summary = analysis["fire_chain_summary"]
        print(f"Total fire chains executed: {summary['total_chains']}")
        print(f"Chains with improvements: {summary['total_improvements']}")
        print(f"Final improvement magnitude: {summary['final_improvement']:.6f}")
        print(f"Convergence achieved: {summary['convergence_achieved']}")
        print(f"Total exploration time: {elapsed_time:.3f}s")
        
        # Emergent discoveries
        discoveries = analysis["emergent_discoveries"]
        print(f"\\n✨ EMERGENT DISCOVERIES:")
        print(f"Total discoveries: {discoveries['total_discoveries']}")
        print(f"Breakthrough discoveries: {len(discoveries['breakthrough_discoveries'])}")
        print(f"Unique emergence types: {discoveries['unique_emergence_types']}")
        print(f"Emergent channels discovered: {discoveries['emergent_channels_discovered']}")
        
        # Breakthrough details
        if discoveries["breakthrough_discoveries"]:
            print("\\n🚨 BREAKTHROUGH DISCOVERIES:")
            for i, discovery in enumerate(discoveries["breakthrough_discoveries"], 1):
                print(f"  {i}. {discovery['emergence_type']}")
                print(f"     Concept: {discovery['hypothesis']['concept'][:60]}...")
                print(f"     Uniqueness: {discovery['uniqueness_score']:.4f}")
        
        # Learning trajectory
        print("\\n📈 LEARNING TRAJECTORY:")
        for step in analysis["learning_trajectory"]:
            print(f"  Chain {step['iteration'] + 1}: Score {step['best_score']:.4f}, "
                  f"Discoveries {step['discoveries']}")
            if step["key_insights"]:
                for insight in step["key_insights"]:
                    print(f"    💡 {insight}")
        
        # Final recommendations
        print("\\n🎯 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    def demonstrate_emergent_discovery(self):
        """Demonstrate emergent discovery capabilities."""
        print("\\n✨ EMERGENT DISCOVERY DEMONSTRATION")
        print("=" * 45)
        
        if not self.setup_complete:
            self.setup_systems()
        
        # Create a vector that might lead to emergent behavior
        emergent_vector = np.array([0.707, 0.707, 0.0, 0.0, -0.707, -0.707, 0.0, 0.0])  # Structured pattern
        emergent_channels = {f"channel_{i+1}": 0.5 + 0.3 * np.cos(i * np.pi / 4) for i in range(8)}
        
        context = {
            "domain_type": "exploratory",
            "complexity_class": "unknown",
            "exploration_type": "emergent",
            "novelty_seeking": True
        }
        
        print("Emergent exploration vector (structured pattern):")
        print(f"  Vector: {emergent_vector}")
        print(f"  Channels: {', '.join(f'{k}={v:.3f}' for k, v in emergent_channels.items())}")
        
        # Execute with focus on emergent discovery
        fire_explorer = IterativeFireChainExplorer(
            self.complete_morsr,
            enable_emergent_discovery=True,
            max_fire_chains=4,  # More chains for emergent discovery
            improvement_threshold=0.05,  # Lower threshold
            outlier_margin=1.8  # Lower outlier threshold
        )
        
        analysis = fire_explorer.iterative_fire_chain_exploration(
            emergent_vector, emergent_channels, context
        )
        
        # Focus on emergent aspects
        discoveries = analysis["emergent_discoveries"]
        
        print(f"\\n🎊 EMERGENT DISCOVERY RESULTS:")
        print(f"Discoveries found: {discoveries['total_discoveries']}")
        
        if discoveries["breakthrough_discoveries"]:
            print(f"\\n🚀 BREAKTHROUGH PATTERNS:")
            for discovery in discoveries["breakthrough_discoveries"]:
                print(f"  • Type: {discovery['emergence_type']}")
                print(f"    Uniqueness: {discovery['uniqueness_score']:.4f}")
                print(f"    Concept: {discovery['hypothesis']['concept']}")
                
                # Show novel properties
                novel_props = [p for p in discovery['evaluation']['novel_properties'] if p]
                if novel_props:
                    print(f"    Novel properties: {', '.join(novel_props)}")
        
        print(f"\\n🔬 CONCEPTUAL EXPLORATIONS:")
        for chain in analysis["learning_trajectory"]:
            if chain["discoveries"] > 0:
                print(f"  Chain {chain['iteration'] + 1}: {chain['discoveries']} emergent patterns")
        
        self.results["emergent_demo"] = analysis
        return analysis
    
    def run_complete_demonstration(self):
        """Run complete fire chain demonstration."""
        print("Fire Chain Explorer - Complete Demonstration")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Main fire chain demonstration
            self.demonstrate_fire_chains()
            
            # Emergent discovery focus
            self.demonstrate_emergent_discovery()
            
        except Exception as e:
            print(f"\\nDemonstration error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        total_time = time.time() - start_time
        
        print("\\n" + "=" * 60)
        print("🎉 FIRE CHAIN DEMONSTRATION COMPLETE")
        print("=" * 60)
        print(f"Total demonstration time: {total_time:.2f} seconds")
        
        # Summary insights
        print("\\n💡 KEY INSIGHTS FROM DEMONSTRATION:")
        print("• Fire chains enable iterative improvement through structured exploration")
        print("• Review phase identifies patterns and learning opportunities")
        print("• Re-stance phase repositions based on accumulated knowledge") 
        print("• Emergent phase discovers novel patterns through conceptual exploration")
        print("• Outlier detection triggers expanded evaluation when needed")
        print("• System validates first-of-kind and breakthrough discoveries")
        
        # Save demonstration results
        self._save_demo_results()
        
        return True
    
    def _save_demo_results(self):
        """Save demonstration results."""
        Path("data/generated").mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        results_file = Path("data/generated") / f"fire_chain_demo_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\\nDemonstration results saved: {results_file}")
