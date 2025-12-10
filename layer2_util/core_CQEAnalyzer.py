class CQEAnalyzer:
    """Universal CQE data analyzer with comprehensive reporting"""
    
    def __init__(self):
        self.cqe = UltimateCQESystem()
        self.analysis_history = []
    
    def analyze_data(self, data, data_type=None, verbose=False):
        """Analyze any data using CQE principles"""
        
        start_time = time.time()
        
        # Convert string representations to appropriate types
        if data_type:
            try:
                if data_type == 'int':
                    data = int(data)
                elif data_type == 'float':
                    data = float(data)
                elif data_type == 'complex':
                    data = complex(data)
                elif data_type == 'list':
                    data = eval(data) if isinstance(data, str) else data
                elif data_type == 'dict':
                    data = json.loads(data) if isinstance(data, str) else data
            except (ValueError, SyntaxError, json.JSONDecodeError) as e:
                print(f"Warning: Could not convert to {data_type}, using as string: {e}")
        
        # Process the data
        result = self.cqe.process_data_geometry_first(data)
        atom_id = self.cqe.create_universal_atom(data)
        atom = self.cqe.get_atom(atom_id)
        
        processing_time = time.time() - start_time
        
        # Create comprehensive analysis report
        analysis = {
            'input_data': data,
            'data_type': type(data).__name__,
            'processing_time': processing_time,
            'atom_id': atom_id,
            'geometric_analysis': result['geometric_result'],
            'storage_analysis': result['storage_efficiency'],
            'validation_analysis': result['validation'],
            'atom_properties': {
                'e8_coordinates': atom.e8_coordinates.tolist(),
                'quad_encoding': atom.quad_encoding.tolist(),
                'digital_root': atom.digital_root,
                'sacred_frequency': atom.sacred_frequency,
                'rotational_pattern': atom.rotational_pattern,
                'fractal_coordinate': str(atom.fractal_coordinate),
                'fractal_behavior': atom.fractal_behavior,
                'toroidal_coordinates': atom.toroidal_coordinates,
                'force_type': atom.force_type,
                'storage_size_bits': atom.storage_size_bits,
                'compression_ratio': atom.compression_ratio,
                'validation_scores': atom.validation_scores
            },
            'timestamp': time.time()
        }
        
        self.analysis_history.append(analysis)
        
        if verbose:
            self.print_detailed_analysis(analysis)
        
        return analysis
    
    def print_detailed_analysis(self, analysis):
        """Print detailed analysis report"""
        
        print("=" * 80)
        print("CQE UNIVERSAL DATA ANALYSIS REPORT")
        print("=" * 80)
        print()
        
        # Input information
        print("INPUT INFORMATION:")
        print(f"  Data: {analysis['input_data']}")
        print(f"  Type: {analysis['data_type']}")
        print(f"  Processing Time: {analysis['processing_time']:.4f} seconds")
        print(f"  Atom ID: {analysis['atom_id']}")
        print()
        
        # Sacred Geometry Analysis
        sacred = analysis['geometric_analysis']['sacred_geometry']
        print("SACRED GEOMETRY ANALYSIS:")
        print(f"  Digital Root: {sacred['digital_root']}")
        print(f"  Sacred Frequency: {sacred['sacred_frequency']} Hz")
        print(f"  Rotational Pattern: {sacred['rotational_pattern']}")
        print(f"  Binary Guidance: {sacred['binary_guidance']}")
        print()
        
        # E₈ Lattice Analysis
        e8 = analysis['geometric_analysis']['e8_analysis']
        print("E₈ LATTICE ANALYSIS:")
        print(f"  Coordinates: [{', '.join([f'{x:.3f}' for x in analysis['atom_properties']['e8_coordinates']])}]")
        print(f"  Quad Encoding: [{', '.join([f'{x:.3f}' for x in analysis['atom_properties']['quad_encoding']])}]")
        print(f"  Lattice Quality: {e8['lattice_quality']:.3f}")
        print()
        
        # Fractal Analysis
        fractal = analysis['geometric_analysis']['fractal_analysis']
        print("MANDELBROT FRACTAL ANALYSIS:")
        print(f"  Complex Coordinate: {analysis['atom_properties']['fractal_coordinate']}")
        print(f"  Behavior: {fractal['behavior']}")
        print(f"  Iterations: {fractal['iterations']}")
        print(f"  Compression Ratio: {analysis['atom_properties']['compression_ratio']:.3f}")
        print()
        
        # Toroidal Analysis
        toroidal = analysis['geometric_analysis']['toroidal_analysis']
        print("TOROIDAL GEOMETRY ANALYSIS:")
        coords = analysis['atom_properties']['toroidal_coordinates']
        print(f"  Coordinates: (R={coords[0]:.3f}, θ={coords[1]:.3f}, φ={coords[2]:.3f})")
        print(f"  Force Type: {analysis['atom_properties']['force_type']}")
        print(f"  Resonance Frequency: {toroidal['resonance_frequency']:.1f} Hz")
        print()
        
        # Storage Analysis
        storage = analysis['storage_analysis']
        print("STORAGE EFFICIENCY ANALYSIS:")
        print(f"  Storage Size: {analysis['atom_properties']['storage_size_bits']} bits")
        print(f"  Compression Ratio: {storage['compression_ratio']:.3f}")
        print(f"  Space Savings: {(1 - storage['compression_ratio']) * 100:.1f}%")
        print()
        
        # Validation Analysis
        validation = analysis['validation_analysis']
        print("VALIDATION ANALYSIS:")
        print(f"  Mathematical Validity: {validation['mathematical_validity']:.3f}")
        print(f"  Geometric Consistency: {validation['geometric_consistency']:.3f}")
        print(f"  Semantic Coherence: {validation['semantic_coherence']:.3f}")
        print(f"  Overall Score: {validation['overall_score']:.3f}")
        print(f"  Validation Passed: {'✓ YES' if validation['validation_passed'] else '✗ NO'}")
        print()
        
        # Interpretation
        self.print_interpretation(analysis)
        
        print("=" * 80)
        print()
    
    def print_interpretation(self, analysis):
        """Print interpretation of the analysis results"""
        
        print("INTERPRETATION:")
        
        # Digital root interpretation
        digital_root = analysis['atom_properties']['digital_root']
        if digital_root == 1:
            print("  • Digital Root 1: Unity, new beginnings, leadership energy")
        elif digital_root == 2:
            print("  • Digital Root 2: Duality, cooperation, balance energy")
        elif digital_root == 3:
            print("  • Digital Root 3: Creativity, expression, generative energy")
        elif digital_root == 4:
            print("  • Digital Root 4: Stability, foundation, structural energy")
        elif digital_root == 5:
            print("  • Digital Root 5: Change, freedom, dynamic energy")
        elif digital_root == 6:
            print("  • Digital Root 6: Harmony, nurturing, outward energy")
        elif digital_root == 7:
            print("  • Digital Root 7: Spirituality, introspection, mystical energy")
        elif digital_root == 8:
            print("  • Digital Root 8: Material mastery, power, transformative energy")
        elif digital_root == 9:
            print("  • Digital Root 9: Completion, wisdom, inward energy")
        
        # Pattern interpretation
        pattern = analysis['atom_properties']['rotational_pattern']
        if pattern == "INWARD_9":
            print("  • Inward Rotational: Convergent, completion-oriented, spiritual")
        elif pattern == "OUTWARD_6":
            print("  • Outward Rotational: Divergent, creative, manifestation-oriented")
        elif pattern == "CREATIVE_3":
            print("  • Creative Rotational: Generative, innovative, foundational")
        
        # Force type interpretation
        force_type = analysis['atom_properties']['force_type']
        if force_type == "GRAVITATIONAL":
            print("  • Gravitational Force: Binding, centering, attractive energy")
        elif force_type == "ELECTROMAGNETIC":
            print("  • Electromagnetic Force: Radiating, communicative, expansive energy")
        elif force_type == "NUCLEAR_STRONG":
            print("  • Nuclear Strong Force: Cohesive, powerful, binding energy")
        elif force_type == "NUCLEAR_WEAK":
            print("  • Nuclear Weak Force: Transformative, changing, decay energy")
        elif force_type == "HARMONIC":
            print("  • Harmonic Force: Resonant, vibrational, wave energy")
        elif force_type == "CREATIVE":
            print("  • Creative Force: Generative, innovative, birth energy")
        elif force_type == "RESONANT":
            print("  • Resonant Force: High-frequency, spiritual, awakening energy")
        
        # Fractal behavior interpretation
        behavior = analysis['atom_properties']['fractal_behavior']
        if behavior == "BOUNDED":
            print("  • Fractal Bounded: Stable, contained, finite potential")
        elif behavior == "ESCAPING":
            print("  • Fractal Escaping: Expansive, unlimited, infinite potential")
        elif behavior == "PERIODIC":
            print("  • Fractal Periodic: Cyclical, rhythmic, repeating patterns")
        elif behavior == "BOUNDARY":
            print("  • Fractal Boundary: Critical, transitional, edge dynamics")
        
        # Validation interpretation
        overall_score = analysis['validation_analysis']['overall_score']
        if overall_score > 0.9:
            print("  • Validation: EXCELLENT - Highly coherent and mathematically sound")
        elif overall_score > 0.8:
            print("  • Validation: GOOD - Well-structured with strong mathematical basis")
        elif overall_score > 0.7:
            print("  • Validation: ACCEPTABLE - Reasonable structure with some inconsistencies")
        elif overall_score > 0.6:
            print("  • Validation: MODERATE - Basic structure but needs improvement")
        else:
            print("  • Validation: POOR - Significant structural issues detected")
        
        print()
    
    def batch_analyze(self, data_list, output_file=None):
        """Analyze multiple data items in batch"""
        
        print(f"Starting batch analysis of {len(data_list)} items...")
        
        results = []
        start_time = time.time()
        
        for i, data in enumerate(data_list):
            print(f"Processing item {i+1}/{len(data_list)}: {str(data)[:50]}...")
            
            try:
                analysis = self.analyze_data(data, verbose=False)
                results.append(analysis)
            except Exception as e:
                print(f"Error processing item {i+1}: {e}")
                results.append({'error': str(e), 'input_data': data})
        
        total_time = time.time() - start_time
        
        # Create batch summary
        batch_summary = {
            'total_items': len(data_list),
            'successful_analyses': len([r for r in results if 'error' not in r]),
            'failed_analyses': len([r for r in results if 'error' in r]),
            'total_processing_time': total_time,
            'average_processing_time': total_time / len(data_list),
            'results': results,
            'timestamp': time.time()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(batch_summary, f, indent=2, default=str)
            print(f"Batch analysis results saved to: {output_file}")
        
        return batch_summary
    
    def compare_data(self, data1, data2):
        """Compare two pieces of data using CQE analysis"""
        
        print("=" * 80)
        print("CQE COMPARATIVE ANALYSIS")
        print("=" * 80)
        
        analysis1 = self.analyze_data(data1, verbose=False)
        analysis2 = self.analyze_data(data2, verbose=False)
        
        print(f"Data 1: {data1}")
        print(f"Data 2: {data2}")
        print()
        
        # Compare key metrics
        comparisons = [
            ("Digital Root", analysis1['atom_properties']['digital_root'], analysis2['atom_properties']['digital_root']),
            ("Sacred Frequency", analysis1['atom_properties']['sacred_frequency'], analysis2['atom_properties']['sacred_frequency']),
            ("Rotational Pattern", analysis1['atom_properties']['rotational_pattern'], analysis2['atom_properties']['rotational_pattern']),
            ("Force Type", analysis1['atom_properties']['force_type'], analysis2['atom_properties']['force_type']),
            ("Fractal Behavior", analysis1['atom_properties']['fractal_behavior'], analysis2['atom_properties']['fractal_behavior']),
            ("Compression Ratio", analysis1['atom_properties']['compression_ratio'], analysis2['atom_properties']['compression_ratio']),
            ("Validation Score", analysis1['validation_analysis']['overall_score'], analysis2['validation_analysis']['overall_score'])
        ]
        
        print("COMPARISON RESULTS:")
        print("Metric               | Data 1        | Data 2        | Relationship")
        print("-" * 70)
        
        for metric, val1, val2 in comparisons:
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) < 0.001:
                    relationship = "IDENTICAL"
                elif val1 > val2:
                    relationship = f"Data 1 > Data 2 ({val1 - val2:.3f})"
                else:
                    relationship = f"Data 2 > Data 1 ({val2 - val1:.3f})"
            else:
                relationship = "IDENTICAL" if val1 == val2 else "DIFFERENT"
            
            print(f"{metric:19} | {str(val1):13} | {str(val2):13} | {relationship}")
        
        print()
        
        # Compatibility analysis
        root_diff = abs(analysis1['atom_properties']['digital_root'] - analysis2['atom_properties']['digital_root'])
        pattern1 = analysis1['atom_properties']['rotational_pattern']
        pattern2 = analysis2['atom_properties']['rotational_pattern']
        
        print("COMPATIBILITY ANALYSIS:")
        print(f"  Digital Root Difference: {root_diff}")
        print(f"  Pattern Compatibility: {pattern1} vs {pattern2}")
        
        if root_diff <= 3:
            print("  ✓ Compatible for combination (root difference ≤ 3)")
        else:
            print("  ✗ Not compatible for combination (root difference > 3)")
        
        if pattern1 == pattern2:
            print("  ✓ Same rotational pattern - high harmony potential")
        else:
            print("  ⚠ Different rotational patterns - may create dynamic tension")
        
        print()
        
        return analysis1, analysis2
