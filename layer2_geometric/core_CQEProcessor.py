class CQEProcessor:
    """CQE-based processing engine"""
    
    def __init__(self, memory_manager: CQEMemoryManager):
        self.memory = memory_manager
        self.operation_queue = queue.PriorityQueue()
        self.result_cache = {}
        self.processing_lock = threading.RLock()
    
    def process_operation(self, operation_type: CQEOperationType, 
                         input_atoms: List[CQEAtom], 
                         parameters: Dict[str, Any] = None) -> List[CQEAtom]:
        """Process CQE operation on input atoms"""
        if parameters is None:
            parameters = {}
        
        with self.processing_lock:
            # Check cache first
            cache_key = self._compute_cache_key(operation_type, input_atoms, parameters)
            if cache_key in self.result_cache:
                return self.result_cache[cache_key]
            
            # Process based on operation type
            if operation_type == CQEOperationType.TRANSFORMATION:
                result = self._transform_atoms(input_atoms, parameters)
            elif operation_type == CQEOperationType.OPTIMIZATION:
                result = self._optimize_atoms(input_atoms, parameters)
            elif operation_type == CQEOperationType.VALIDATION:
                result = self._validate_atoms(input_atoms, parameters)
            elif operation_type == CQEOperationType.REASONING:
                result = self._reason_with_atoms(input_atoms, parameters)
            else:
                result = input_atoms  # Default: no change
            
            # Cache result
            self.result_cache[cache_key] = result
            
            # Store result atoms in memory
            for atom in result:
                self.memory.store_atom(atom)
            
            return result
    
    def _transform_atoms(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> List[CQEAtom]:
        """Transform atoms using CQE principles"""
        transformation_type = parameters.get('type', 'identity')
        result_atoms = []
        
        for atom in atoms:
            if transformation_type == 'quad_shift':
                # Shift quad encoding
                shift = parameters.get('shift', (1, 0, 0, 0))
                new_quad = tuple((q + s - 1) % 4 + 1 for q, s in zip(atom.quad_encoding, shift))
                
                new_atom = CQEAtom(
                    data=atom.data,
                    quad_encoding=new_quad,
                    parent_id=atom.id,
                    metadata={'transformation': 'quad_shift', 'original_id': atom.id}
                )
                
            elif transformation_type == 'e8_rotation':
                # Rotate in E8 space
                rotation_matrix = parameters.get('rotation_matrix', np.eye(8))
                new_embedding = rotation_matrix @ atom.e8_embedding
                
                new_atom = CQEAtom(data=atom.data, parent_id=atom.id)
                new_atom.e8_embedding = new_atom._project_to_e8_lattice(new_embedding)
                new_atom._compute_parity_channels()
                new_atom._validate_governance()
                new_atom.metadata = {'transformation': 'e8_rotation', 'original_id': atom.id}
                
            else:
                # Identity transformation
                new_atom = CQEAtom(
                    data=atom.data,
                    quad_encoding=atom.quad_encoding,
                    parent_id=atom.id,
                    metadata={'transformation': 'identity', 'original_id': atom.id}
                )
            
            result_atoms.append(new_atom)
        
        return result_atoms
    
    def _optimize_atoms(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> List[CQEAtom]:
        """Optimize atoms using MORSR protocol"""
        optimization_target = parameters.get('target', 'governance')
        max_iterations = parameters.get('max_iterations', 100)
        
        current_atoms = atoms.copy()
        
        for iteration in range(max_iterations):
            improved = False
            
            for i, atom in enumerate(current_atoms):
                # Try different transformations
                candidates = []
                
                # Quad space optimization
                for shift in [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)]:
                    candidate = self._transform_atoms([atom], {'type': 'quad_shift', 'shift': shift})[0]
                    candidates.append(candidate)
                
                # Select best candidate based on optimization target
                best_candidate = self._select_best_candidate(atom, candidates, optimization_target)
                
                if best_candidate and self._is_improvement(atom, best_candidate, optimization_target):
                    current_atoms[i] = best_candidate
                    improved = True
            
            if not improved:
                break  # Converged
        
        return current_atoms
    
    def _validate_atoms(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> List[CQEAtom]:
        """Validate atoms using parity channels and governance"""
        validation_level = parameters.get('level', 'basic')
        result_atoms = []
        
        for atom in atoms:
            validation_result = {
                'quad_valid': all(1 <= q <= 4 for q in atom.quad_encoding),
                'parity_valid': len(atom.parity_channels) == 8,
                'governance_valid': atom.governance_state != 'unlawful',
                'e8_valid': np.linalg.norm(atom.e8_embedding) <= 3.0
            }
            
            if validation_level == 'strict':
                validation_result['tqf_valid'] = atom.governance_state == 'tqf_lawful'
                validation_result['uvibs_valid'] = atom.governance_state == 'uvibs_compliant'
            
            # Create validation result atom
            result_atom = CQEAtom(
                data=validation_result,
                parent_id=atom.id,
                metadata={'validation_level': validation_level, 'original_id': atom.id}
            )
            
            result_atoms.append(result_atom)
        
        return result_atoms
    
    def _reason_with_atoms(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> List[CQEAtom]:
        """Perform reasoning operations on atoms"""
        reasoning_type = parameters.get('type', 'similarity')
        
        if reasoning_type == 'similarity':
            # Find similar atoms and create reasoning chains
            result_atoms = []
            
            for atom in atoms:
                similar_atoms = self.memory.find_similar_atoms(atom, max_distance=2.0, limit=5)
                
                reasoning_data = {
                    'source_atom': atom.id,
                    'similar_atoms': [(sim_atom.id, distance) for sim_atom, distance in similar_atoms],
                    'reasoning_type': 'similarity',
                    'confidence': 1.0 - (len(similar_atoms) / 10.0)  # More similar = higher confidence
                }
                
                reasoning_atom = CQEAtom(
                    data=reasoning_data,
                    parent_id=atom.id,
                    metadata={'reasoning_type': reasoning_type}
                )
                
                result_atoms.append(reasoning_atom)
            
            return result_atoms
        
        elif reasoning_type == 'inference':
            # Perform logical inference
            return self._perform_inference(atoms, parameters)
        
        else:
            return atoms
    
    def _perform_inference(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> List[CQEAtom]:
        """Perform logical inference using CQE principles"""
        # Simplified inference - in practice would use full CQE reasoning
        inference_rules = parameters.get('rules', [])
        result_atoms = []
        
        for atom in atoms:
            # Apply inference rules
            for rule in inference_rules:
                if self._rule_applies(atom, rule):
                    inferred_data = self._apply_rule(atom, rule)
                    
                    inference_atom = CQEAtom(
                        data=inferred_data,
                        parent_id=atom.id,
                        metadata={'inference_rule': rule, 'confidence': rule.get('confidence', 0.8)}
                    )
                    
                    result_atoms.append(inference_atom)
        
        return result_atoms
    
    def _rule_applies(self, atom: CQEAtom, rule: Dict[str, Any]) -> bool:
        """Check if inference rule applies to atom"""
        conditions = rule.get('conditions', [])
        
        for condition in conditions:
            if condition['type'] == 'governance':
                if atom.governance_state != condition['value']:
                    return False
            elif condition['type'] == 'quad_pattern':
                if atom.quad_encoding != tuple(condition['value']):
                    return False
            elif condition['type'] == 'data_type':
                if not isinstance(atom.data, condition['value']):
                    return False
        
        return True
    
    def _apply_rule(self, atom: CQEAtom, rule: Dict[str, Any]) -> Any:
        """Apply inference rule to atom"""
        action = rule.get('action', {})
        
        if action['type'] == 'transform':
            return action['transformation'](atom.data)
        elif action['type'] == 'conclude':
            return action['conclusion']
        else:
            return f"Rule {rule.get('name', 'unknown')} applied to {atom.id}"
    
    def _select_best_candidate(self, original: CQEAtom, candidates: List[CQEAtom], 
                              target: str) -> Optional[CQEAtom]:
        """Select best candidate based on optimization target"""
        if not candidates:
            return None
        
        if target == 'governance':
            # Prefer better governance states
            governance_order = ['tqf_lawful', 'uvibs_compliant', 'lawful', 'unlawful']
            best_candidate = min(candidates, 
                               key=lambda c: governance_order.index(c.governance_state))
        
        elif target == 'e8_norm':
            # Prefer smaller E8 norm (closer to origin)
            best_candidate = min(candidates, 
                               key=lambda c: np.linalg.norm(c.e8_embedding))
        
        else:
            # Default: first candidate
            best_candidate = candidates[0]
        
        return best_candidate
    
    def _is_improvement(self, original: CQEAtom, candidate: CQEAtom, target: str) -> bool:
        """Check if candidate is improvement over original"""
        if target == 'governance':
            governance_order = ['unlawful', 'lawful', 'uvibs_compliant', 'tqf_lawful']
            return (governance_order.index(candidate.governance_state) > 
                   governance_order.index(original.governance_state))
        
        elif target == 'e8_norm':
            return (np.linalg.norm(candidate.e8_embedding) < 
                   np.linalg.norm(original.e8_embedding))
        
        return False
    
    def _compute_cache_key(self, operation_type: CQEOperationType, 
                          atoms: List[CQEAtom], parameters: Dict[str, Any]) -> str:
        """Compute cache key for operation"""
        atom_ids = [atom.id for atom in atoms]
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        
        key_data = f"{operation_type.value}:{':'.join(atom_ids)}:{param_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
