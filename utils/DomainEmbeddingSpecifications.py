class DomainEmbeddingSpecifications:
    """
    Precise domain embedding specifications with worked examples.

    Addresses: "How are inversion counts or prosodic features quantitatively 
    normalized into lane vectors?"
    """

    @staticmethod
    def superpermutation_to_e8(permutation: List[int]) -> np.ndarray:
        """
        Embed superpermutation into E₈ space with complete specification.

        Args:
            permutation: List representing permutation (e.g., [3, 1, 4, 2])

        Returns:
            8D E₈ vector with formal normalization
        """
        n = len(permutation)

        # Step 1: Inversion count analysis
        inversions = []
        for i in range(n):
            for j in range(i + 1, n):
                if permutation[i] > permutation[j]:
                    inversions.append((i, j, permutation[i] - permutation[j]))

        total_inversions = len(inversions)
        max_inversions = n * (n - 1) // 2  # Theoretical maximum

        # Step 2: Feature extraction (8 components for E₈)
        features = np.zeros(8)

        # Feature 1: Normalized inversion density
        features[0] = total_inversions / max_inversions if max_inversions > 0 else 0

        # Feature 2: Longest increasing subsequence (LIS) ratio
        lis_length = DomainEmbeddingSpecifications._compute_lis_length(permutation)
        features[1] = lis_length / n if n > 0 else 0

        # Feature 3: Cycle structure complexity
        cycles = DomainEmbeddingSpecifications._get_cycle_structure(permutation)
        features[2] = len(cycles) / n if n > 0 else 0

        # Feature 4: Deviation from identity
        identity_deviation = sum(abs(permutation[i] - (i + 1)) for i in range(n))
        max_deviation = sum(range(n))
        features[3] = identity_deviation / max_deviation if max_deviation > 0 else 0

        # Feature 5: Entropy of position distribution
        position_entropy = DomainEmbeddingSpecifications._compute_entropy(permutation)
        max_entropy = np.log2(n) if n > 1 else 1
        features[4] = position_entropy / max_entropy

        # Feature 6: Fixed point ratio
        fixed_points = sum(1 for i in range(n) if permutation[i] == i + 1)
        features[5] = fixed_points / n if n > 0 else 0

        # Feature 7: Alternation pattern strength
        alternations = sum(1 for i in range(n-1) 
                          if (permutation[i] < permutation[i+1]) != (i % 2 == 0))
        features[6] = alternations / (n - 1) if n > 1 else 0

        # Feature 8: Spectral property (Fourier-like)
        if n > 0:
            normalized_perm = np.array(permutation) / n
            fft_magnitude = np.abs(np.fft.fft(normalized_perm, n=8))
            features[7] = np.mean(fft_magnitude)
        else:
            features[7] = 0

        # Step 3: Normalization to E₈ lattice scale
        # Ensure features are in [0, 1] then scale to lattice norm ≈ √2
        features = np.clip(features, 0, 1)
        norm_factor = np.sqrt(2) / (np.linalg.norm(features) + 1e-10)

        return features * norm_factor

    @staticmethod
    def audio_frame_to_e8(audio_frame: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Embed audio frame into E₈ space with prosodic feature extraction.

        Args:
            audio_frame: 1D audio samples (e.g., 1024 samples)
            sample_rate: Audio sample rate

        Returns:
            8D E₈ vector with prosodic features
        """
        # Step 1: Prosodic feature extraction
        features = np.zeros(8)

        # Feature 1: RMS energy (amplitude)
        rms = np.sqrt(np.mean(audio_frame ** 2))
        features[0] = np.clip(rms * 10, 0, 1)  # Scale factor for typical audio

        # Feature 2: Zero crossing rate (related to pitch)
        zero_crossings = np.sum(np.diff(np.sign(audio_frame)) != 0)
        features[1] = zero_crossings / len(audio_frame)

        # Feature 3: Spectral centroid (brightness)
        fft = np.abs(np.fft.fft(audio_frame))
        freqs = np.fft.fftfreq(len(audio_frame), 1/sample_rate)
        spectral_centroid = np.sum(freqs[:len(freqs)//2] * fft[:len(fft)//2]) / np.sum(fft[:len(fft)//2])
        features[2] = spectral_centroid / (sample_rate / 2)  # Normalize to Nyquist

        # Feature 4: Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs[:len(freqs)//2] - spectral_centroid) ** 2) * fft[:len(fft)//2]) / np.sum(fft[:len(fft)//2]))
        features[3] = spectral_bandwidth / (sample_rate / 4)  # Normalize

        # Feature 5: Spectral rolloff (90% of energy)
        cumulative_energy = np.cumsum(fft[:len(fft)//2] ** 2)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.9 * total_energy)[0][0]
        features[4] = rolloff_idx / (len(fft) // 2)

        # Feature 6: Mel-frequency cepstral coefficient (MFCC) mean
        # Simplified MFCC computation
        mel_filters = DomainEmbeddingSpecifications._create_mel_filter_bank(len(fft)//2, sample_rate)
        mfcc = np.log(np.dot(mel_filters, fft[:len(fft)//2] ** 2) + 1e-10)
        features[5] = np.mean(mfcc) / 10  # Scale factor

        # Feature 7: Temporal envelope variance
        envelope = np.abs(audio_frame)
        features[6] = np.var(envelope) / (np.mean(envelope) ** 2 + 1e-10)

        # Feature 8: Harmonic-to-noise ratio estimate
        # Simple harmonic detection via autocorrelation
        autocorr = np.correlate(audio_frame, audio_frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Find peak in autocorrelation (fundamental frequency)
        if len(autocorr) > 1:
            peak_idx = np.argmax(autocorr[1:]) + 1
            harmonic_strength = autocorr[peak_idx] / (autocorr[0] + 1e-10)
            features[7] = np.clip(harmonic_strength, 0, 1)
        else:
            features[7] = 0

        # Step 3: Normalization to E₈ lattice scale
        features = np.clip(features, 0, 1)
        norm_factor = np.sqrt(2) / (np.linalg.norm(features) + 1e-10)

        return features * norm_factor

    @staticmethod
    def scene_graph_to_e8(scene_graph: Dict[str, Any]) -> np.ndarray:
        """
        Embed scene graph into E₈ space with structural features.

        Args:
            scene_graph: Dictionary with nodes, edges, attributes
            Example: {
                'nodes': ['person', 'chair', 'room'],
                'edges': [('person', 'sits_on', 'chair'), ('chair', 'in', 'room')],
                'attributes': {'person': {'age': 25}, 'chair': {'color': 'red'}}
            }

        Returns:
            8D E₈ vector with scene structure features
        """
        nodes = scene_graph.get('nodes', [])
        edges = scene_graph.get('edges', [])
        attributes = scene_graph.get('attributes', {})

        features = np.zeros(8)

        # Feature 1: Node density
        features[0] = min(len(nodes) / 20, 1.0)  # Normalize by typical scene size

        # Feature 2: Edge density (connectivity)
        max_edges = len(nodes) * (len(nodes) - 1) if len(nodes) > 1 else 1
        features[1] = len(edges) / max_edges

        # Feature 3: Attribute complexity
        total_attributes = sum(len(attrs) for attrs in attributes.values())
        features[2] = min(total_attributes / (len(nodes) * 5), 1.0) if nodes else 0

        # Feature 4: Graph diameter (simplified)
        diameter = DomainEmbeddingSpecifications._compute_graph_diameter(nodes, edges)
        features[3] = diameter / len(nodes) if len(nodes) > 0 else 0

        # Feature 5: Clustering coefficient
        clustering = DomainEmbeddingSpecifications._compute_clustering_coefficient(nodes, edges)
        features[4] = clustering

        # Feature 6: Degree centralization
        degrees = DomainEmbeddingSpecifications._compute_node_degrees(nodes, edges)
        if degrees:
            max_degree = max(degrees.values())
            features[5] = max_degree / (len(nodes) - 1) if len(nodes) > 1 else 0
        else:
            features[5] = 0

        # Feature 7: Semantic diversity (simplified via edge types)
        unique_edge_types = set(edge[1] for edge in edges if len(edge) >= 3)
        features[6] = min(len(unique_edge_types) / 10, 1.0)  # Normalize by typical variety

        # Feature 8: Hierarchical depth
        hierarchy_depth = DomainEmbeddingSpecifications._compute_hierarchy_depth(nodes, edges)
        features[7] = min(hierarchy_depth / 5, 1.0)  # Normalize by typical depth

        # Step 3: Normalization to E₈ lattice scale
        features = np.clip(features, 0, 1)
        norm_factor = np.sqrt(2) / (np.linalg.norm(features) + 1e-10)

        return features * norm_factor

    # Helper methods for domain embedding
    @staticmethod
    def _compute_lis_length(seq: List[int]) -> int:
        """Compute longest increasing subsequence length."""
        if not seq:
            return 0

        dp = [1] * len(seq)
        for i in range(1, len(seq)):
            for j in range(i):
                if seq[j] < seq[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    @staticmethod
    def _get_cycle_structure(perm: List[int]) -> List[List[int]]:
        """Get cycle decomposition of permutation."""
        n = len(perm)
        visited = [False] * n
        cycles = []

        for i in range(n):
            if not visited[i]:
                cycle = []
                curr = i
                while not visited[curr]:
                    visited[curr] = True
                    cycle.append(curr + 1)  # 1-indexed
                    curr = perm[curr] - 1  # Convert to 0-indexed
                if len(cycle) > 1:
                    cycles.append(cycle)

        return cycles

    @staticmethod
    def _compute_entropy(seq: List[int]) -> float:
        """Compute Shannon entropy of sequence."""
        if not seq:
            return 0

        from collections import Counter
        counts = Counter(seq)
        probs = np.array(list(counts.values())) / len(seq)
        return -np.sum(probs * np.log2(probs + 1e-10))

    @staticmethod
    def _create_mel_filter_bank(n_filters: int, sample_rate: int, n_fft: int = 512) -> np.ndarray:
        """Create simplified mel filter bank."""
        # Simplified mel filter bank for demonstration
        filters = np.random.rand(13, n_fft // 2)  # 13 standard mel filters
        return filters / np.sum(filters, axis=1, keepdims=True)

    @staticmethod
    def _compute_graph_diameter(nodes: List[str], edges: List[Tuple]) -> int:
        """Compute graph diameter (simplified)."""
        if not nodes or not edges:
            return 0

        # Build adjacency list
        adj = {node: set() for node in nodes}
        for edge in edges:
            if len(edge) >= 2:
                adj[edge[0]].add(edge[1])
                adj[edge[1]].add(edge[0])

        max_distance = 0
        for start in nodes:
            distances = {start: 0}
            queue = [start]

            while queue:
                current = queue.pop(0)
                for neighbor in adj[current]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
                        max_distance = max(max_distance, distances[neighbor])

        return max_distance

    @staticmethod
    def _compute_clustering_coefficient(nodes: List[str], edges: List[Tuple]) -> float:
        """Compute graph clustering coefficient."""
        if len(nodes) < 3:
            return 0

        adj = {node: set() for node in nodes}
        for edge in edges:
            if len(edge) >= 2:
                adj[edge[0]].add(edge[1])
                adj[edge[1]].add(edge[0])

        total_clustering = 0
        for node in nodes:
            neighbors = list(adj[node])
            if len(neighbors) < 2:
                continue

            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        triangles += 1

            if possible_triangles > 0:
                total_clustering += triangles / possible_triangles

        return total_clustering / len(nodes) if nodes else 0

    @staticmethod
    def _compute_node_degrees(nodes: List[str], edges: List[Tuple]) -> Dict[str, int]:
        """Compute node degrees."""
        degrees = {node: 0 for node in nodes}
        for edge in edges:
            if len(edge) >= 2:
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
        return degrees

    @staticmethod
    def _compute_hierarchy_depth(nodes: List[str], edges: List[Tuple]) -> int:
        """Compute maximum hierarchy depth."""
        # Simplified: assume edges with certain relationships indicate hierarchy
        hierarchical_edges = [e for e in edges if len(e) >= 3 and e[1] in ['contains', 'has', 'owns']]

        if not hierarchical_edges:
            return 1

        # Build directed graph for hierarchy
        children = {node: [] for node in nodes}
        for edge in hierarchical_edges:
            children[edge[0]].append(edge[2])

        def dfs_depth(node):
            if not children[node]:
                return 1
            return 1 + max(dfs_depth(child) for child in children[node])

        return max(dfs_depth(node) for node in nodes)
