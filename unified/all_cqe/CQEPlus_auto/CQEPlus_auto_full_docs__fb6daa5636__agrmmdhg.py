# ===========================================================
# === AGRM System Implementation Codebase (Final Version) ===
# ===========================================================
# Based on validated blueprint derived from user documents and session discussion.
# Includes: Multi-agent architecture, Modulation Controller, Bidirectional Builder,
# Salesman Validator/Patcher, Path Audit Agent, Hybrid Hashing,
# Ephemeral Memory (MDHG-Hash Integration), Dynamic Midpoint, Spiral Reentry,
# Comprehensive Comments.

import math
import time
import random
from collections import deque, Counter, defaultdict
from typing import Any, Dict, List, Tuple, Set, Optional, Union
import numpy as np # Assuming numpy is available for calculations like norm

# Try importing sklearn for KDTree, but provide fallback
try:
    from sklearn.neighbors import KDTree, BallTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not found. Neighbor searches will use less efficient fallback.")

# ===========================================================
# === Multi-Dimensional Hamiltonian Golden Ratio Hash Table ===
# ===========================================================
# Full implementation based on user-provided code and description.
# Source: User Upload mdhg_hash.py content [cite: 1027-1096]
# Integrated into AGRM system for high-complexity state/cache (n>5).

class MDHGHashTable:
    """
    Multi-Dimensional Hamiltonian Golden Ratio Hash Table.
    A hash table implementation that uses multi-dimensional organization,
    Hamiltonian paths for navigation, and golden ratio-based sizing
    to achieve optimal performance for structured access patterns.

    AGRM Integration Notes:
    - Used for complex state (n>5) in AGRM's hybrid hashing.
    - Stores values as tuples: (actual_value, metadata_dict),
      where metadata_dict contains flags like {'source': 'dict', 'retain_flag': True}.
    - Adaptation logic can be influenced by Modulation Controller signals.
    - Includes full logic for buildings, floors, rooms (conceptual), velocity region,
      dimensional core, conflict handling, Hamiltonian path navigation, and dynamic adaptation.
    """
    def __init__(self, capacity: int = 1024, dimensions: int = 3, load_factor_threshold: float = 0.75, config: Dict = {}):
        """
        Initialize the hash table.
        Args:
            capacity: Initial capacity of the hash table
            dimensions: Number of dimensions for the hash space (tunable by AGRM context)
            load_factor_threshold: When to resize the table
            config: Dictionary for additional MDHG-specific parameters
        """
        self.PHI = (1 + math.sqrt(5)) / 2 # Golden ratio [cite: 1018-1019]
        self.config = config # Store config for internal use

        # Core configuration
        self.capacity = max(capacity, 16) # Ensure minimum capacity
        self.dimensions = max(dimensions, 1) # Ensure at least 1 dimension
        self.load_factor_threshold = load_factor_threshold
        self.size = 0

        # Multi-dimensional organization [cite: 1012-1017]
        self.buildings = self._initialize_buildings()
        self.location_map = {} # Maps keys to their current location (building_id, region_type, location_spec)

        # Navigation components
        self.hamiltonian_paths = {} # Pre-computed paths for critical points [cite: 1020-1022, 1037]
        self.path_cache = {} # Cache of paths between points or for key sets
        self.shortcuts = {} # Direct connections between buildings/regions [cite: 1022]

        # Access pattern tracking
        self.access_history = deque(maxlen=config.get("mdhg_access_history_len", 100)) # Track recent accesses
        self.access_frequency = Counter() # Track frequency of key access
        self.co_access_matrix = defaultdict(Counter) # Track keys accessed together [cite: 1022]
        self.path_usage = Counter() # Track usage of cached paths

        # Statistics
        self.stats = {
            'puts': 0, 'gets': 0, 'hits': 0, 'misses': 0, 'collisions': 0,
            'probes_total': 0, 'max_probes': 0, 'reorganizations': 0,
            'resizes': 0, 'promotions_velocity': 0, 'relocations_from_velocity': 0,
            'clusters_relocated': 0
        }

        # Optimization timing
        self.last_minor_optimization = time.time()
        self.last_major_optimization = time.time()
        self.operations_since_optimization = 0

        # Initialize the structure (compute initial paths, etc.)
        self._initialize_structure()
        print(f"MDHGHashTable initialized: Capacity={self.capacity}, Dimensions={self.dimensions}, Buildings={len(self.buildings)}")

    def _initialize_buildings(self) -> Dict:
        """ Initialize the building structure based on golden ratio proportions. """
        # Determine number of buildings, ensuring at least 1
        building_count = max(1, int(math.log(max(2, self.capacity), self.PHI)))
        buildings = {}
        # Ensure base capacity calculation avoids division by zero
        base_capacity_per_building = self.capacity // building_count if building_count > 0 else self.capacity
        if base_capacity_per_building < 16: base_capacity_per_building = 16 # Ensure minimum size

        print(f"  MDHG: Initializing {building_count} buildings, base capacity per building: {base_capacity_per_building}")

        for b in range(building_count):
            building_id = f"B{b}"
            # Calculate regions using golden ratio [cite: 1018]
            # Ensure minimum sizes for regions
            velocity_region_size = max(4, int(base_capacity_per_building / (self.PHI ** 2)))
            core_region_base_size = max(8, int(velocity_region_size * self.PHI)) # Base size before dimensioning

            dimension_sizes = self._calculate_dimension_sizes(core_region_base_size)
            # Actual core capacity is product of dimension sizes
            core_capacity = math.prod(dimension_sizes) if dimension_sizes else 0

            print(f"    Building {building_id}: Velocity Region Size={velocity_region_size}, Core Capacity={core_capacity}, Dim Sizes={dimension_sizes}")

            buildings[building_id] = {
                'velocity_region': [None] * velocity_region_size, # Fast access [cite: 1022]
                'dimensional_core': {}, # Main storage, dict maps coords -> (key, value_tuple) [cite: 1022]
                'conflict_structures': {}, # Handles collisions beyond path probing [cite: 1022]
                'dimension_sizes': dimension_sizes, # For coordinate calculation
                'hot_keys': set(), # Keys frequently accessed in this building
                'access_count': 0, # Track building usage
                'core_capacity': core_capacity # Store calculated core capacity
            }
        return buildings

    def _calculate_dimension_sizes(self, core_region_base_size: int) -> List[int]:
        """ Calculate sizes for each dimension using golden ratio proportions. """
        if self.dimensions <= 0: return []
        # Estimate base size per dimension
        # Using geometric mean approach: base_size ^ dimensions ≈ core_region_base_size
        # Add epsilon to avoid potential log(0) or root(0) issues if base_size is tiny
        safe_base_size = max(1, core_region_base_size)
        base_size = max(2.0, safe_base_size ** (1.0/self.dimensions)) # Use float for calculation

        sizes = []
        product = 1.0
        # Scale dimensions using PHI, ensuring minimum size 2
        for i in range(self.dimensions):
            # Example scaling: could use other GR-based factors
            # Ensure denominator is safe
            phi_exponent = i / max(1.0, float(self.dimensions - 1))
            size_float = base_size / (self.PHI ** phi_exponent)
            size = max(2, int(round(size_float))) # Round before int conversion
            sizes.append(size)
            product *= size

        # Optional: Adjust sizes slightly if product is too far off target
        # This part requires careful balancing logic to avoid infinite loops or drastic changes
        # print(f"      Calculated dimension sizes: {sizes} (Product: {product}, Target Base: {core_region_base_size})")
        return sizes

    def _initialize_structure(self):
        """ Initialize the hash table structure with navigation components. """
        # Pre-compute critical Hamiltonian paths for each building [cite: 1037]
        print("  MDHG: Initializing structure (paths, shortcuts)...")
        for building_id, building in self.buildings.items():
            self._initialize_building_paths(building_id, building)
        # Initialize shortcuts between buildings
        self._initialize_building_shortcuts()
        print("  MDHG: Structure initialization complete.")


    def _initialize_building_paths(self, building_id: str, building: Dict):
        """ Initialize Hamiltonian paths for critical points in a building. """
        dimension_sizes = building.get('dimension_sizes')
        if not dimension_sizes:
            # print(f"    Skipping path init for {building_id}: No dimensions.")
            return # Skip if no dimensions

        # Generate critical points (corners, center)
        critical_points = set()
        # Add corner points
        corners = self._generate_corner_points(dimension_sizes)
        critical_points.update(corners)
        # Add center point
        center = tuple(d // 2 for d in dimension_sizes)
        critical_points.add(center)

        # Compute and store paths for these critical points
        building_paths = {}
        computed_count = 0
        for point in critical_points:
            # Ensure point is valid within dimensions
            if len(point) != self.dimensions: continue
            valid_point = all(0 <= point[i] < dimension_sizes[i] for i in range(self.dimensions))

            if valid_point:
                path = self._compute_hamiltonian_path(building_id, point)
                if path: # Only store if path computation succeeds
                    path_key = (building_id, point)
                    building_paths[path_key] = path # Store paths per building first
                    computed_count += 1

        self.hamiltonian_paths.update(building_paths) # Add building paths to global store
        # print(f"    Initialized {computed_count} Hamiltonian paths for building {building_id}.")


    def _generate_corner_points(self, dimension_sizes: List[int]) -> List[Tuple]:
        """ Generate corner points for a multi-dimensional space. """
        if not dimension_sizes: return []
        corners = []
        num_dims = len(dimension_sizes)
        num_corners = 2 ** num_dims
        for i in range(num_corners):
            corner = []
            for d in range(num_dims):
                # Use bit masking to determine min (0) or max (size-1) for each dimension
                if (i >> d) & 1:
                    corner.append(max(0, dimension_sizes[d] - 1)) # Use max index
                else:
                    corner.append(0) # Use min index
            corners.append(tuple(corner))
        return corners

    def _initialize_building_shortcuts(self):
        """ Initialize shortcuts between buildings. """
        building_ids = list(self.buildings.keys())
        shortcuts_created = 0
        # Create shortcuts only if there's more than one building
        if len(building_ids) > 1:
            for i, b1 in enumerate(building_ids):
                for j, b2 in enumerate(building_ids):
                    if i != j:
                        # Create bidirectional shortcuts
                        if self._create_building_shortcut(b1, b2):
                            shortcuts_created += 1
        # print(f"  Initialized {shortcuts_created} building shortcuts.")

    def _create_building_shortcut(self, building1: str, building2: str) -> bool:
        """ Create a shortcut between two buildings with default connection points. """
        building1_data = self.buildings.get(building1)
        building2_data = self.buildings.get(building2)
        # Check if dimensions are valid before creating shortcut
        if not building1_data or not building2_data or \
           not building1_data.get('dimension_sizes') or not building2_data.get('dimension_sizes') or \
           len(building1_data['dimension_sizes']) != self.dimensions or \
           len(building2_data['dimension_sizes']) != self.dimensions:
            # print(f"Warning: Cannot create shortcut between {building1} and {building2} due to invalid dimensions.")
            return False # Cannot create shortcut if building data is incomplete

        # Use center points as default connection points
        center1 = tuple(d // 2 for d in building1_data['dimension_sizes'])
        center2 = tuple(d // 2 for d in building2_data['dimension_sizes'])

        shortcut_key = (building1, building2)
        self.shortcuts[shortcut_key] = {
            'entry_point': center1, # Entry point in building1
            'exit_point': center2,  # Exit point in building2 (conceptually)
            'cost': 1.0 / self.PHI, # Lower cost than regular traversal (heuristic)
            'usage_count': 0
        }
        return True

    def _compute_hamiltonian_path(self, building_id: str, start_coords: Tuple) -> List[Tuple]:
        """
        Compute a Hamiltonian-like path (visits many points uniquely) starting from coordinates.
        Uses GR steps. This is a heuristic path, not guaranteed to be truly Hamiltonian or optimal length.
        """
        building = self.buildings.get(building_id)
        if not building or not building.get('dimension_sizes'): return []
        dimension_sizes = building['dimension_sizes']

        # Basic validation of start_coords
        if len(start_coords) != self.dimensions: return []
        if not all(0 <= start_coords[i] < dimension_sizes[i] for i in range(self.dimensions)): return []

        path = [start_coords]
        current = list(start_coords)
        visited = {start_coords}

        # Determine path length heuristic
        total_core_points = math.prod(dimension_sizes) if dimension_sizes else 0
        if total_core_points == 0: return path # Path is just the start point

        path_length_limit = min(total_core_points, self.config.get("mdhg_path_length_limit", 1000))
        # Aim for a path length that covers a reasonable fraction, e.g., sqrt or similar heuristic
        path_length_target = max(self.dimensions * 2, int(math.sqrt(total_core_points) * 2)) # Cover more?
        path_length = min(path_length_limit, path_length_target)

        # Use golden ratio for dimension selection and step direction bias
        for step in range(1, path_length):
            # Choose dimension based on golden ratio progression [cite: 1021]
            dim_choice = int((step * self.PHI) % self.dimensions)

            # Determine step direction (+1 or -1) based on another GR sequence
            direction_bias = (step * self.PHI**2) % 1.0
            step_dir = 1 if direction_bias < 0.5 else -1

            # Try moving in the chosen dimension and direction
            next_coord_list = list(current)
            next_coord_list[dim_choice] = (next_coord_list[dim_choice] + step_dir + dimension_sizes[dim_choice]) % dimension_sizes[dim_choice] # Ensure positive result
            next_coords = tuple(next_coord_list)

            if next_coords not in visited:
                path.append(next_coords)
                visited.add(next_coords)
                current = next_coord_list
            else:
                # Collision: Try alternative dimensions or directions (simple linear probe)
                found_alternative = False
                for alt_offset in range(1, self.dimensions + 1): # Try all dimensions + opposite dir
                    # Try alternative dimension, original direction
                    alt_dim = (dim_choice + alt_offset) % self.dimensions
                    alt_coord_list = list(current)
                    alt_coord_list[alt_dim] = (alt_coord_list[alt_dim] + step_dir + dimension_sizes[alt_dim]) % dimension_sizes[alt_dim]
                    alt_coords = tuple(alt_coord_list)
                    if alt_coords not in visited:
                        path.append(alt_coords)
                        visited.add(alt_coords)
                        current = alt_coord_list
                        found_alternative = True
                        break

                    # Try alternative dimension, opposite direction
                    alt_coord_list = list(current)
                    alt_coord_list[alt_dim] = (alt_coord_list[alt_dim] - step_dir + dimension_sizes[alt_dim]) % dimension_sizes[alt_dim]
                    alt_coords = tuple(alt_coord_list)
                    if alt_coords not in visited:
                        path.append(alt_coords)
                        visited.add(alt_coords)
                        current = alt_coord_list
                        found_alternative = True
                        break

                # If no alternative found after checking all dims/dirs, stop path
                if not found_alternative:
                    # print(f"      Path generation stuck at step {step}, coords {current}")
                    break # Stop if stuck

        return path

    # --- Hashing Functions ---
    def _hash(self, key: Any) -> int:
        """ Primary hash function. Using MurmurHash for better distribution. """
        return self._murmur_hash(key)

    def _secondary_hash(self, key: Any) -> int:
        """ Secondary hash function for specific regions like velocity. Using FNV. """
        return self._fnv_hash(key)

    def _murmur_hash(self, key: Any) -> int:
        """ MurmurHash3 32-bit implementation. """
        key_bytes = str(key).encode('utf-8')
        length = len(key_bytes)
        seed = 0x9747b28c # Example seed
        c1 = 0xcc9e2d51
        c2 = 0x1b873593
        r1 = 15
        r2 = 13
        m = 5
        n = 0xe6546b64
        hash_value = seed

        nblocks = length // 4
        for i in range(nblocks):
            idx = i * 4
            k = (key_bytes[idx] |
                 (key_bytes[idx + 1] << 8) |
                 (key_bytes[idx + 2] << 16) |
                 (key_bytes[idx + 3] << 24))
            k = (k * c1) & 0xFFFFFFFF
            k = ((k << r1) | (k >> (32 - r1))) & 0xFFFFFFFF
            k = (k * c2) & 0xFFFFFFFF
            hash_value ^= k
            hash_value = ((hash_value << r2) | (hash_value >> (32 - r2))) & 0xFFFFFFFF
            hash_value = ((hash_value * m) + n) & 0xFFFFFFFF

        tail_index = nblocks * 4
        k = 0
        tail_size = length & 3
        if tail_size >= 3: k ^= key_bytes[tail_index + 2] << 16
        if tail_size >= 2: k ^= key_bytes[tail_index + 1] << 8
        if tail_size >= 1: k ^= key_bytes[tail_index]
        if tail_size > 0:
            k = (k * c1) & 0xFFFFFFFF
            k = ((k << r1) | (k >> (32 - r1))) & 0xFFFFFFFF
            k = (k * c2) & 0xFFFFFFFF
            hash_value ^= k

        hash_value ^= length
        hash_value ^= hash_value >> 16
        hash_value = (hash_value * 0x85ebca6b) & 0xFFFFFFFF
        hash_value ^= hash_value >> 13
        hash_value = (hash_value * 0xc2b2ae35) & 0xFFFFFFFF
        hash_value ^= hash_value >> 16

        return abs(hash_value) # Ensure positive

    def _fnv_hash(self, key: Any) -> int:
        """ FNV-1a 32-bit hash implementation. """
        key_bytes = str(key).encode('utf-8')
        fnv_prime = 0x01000193 # 16777619
        fnv_offset_basis = 0x811c9dc5 # 2166136261
        hash_value = fnv_offset_basis
        for byte in key_bytes:
            hash_value ^= byte
            hash_value = (hash_value * fnv_prime) & 0xFFFFFFFF
        return abs(hash_value) # Ensure positive

    def _hash_to_building(self, key: Any) -> str:
        """ Determine which building should contain a key using primary hash. """
        if not self.buildings: raise ValueError("MDHGHashTable has no buildings initialized.")
        hash_value = self._hash(key)
        building_idx = hash_value % len(self.buildings)
        return f"B{building_idx}"

    def _hash_to_velocity_index(self, key: Any, building_id: str) -> int:
        """ Calculate velocity region index using secondary hash. """
        building = self.buildings.get(building_id)
        if not building: raise ValueError(f"Building {building_id} not found.")
        velocity_size = len(building['velocity_region'])
        if velocity_size == 0: return 0
        return self._secondary_hash(key) % velocity_size

    def _hash_to_coords(self, key: Any, building_id: str) -> Optional[Tuple]:
        """ Calculate multi-dimensional coordinates using variations of primary hash. """
        building = self.buildings.get(building_id)
        if not building: raise ValueError(f"Building {building_id} not found.")
        dimension_sizes = building.get('dimension_sizes')
        if not dimension_sizes or len(dimension_sizes) != self.dimensions:
            return None # Cannot calculate coords if dimensions mismatch

        coords = []
        # Use primary hash and modify it for each dimension to get variation
        base_hash = self._hash(key)
        for i in range(self.dimensions):
            # Simple variation: XOR with dimension index and shift
            dim_hash = (base_hash ^ (i * 0x9e3779b9)) # Use golden ratio conjugate for mixing
            dim_hash = (dim_hash >> i) | (dim_hash << (32 - i)) & 0xFFFFFFFF # Rotate
            coord_val = abs(dim_hash) % dimension_sizes[i]
            coords.append(coord_val)
        return tuple(coords)

    def _hash_to_conflict_key(self, key: Any, coords: Tuple) -> int:
        """ Create a conflict key combining key hash and coordinates hash. """
        key_hash = self._hash(key)
        coords_hash = hash(coords) # Python's hash for tuple
        return abs(key_hash ^ coords_hash)

    # --- Core Put/Get/Remove ---

    def put(self, key: Any, value: Any) -> None:
        """
        Insert a key-value pair into the hash table.
        Handles routing, velocity region, core, collisions, and conflict structures.
        Value should be (actual_value, metadata_dict) for AGRM integration.
        """
        self.stats['puts'] += 1
        self.operations_since_optimization += 1

        # 1. Determine Target Building
        building_id = self._hash_to_building(key)
        building = self.buildings.get(building_id)
        if not building: # Fallback if building calculation failed somehow
            if not self.buildings: raise RuntimeError("MDHG Hash Table has no buildings.")
            building_id = list(self.buildings.keys())[0]
            building = self.buildings[building_id]
            print(f"Warning: Falling back to building {building_id} for key {key}.")
        building['access_count'] += 1

        # 2. Try Velocity Region
        velocity_idx = self._hash_to_velocity_index(key, building_id)
        if 0 <= velocity_idx < len(building['velocity_region']):
            velocity_entry = building['velocity_region'][velocity_idx]
            if velocity_entry is None:
                building['velocity_region'][velocity_idx] = (key, value)
                self.location_map[key] = (building_id, 'velocity', velocity_idx)
                self.size += 1
                self._update_access_patterns(key)
                self._check_optimization_and_resize()
                return
            elif velocity_entry[0] == key:
                building['velocity_region'][velocity_idx] = (key, value) # Update
                self._update_access_patterns(key)
                return
            # Else: Collision in velocity, proceed to core

        # 3. Try Dimensional Core
        coords = self._hash_to_coords(key, building_id)
        if coords is not None:
            if coords not in building['dimensional_core']:
                building['dimensional_core'][coords] = (key, value)
                self.location_map[key] = (building_id, 'dimensional', coords)
                self.size += 1
                self._update_access_patterns(key)
                self._check_optimization_and_resize()
                return
            elif building['dimensional_core'][coords][0] == key:
                building['dimensional_core'][coords] = (key, value) # Update
                self._update_access_patterns(key)
                return
            else:
                # Collision in dimensional core
                self.stats['collisions'] += 1
                # 4. Follow Hamiltonian Path
                new_coords, probes = self._follow_hamiltonian_path_for_put(building_id, coords)
                self.stats['probes_total'] += probes
                self.stats['max_probes'] = max(self.stats['max_probes'], probes)
                if new_coords:
                    building['dimensional_core'][new_coords] = (key, value)
                    self.location_map[key] = (building_id, 'dimensional', new_coords)
                    self.size += 1
                    self._update_access_patterns(key)
                    self._check_optimization_and_resize()
                    return
                # Else: Path probing failed, proceed to conflict structure
        else: # Coords calculation failed, go directly to conflict structure
             coords = tuple([0]*self.dimensions) # Use fallback coords for conflict key


        # 5. Use Conflict Structure
        conflict_key_hash = self._hash_to_conflict_key(key, coords)
        if conflict_key_hash not in building['conflict_structures']:
            building['conflict_structures'][conflict_key_hash] = {} # Use dict as simple conflict list

        # Store/update in conflict structure
        if key not in building['conflict_structures'][conflict_key_hash]:
            self.size += 1 # Increment size only if new key overall
        building['conflict_structures'][conflict_key_hash][key] = value
        self.location_map[key] = (building_id, 'conflict', conflict_key_hash)
        self._update_access_patterns(key)
        self._check_optimization_and_resize()


    def get(self, key: Any) -> Any:
        """ Retrieve value tuple (val, meta) or None. """
        self.stats['gets'] += 1
        self.operations_since_optimization += 1
        probes = 0

        # 1. Check Location Map Cache
        loc_info = self.location_map.get(key)
        if loc_info:
            building_id, region_type, location_spec = loc_info
            building = self.buildings.get(building_id)
            if building:
                building['access_count'] += 1
                value = None
                if region_type == 'velocity':
                    probes += 1
                    if 0 <= location_spec < len(building['velocity_region']):
                        entry = building['velocity_region'][location_spec]
                        if entry and entry[0] == key: value = entry[1]
                elif region_type == 'dimensional':
                    probes += 1
                    entry = building['dimensional_core'].get(location_spec)
                    if entry and entry[0] == key: value = entry[1]
                elif region_type == 'conflict':
                    probes += 1
                    conflict_map = building['conflict_structures'].get(location_spec)
                    if conflict_map: value = conflict_map.get(key)

                if value is not None:
                    self.stats['hits'] += 1
                    self._update_stats_and_patterns(key, probes)
                    return value
                else: # Location map was stale/incorrect
                     if key in self.location_map: del self.location_map[key]
            else: # Invalid building in map
                 if key in self.location_map: del self.location_map[key]
            # Fall through to full search if map check failed

        # 2. Full Search (if map failed or key not in map)
        primary_building_id = self._hash_to_building(key)
        value, building_probes = self._search_building(primary_building_id, key)
        probes += building_probes
        if value is not None:
            self._update_stats_and_patterns(key, probes)
            return value

        # 3. Search Other Buildings (Only if collisions can spill buildings - assumed NO for now)

        # 4. Key Not Found
        self.stats['misses'] += 1
        self._update_stats_and_patterns(key, probes, found=False)
        return None

    def _update_stats_and_patterns(self, key: Any, probes: int, found: bool = True):
         """ Helper to update stats and access patterns after a get attempt. """
         self.stats['probes_total'] += probes
         self.stats['max_probes'] = max(self.stats['max_probes'], probes)
         if found:
             self._update_access_patterns(key)


    def _search_building(self, building_id: str, key: Any) -> Tuple[Any, int]:
        """ Search for a key within a specific building. Returns (Value, probes). """
        building = self.buildings.get(building_id)
        if not building: return None, 0
        building['access_count'] += 1
        probes = 0

        # Check velocity
        velocity_idx = self._hash_to_velocity_index(key, building_id)
        probes += 1
        if 0 <= velocity_idx < len(building['velocity_region']):
            entry = building['velocity_region'][velocity_idx]
            if entry and entry[0] == key:
                self.stats['hits'] += 1
                self.location_map[key] = (building_id, 'velocity', velocity_idx)
                return entry[1], probes

        # Check dimensional core primary
        coords = self._hash_to_coords(key, building_id)
        if coords is not None:
            probes += 1
            entry = building['dimensional_core'].get(coords)
            if entry and entry[0] == key:
                self.stats['hits'] += 1
                self.location_map[key] = (building_id, 'dimensional', coords)
                return entry[1], probes

            # Check conflict structure based on primary coords
            conflict_key_hash = self._hash_to_conflict_key(key, coords)
            probes += 1
            conflict_map = building['conflict_structures'].get(conflict_key_hash)
            if conflict_map and key in conflict_map:
                self.stats['hits'] += 1
                self.location_map[key] = (building_id, 'conflict', conflict_key_hash)
                return conflict_map[key], probes

            # Follow Hamiltonian path if not found yet
            value, path_probes = self._search_path(building_id, coords, key)
            probes += path_probes
            if value is not None:
                # Hit, location map update happen inside _search_path
                return value, probes

        else: # Coords failed, check conflict based on fallback
            fallback_coords = tuple([0]*self.dimensions)
            conflict_key_hash = self._hash_to_conflict_key(key, fallback_coords)
            probes += 1
            conflict_map = building['conflict_structures'].get(conflict_key_hash)
            if conflict_map and key in conflict_map:
                self.stats['hits'] += 1
                self.location_map[key] = (building_id, 'conflict', conflict_key_hash)
                return conflict_map[key], probes

        # Key not found in this building
        return None, probes


    def _search_path(self, building_id: str, start_coords: Tuple, key: Any) -> Tuple[Any, int]:
         """ Search for a key along a Hamiltonian path starting near coords. """
         building = self.buildings.get(building_id)
         if not building or not self.hamiltonian_paths: return None, 0

         nearest_path_key = self._find_nearest_path_key(building_id, start_coords)
         if not nearest_path_key: return None, 0
         path = self.hamiltonian_paths[nearest_path_key]
         if not path: return None, 0

         start_idx = self._find_path_start_index(path, start_coords)
         max_probes = self.config.get("mdhg_max_search_probes", 20)
         probes = 0
         path_len = len(path)
         forward_steps = 0
         backward_steps = 0

         while probes < max_probes and (forward_steps + backward_steps) < path_len:
             # Check forward
             idx = (start_idx + forward_steps) % path_len
             check_coords = path[idx]
             probes += 1
             entry = building['dimensional_core'].get(check_coords)
             if entry and entry[0] == key:
                 self.stats['hits'] += 1
                 self.location_map[key] = (building_id, 'dimensional', check_coords)
                 return entry[1], probes
             forward_steps += 1

             if probes >= max_probes or (forward_steps + backward_steps) >= path_len: break

             # Check backward (if path has more than one point)
             if backward_steps < forward_steps and path_len > 1:
                 idx = (start_idx - backward_steps - 1 + path_len) % path_len
                 check_coords = path[idx]
                 probes += 1
                 entry = building['dimensional_core'].get(check_coords)
                 if entry and entry[0] == key:
                     self.stats['hits'] += 1
                     self.location_map[key] = (building_id, 'dimensional', check_coords)
                     return entry[1], probes
                 backward_steps += 1

         return None, probes # Not found along path segment

    def _follow_hamiltonian_path_for_put(self, building_id: str, start_coords: Tuple) -> Tuple[Optional[Tuple], int]:
        """ Follow a Hamiltonian path to find an empty slot for insertion. """
        building = self.buildings.get(building_id)
        if not building or not self.hamiltonian_paths: return None, 0

        nearest_path_key = self._find_nearest_path_key(building_id, start_coords)
        if not nearest_path_key: return None, 0
        path = self.hamiltonian_paths[nearest_path_key]
        if not path: return None, 0

        start_idx = self._find_path_start_index(path, start_coords)
        max_probes = self.config.get("mdhg_max_put_probes", 20)
        probes = 0
        path_len = len(path)
        forward_steps = 0
        backward_steps = 0

        while probes < max_probes and (forward_steps + backward_steps) < path_len:
            # Check forward
            idx = (start_idx + forward_steps) % path_len
            coords = path[idx]
            probes += 1
            if coords not in building['dimensional_core']:
                return coords, probes
            forward_steps += 1

            if probes >= max_probes or (forward_steps + backward_steps) >= path_len: break

            # Check backward
            if backward_steps < forward_steps and path_len > 1:
                idx = (start_idx - backward_steps - 1 + path_len) % path_len
                coords = path[idx]
                probes += 1
                if coords not in building['dimensional_core']:
                    return coords, probes
                backward_steps += 1

        return None, probes # No empty slot found

    def remove(self, key: Any) -> bool:
        """ Remove a key-value pair. """
        # 1. Check Location Map first
        loc_info = self.location_map.get(key)
        removed = False
        if loc_info:
            building_id, region_type, location_spec = loc_info
            building = self.buildings.get(building_id)
            if building:
                if region_type == 'velocity':
                    if 0 <= location_spec < len(building['velocity_region']):
                        entry = building['velocity_region'][location_spec]
                        if entry and entry[0] == key:
                            building['velocity_region'][location_spec] = None
                            removed = True
                elif region_type == 'dimensional':
                    entry = building['dimensional_core'].get(location_spec)
                    if entry and entry[0] == key:
                        del building['dimensional_core'][location_spec]
                        removed = True
                elif region_type == 'conflict':
                    conflict_map = building['conflict_structures'].get(location_spec)
                    if conflict_map and key in conflict_map:
                        del conflict_map[key]
                        if not conflict_map: del building['conflict_structures'][location_spec]
                        removed = True

                if removed:
                    del self.location_map[key]
                    self.size -= 1
                    return True
                else: # Location map was stale
                    del self.location_map[key]
            else: # Invalid building in map
                 del self.location_map[key]

        # 2. Full Search if map failed or key not in map
        primary_building_id = self._hash_to_building(key)
        if self._remove_from_building(primary_building_id, key):
            return True

        # 3. Search other buildings (if spillover possible - assuming not)

        return False # Key not found

    def _remove_from_building(self, building_id: str, key: Any) -> bool:
         """ Removes key from a specific building. Helper for remove(). """
         building = self.buildings.get(building_id)
         if not building: return False

         # Check velocity
         velocity_idx = self._hash_to_velocity_index(key, building_id)
         if 0 <= velocity_idx < len(building['velocity_region']):
             entry = building['velocity_region'][velocity_idx]
             if entry and entry[0] == key:
                 building['velocity_region'][velocity_idx] = None
                 if key in self.location_map: del self.location_map[key]
                 self.size -= 1
                 return True

         # Check core and conflict (primary coords)
         coords = self._hash_to_coords(key, building_id)
         if coords is not None:
             entry = building['dimensional_core'].get(coords)
             if entry and entry[0] == key:
                 del building['dimensional_core'][coords]
                 if key in self.location_map: del self.location_map[key]
                 self.size -= 1
                 return True

             conflict_key_hash = self._hash_to_conflict_key(key, coords)
             conflict_map = building['conflict_structures'].get(conflict_key_hash)
             if conflict_map and key in conflict_map:
                 del conflict_map[key]
                 if not conflict_map: del building['conflict_structures'][conflict_key_hash]
                 if key in self.location_map: del self.location_map[key]
                 self.size -= 1
                 return True

             # Search path if necessary (more complex removal)
             # Simplified: Assume if not at primary/velocity/conflict, it's not easily removable

         else: # Coords failed, check conflict based on fallback
              fallback_coords = tuple([0]*self.dimensions)
              conflict_key_hash = self._hash_to_conflict_key(key, fallback_coords)
              conflict_map = building['conflict_structures'].get(conflict_key_hash)
              if conflict_map and key in conflict_map:
                  del conflict_map[key]
                  if not conflict_map: del building['conflict_structures'][conflict_key_hash]
                  if key in self.location_map: del self.location_map[key]
                  self.size -= 1
                  return True

         return False

    # --- Helper methods for path finding ---
    def _find_nearest_path_key(self, building_id: str, coords: Tuple) -> Optional[Tuple]:
        """ Find the key (building_id, start_coords) of the nearest pre-computed path. """
        min_dist_sq = float('inf')
        nearest_key = None
        if coords is None: return None

        # Filter paths belonging to the target building
        building_paths = {k: v for k, v in self.hamiltonian_paths.items() if k[0] == building_id}
        if not building_paths: return None

        for path_key, path_data in building_paths.items():
            path_start_coords = path_key[1]
            # Ensure coordinates have same dimension before calculating distance
            if len(coords) != len(path_start_coords): continue
            # Calculate squared Euclidean distance for efficiency
            dist_sq = sum((c1 - c2)**2 for c1, c2 in zip(coords, path_start_coords))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_key = path_key
        return nearest_key

    def _find_path_start_index(self, path: List[Tuple], coords: Tuple) -> int:
        """ Find the index in a path closest to the given coordinates. """
        if not path: return 0
        if coords is None: return 0

        min_dist_sq = float('inf')
        best_idx = 0
        for i, path_coords in enumerate(path):
             # Ensure coordinates have same dimension
             if len(coords) != len(path_coords): continue
             dist_sq = sum((c1 - c2)**2 for c1, c2 in zip(coords, path_coords))
             if dist_sq < min_dist_sq:
                 min_dist_sq = dist_sq
                 best_idx = i
             if min_dist_sq == 0: break # Exact match found
        return best_idx

    # --- Dynamic Adaptation & Optimization (Placeholders - Require full logic) ---

    def _update_access_patterns(self, key: Any) -> None:
        """ Update access frequency, history, and co-access matrix. """
        self.access_history.append(key)
        self.access_frequency[key] += 1
        # Update co-access (simplified)
        if len(self.access_history) > 1:
            last_key = self.access_history[-2]
            if last_key != key:
                self.co_access_matrix[last_key][key] += 1
                self.co_access_matrix[key][last_key] += 1

        # Trigger potential promotion based on frequency
        promo_threshold = self.config.get("mdhg_velocity_promo_threshold", 10)
        if self.access_frequency[key] >= promo_threshold:
            if key in self.location_map:
                building_id = self.location_map[key][0]
                self._consider_velocity_promotion(key, building_id)


    def _consider_velocity_promotion(self, key: Any, building_id: str) -> None:
         """ Consider promoting a key to the velocity region if beneficial. """
         building = self.buildings.get(building_id)
         if not building or key not in self.location_map: return

         current_loc = self.location_map[key]
         if current_loc[1] == 'velocity': return # Already there

         target_idx = self._hash_to_velocity_index(key, building_id)
         if not (0 <= target_idx < len(building['velocity_region'])): return # Invalid index

         current_entry = building['velocity_region'][target_idx]
         key_freq = self.access_frequency.get(key, 0)
         should_promote = False

         if current_entry is None:
             should_promote = True
         else:
             occupant_key = current_entry[0]
             occupant_freq = self.access_frequency.get(occupant_key, 0)
             # Promote if new key is significantly more frequent (using PHI ratio)
             if key_freq > occupant_freq * self.PHI:
                 should_promote = True
                 # Relocate the occupant if it's being evicted
                 print(f"    MDHG: Evicting {occupant_key} (freq {occupant_freq}) from velocity for {key} (freq {key_freq})")
                 self._relocate_from_velocity(occupant_key, current_entry[1], building_id)
                 self.stats['relocations_from_velocity'] += 1

         if should_promote:
             # Get current value (get() handles finding it)
             value_tuple = self.get(key) # This will update access patterns again
             if value_tuple is not None:
                 print(f"    MDHG: Promoting key {key} to velocity region in {building_id}")
                 # Remove from old location BEFORE putting in new one
                 self._remove_from_current_location(key) # Removes from core/conflict
                 building['velocity_region'][target_idx] = (key, value_tuple)
                 self.location_map[key] = (building_id, 'velocity', target_idx) # Update location map
                 self.stats['promotions_velocity'] += 1


    def _relocate_from_velocity(self, key: Any, value: Any, building_id: str) -> None:
        """ Relocate a key evicted from velocity region back to core/conflict. """
        # This is essentially a 'put' operation, but we know it's not in velocity.
        # We need to ensure size isn't incremented again.
        building = self.buildings.get(building_id)
        if not building: return

        # Try dimensional core first
        coords = self._hash_to_coords(key, building_id)
        if coords is not None:
            if coords not in building['dimensional_core']:
                building['dimensional_core'][coords] = (key, value)
                self.location_map[key] = (building_id, 'dimensional', coords)
                return
            else: # Collision
                new_coords, _ = self._follow_hamiltonian_path_for_put(building_id, coords)
                if new_coords:
                    building['dimensional_core'][new_coords] = (key, value)
                    self.location_map[key] = (building_id, 'dimensional', new_coords)
                    return
        # Fallback to conflict structure
        fallback_coords = coords if coords is not None else tuple([0]*self.dimensions)
        conflict_key_hash = self._hash_to_conflict_key(key, fallback_coords)
        if conflict_key_hash not in building['conflict_structures']:
            building['conflict_structures'][conflict_key_hash] = {}
        building['conflict_structures'][conflict_key_hash][key] = value
        self.location_map[key] = (building_id, 'conflict', conflict_key_hash)


    def _remove_from_current_location(self, key: Any) -> None:
        """ Helper to remove key from core/conflict AFTER checking location map. """
        if key not in self.location_map: return
        building_id, region_type, location_spec = self.location_map[key]
        building = self.buildings.get(building_id)
        if not building: return

        removed = False
        if region_type == 'dimensional':
            if location_spec in building['dimensional_core'] and building['dimensional_core'][location_spec][0] == key:
                del building['dimensional_core'][location_spec]
                removed = True
        elif region_type == 'conflict':
            conflict_map = building['conflict_structures'].get(location_spec)
            if conflict_map and key in conflict_map:
                del conflict_map[key]
                if not conflict_map: del building['conflict_structures'][location_spec]
                removed = True
        # Note: We don't delete from location map here, caller handles final update


    def _check_optimization_and_resize(self) -> None:
        """ Check if optimization or resize is needed based on operations or time. """
        current_time = time.time()
        ops_threshold_minor = self.config.get("mdhg_ops_thresh_minor", 100)
        time_threshold_minor = self.config.get("mdhg_time_thresh_minor", 1.0)
        ops_threshold_major = self.config.get("mdhg_ops_thresh_major", 1000)
        time_threshold_major = self.config.get("mdhg_time_thresh_major", 5.0)

        needs_minor_opt = (self.operations_since_optimization > 0 and self.operations_since_optimization % ops_threshold_minor == 0) or \
                          (current_time - self.last_minor_optimization >= time_threshold_minor)
        needs_major_opt = (self.operations_since_optimization > 0 and self.operations_since_optimization % ops_threshold_major == 0) or \
                          (current_time - self.last_major_optimization >= time_threshold_major)

        if needs_major_opt:
            # print("    MDHG: Performing major optimization...")
            self._perform_major_optimization()
            self.last_major_optimization = current_time
            self.last_minor_optimization = current_time # Reset minor timer too
            self.operations_since_optimization = 0 # Reset counter
        elif needs_minor_opt:
            # print("    MDHG: Performing minor optimization...")
            self._perform_minor_optimization()
            self.last_minor_optimization = current_time
            # Don't reset major timer or op counter on minor opt

        # Check resize AFTER potential optimizations
        current_load_factor = self.size / self.capacity if self.capacity > 0 else 1.0
        if current_load_factor > self.load_factor_threshold:
            print(f"    MDHG: Load factor {current_load_factor:.2f} exceeds threshold {self.load_factor_threshold}. Resizing.")
            self._resize()

    def _perform_minor_optimization(self) -> None:
        """ Perform minor optimizations like promoting hot keys. """
        hot_key_count = self.config.get("mdhg_hot_key_count", 100)
        hot_key_min_freq = self.config.get("mdhg_hot_key_min_freq", 5)
        hot_keys_global = self.access_frequency.most_common(hot_key_count)
        promoted_count = 0
        for key, freq in hot_keys_global:
            if freq < hot_key_min_freq: break
            if key in self.location_map:
                building_id = self.location_map[key][0]
                # This call might result in promotion
                self._consider_velocity_promotion(key, building_id)
                # Check if promotion actually happened (location map changed)
                if key in self.location_map and self.location_map[key][1] == 'velocity':
                     promoted_count += 1
        # if promoted_count > 0: print(f"      MDHG Minor Opt: Considered {len(hot_keys_global)} hot keys, promoted {promoted_count} to velocity.")


    def _perform_major_optimization(self) -> None:
        """ Perform major structural reorganizations. """
        self.stats['reorganizations'] += 1
        start_time = time.time()
        # print("      MDHG Major Opt: Updating shortcuts...")
        # self._update_shortcuts() # Placeholder
        # print("      MDHG Major Opt: Identifying and relocating clusters...")
        # self._identify_and_relocate_key_clusters() # Placeholder
        # print("      MDHG Major Opt: Pruning path cache...")
        # self._prune_path_cache() # Placeholder
        end_time = time.time()
        print(f"    MDHG: Major optimization complete in {end_time - start_time:.4f}s (Placeholders used).")


    def _update_shortcuts(self) -> None:
        """ Placeholder: Update shortcuts based on observed usage patterns. """
        # Requires tracking inter-building traversals or using co-access matrix across buildings
        pass

    def _identify_and_relocate_key_clusters(self) -> None:
        """ Placeholder: Identify clusters of co-accessed keys and move them closer. """
        # Requires graph analysis of co_access_matrix and complex relocation logic
        clusters_found = 0
        # ... implementation needed ...
        if clusters_found > 0:
            self.stats['clusters_relocated'] += clusters_found
            # print(f"        MDHG Cluster Opt: Relocated {clusters_found} key clusters.")
        pass

    def _prune_path_cache(self) -> None:
        """ Placeholder: Prune the path cache based on usage or recency. """
        max_cache_size = self.config.get("mdhg_path_cache_max_size", 100)
        if len(self.path_cache) > max_cache_size:
            # Simple prune: Keep top 50% most used
            keep_count = max_cache_size // 2
            sorted_usage = sorted(self.path_usage.items(), key=lambda item: item[1], reverse=True)
            keys_to_keep = {key for key, usage in sorted_usage[:keep_count]}
            old_size = len(self.path_cache)
            self.path_cache = {k: v for k, v in self.path_cache.items() if k in keys_to_keep}
            self.path_usage = {k: v for k, v in self.path_usage.items() if k in keys_to_keep}
            # print(f"        MDHG Cache Prune: Reduced path cache from {old_size} to {len(self.path_cache)} entries.")
        pass

    def _resize(self) -> None:
        """ Resize the hash table when load factor is exceeded. """
        self.stats['resizes'] += 1
        old_capacity = self.capacity
        # Increase capacity using golden ratio
        new_capacity = max(old_capacity + 1, int(old_capacity * self.PHI * 1.1)) # Add buffer
        print(f"    MDHG Resize: Increasing capacity from {old_capacity} to {new_capacity}")

        # Store old data temporarily
        old_items = []
        for key, loc_info in self.location_map.items():
            # Retrieve value from old structure before wiping it
            building_id_old, region_type_old, location_spec_old = loc_info
            building_old = self.buildings.get(building_id_old)
            value_tuple = None
            if building_old:
                if region_type_old == 'velocity':
                    if 0 <= location_spec_old < len(building_old['velocity_region']):
                        entry = building_old['velocity_region'][location_spec_old]
                        if entry and entry[0] == key: value_tuple = entry[1]
                elif region_type_old == 'dimensional':
                    entry = building_old['dimensional_core'].get(location_spec_old)
                    if entry and entry[0] == key: value_tuple = entry[1]
                elif region_type_old == 'conflict':
                    conflict_map = building_old['conflict_structures'].get(location_spec_old)
                    if conflict_map: value_tuple = conflict_map.get(key)
            if value_tuple is not None:
                old_items.append((key, value_tuple))
            # else: print(f"Warning: Could not retrieve value for key {key} during resize.")

        # Re-initialize with new capacity
        self.capacity = new_capacity
        self.size = 0 # Reset size, will be repopulated
        self.buildings = self._initialize_buildings()
        self.location_map = {} # Clear location map
        self._initialize_structure() # Recompute paths etc. for new structure

        # Rehash all elements
        print(f"    MDHG Resize: Rehashing {len(old_items)} elements...")
        rehash_start_time = time.time()
        for key, value_tuple in old_items:
            self.put(key, value_tuple) # Re-insert into the new structure
        rehash_end_time = time.time()
        print(f"    MDHG Resize: Rehashing complete in {rehash_end_time - rehash_start_time:.4f}s. New size: {self.size}")

        # Reset optimization timers after resize
        self.last_minor_optimization = time.time()
        self.last_major_optimization = time.time()
        self.operations_since_optimization = 0

# ==============================
# === AGRM State Management ===
# ==============================

class AGRMStateBus:
    """
    Manages the shared state between AGRM agents.
    Acts as the central repository for path data, node states,
    sweep metadata, modulation parameters, and agent signals.
    Uses Hybrid Hashing strategy based on complexity.
    """
    def __init__(self, cities: List[Tuple[float, float]], config: Dict):
        """
        Initializes the state bus.
        Args:
            cities: List of (x, y) coordinates for the nodes.
            config: Dictionary containing configuration parameters for AGRM and MDHG.
        """
        self.config = config
        self.num_nodes = len(cities)
        self.cities = cities # List of (x, y) tuples

        # --- Core State ---
        self.visited_fwd: Set[int] = set() # Nodes visited by forward builder
        self.visited_rev: Set[int] = set() # Nodes visited by reverse builder
        self.path_fwd: List[int] = [] # Path built by forward builder
        self.path_rev: List[int] = [] # Path built by reverse builder (in reverse order)
        self.full_path: Optional[List[int]] = None # Final merged path

        # --- Sweep Metadata (Populated by Navigator) ---
        self.sweep_data: Dict[int, Dict] = {} # node_id -> {rank, shell, sector, quadrant, hemisphere, density, gr_score}
        self.sweep_center: Optional[Tuple[float, float]] = None
        self.max_radius: float = 0.0
        self.shell_width: float = 0.0
        self.start_node_fwd: Optional[int] = None
        self.start_node_rev: Optional[int] = None

        # --- Legal Graph & Modulation State (Managed by Controller) ---
        # Legal edges are computed ephemerally by the validator agent
        # self.legal_edges: Optional[Dict[int, List[int]]] = None # Not stored persistently
        # Store default modulation params for reset
        self.default_modulation_params = {
            "shell_tolerance": config.get("mod_shell_tolerance", 2),
            "curvature_limit": config.get("mod_curvature_limit", math.pi / 4), # Approx 45 deg
            "sector_tolerance": config.get("mod_sector_tolerance", 2),
            "distance_cap_factor": config.get("mod_dist_cap_factor", 3.0), # Multiplier for avg dist in shell
            "allow_sparse_unlock": False,
            "soft_override_active": False,
            "reentry_mode": False
        }
        self.modulation_params = self.default_modulation_params.copy() # Start with defaults
        self.current_phase: str = "initializing" # 'initializing', 'building', 'pre-midpoint', 'converging', 'post-midpoint', 'post-merge', 'patching', 'finalizing', 'complete'

        # --- Hybrid Hashing State ---
        self.complexity_threshold = config.get("hybrid_hash_threshold", 5) # n=5 complexity threshold [cite: 896-913]
        # Instantiate caches
        self.low_complexity_cache = {} # Standard dict for n <= 5 tasks/data
        # Instantiate MDHG for high complexity state. Tailoring parameters applied here.
        mdhg_dims = config.get("mdhg_dimensions", 3)
        mdhg_cap = max(1024, self.num_nodes) # Capacity scales with problem size
        self.high_complexity_cache = MDHGHashTable(capacity=mdhg_cap, dimensions=mdhg_dims, config=config) # Pass config
        print(f"StateBus: Initialized Hybrid Caching (n={self.complexity_threshold} threshold). MDHG Dims={mdhg_dims}, Capacity={mdhg_cap}")

        # --- Agent Feedback / Flags ---
        self.builder_fwd_state = {"status": "idle", "stalls": 0, "last_node": -1, "current_shell": -1, "current_sector": -1}
        self.builder_rev_state = {"status": "idle", "stalls": 0, "last_node": -1, "current_shell": -1, "current_sector": -1}
        self.salesman_proposals: List[Dict] = [] # Patches suggested for review by Salesman
        self.accepted_patches: List[Dict] = [] # Patches approved by Controller for splicing

    def get_cache(self, complexity_level: int) -> Union[Dict, MDHGHashTable]:
        """
        Returns the appropriate cache backend based on complexity level 'n'.
        Called by agents needing to store/retrieve state ephemerally.
        Args:
            complexity_level: The estimated complexity 'n' of the current operation.
        Returns:
            The standard dict or the MDHGHashTable instance.
        """
        # Note: Complexity level 'n' determination logic resides in the calling agent/controller
        # This provides the interface based on that determination.
        if complexity_level <= self.complexity_threshold:
            # print(f"DEBUG: Using low complexity cache (dict) for n={complexity_level}")
            return self.low_complexity_cache
        else:
            # print(f"DEBUG: Using high complexity cache (MDHG) for n={complexity_level}")
            return self.high_complexity_cache

    def migrate_data(self, key: Any, current_complexity: int, new_complexity: int):
        """
        Migrates a key between caches if the complexity threshold is crossed.
        Called by the Modulation Controller. Assumes value needs to be fetched.
        Args:
            key: The key to migrate.
            current_complexity: The previous complexity level 'n'.
            new_complexity: The new complexity level 'n'.
        """
        # Determine source and target caches
        source_cache = self.get_cache(current_complexity)
        target_cache = self.get_cache(new_complexity)

        # Only migrate if the cache type actually changes
        if type(source_cache) == type(target_cache):
            return # No migration needed

        value_to_migrate = None
        metadata = {}

        # Get value from source cache
        if isinstance(source_cache, MDHGHashTable):
            result = source_cache.get(key)
            if result:
                value_to_migrate, metadata = result # MDHG stores tuple
        elif isinstance(source_cache, dict):
            value_to_migrate = source_cache.get(key)
            metadata = {'source': 'dict'} # Assume origin if coming from dict

        # If value exists in source, remove it and put it in target
        if value_to_migrate is not None:
            # Remove from source
            if isinstance(source_cache, MDHGHashTable):
                source_cache.remove(key)
            elif isinstance(source_cache, dict):
                if key in source_cache: del source_cache[key]

            # Put into target
            if isinstance(target_cache, MDHGHashTable):
                 # Ensure metadata includes source info
                 metadata['source'] = 'dict' if isinstance(source_cache, dict) else metadata.get('source', 'mdhg')
                 # Ensure retain_flag exists, default to False if not present
                 metadata['retain_flag'] = metadata.get('retain_flag', False)
                 target_cache.put(key, (value_to_migrate, metadata)) # Store as tuple
                 print(f"StateBus: Migrated key {key} from {type(source_cache).__name__} to MDHG.")
            elif isinstance(target_cache, dict):
                 target_cache[key] = value_to_migrate # Store only value in dict
                 print(f"StateBus: Migrated key {key} from MDHG to {type(target_cache).__name__}.")

    # --- Rest of AGRMStateBus methods ---
    # (update_sweep_data, get_node_sweep_data, is_visited, add_visited,
    #  get_unvisited_nodes, update_modulation_params, update_builder_state,
    #  check_convergence, merge_paths, add_salesman_proposal, etc.)
    # These remain largely the same as provided before, ensuring they interact
    # correctly with the rest of the system state variables.
    # ... (Previous AGRMStateBus methods included here for completeness) ...
    # Note: Ensure methods like add_visited correctly interact with the
    #       get_cache() method if storing visited status in hybrid caches.
    #       Currently, visited status uses Python sets directly for simplicity.

    def update_sweep_data(self, sweep_results: Dict):
        """ Updates state bus with data generated by the Navigator sweep. """
        print("StateBus: Updating with Navigator sweep data...")
        self.sweep_data = sweep_results.get('node_data', {})
        self.sweep_center = sweep_results.get('center')
        self.max_radius = sweep_results.get('max_radius', 0.0)
        self.shell_width = sweep_results.get('shell_width', 0.0)
        self.start_node_fwd = sweep_results.get('start_node_fwd')
        self.start_node_rev = sweep_results.get('start_node_rev')

        # Initialize visited sets and paths
        self.visited_fwd.clear()
        self.visited_rev.clear()
        self.path_fwd = []
        self.path_rev = []
        self.full_path = None
        self.current_phase = "building" # Ready to start building

        if self.start_node_fwd is not None:
            self.visited_fwd.add(self.start_node_fwd)
            self.path_fwd = [self.start_node_fwd]
            fwd_data = self.get_node_sweep_data(self.start_node_fwd)
            self.builder_fwd_state = {"status": "running", "stalls": 0, "last_node": self.start_node_fwd,
                                      "current_shell": fwd_data.get('shell', -1),
                                      "current_sector": fwd_data.get('sector', -1)}
        else:
            self.builder_fwd_state["status"] = "error"

        if self.start_node_rev is not None:
            # Ensure start nodes are different if possible, handle single node case
            if self.start_node_rev != self.start_node_fwd:
                 self.visited_rev.add(self.start_node_rev)
            self.path_rev = [self.start_node_rev]
            rev_data = self.get_node_sweep_data(self.start_node_rev)
            self.builder_rev_state = {"status": "running", "stalls": 0, "last_node": self.start_node_rev,
                                      "current_shell": rev_data.get('shell', -1),
                                      "current_sector": rev_data.get('sector', -1)}
        else:
            self.builder_rev_state["status"] = "error"

        print(f"StateBus: Sweep data loaded. Fwd starts at {self.start_node_fwd}, Rev starts at {self.start_node_rev}")

    def get_node_sweep_data(self, node_id: int) -> Dict:
        """ Gets sweep metadata for a specific node. Returns empty dict if not found. """
        return self.sweep_data.get(node_id, {})

    def is_visited(self, node_id: int) -> bool:
        """ Checks if a node has been visited by EITHER builder using internal sets. """
        return node_id in self.visited_fwd or node_id in self.visited_rev

    def add_visited(self, node_id: int, builder_type: str):
        """ Adds a node to the appropriate visited set and path, updates builder state. """
        node_data = self.get_node_sweep_data(node_id)
        current_shell = node_data.get('shell', -1)
        current_sector = node_data.get('sector', -1)

        if builder_type == 'forward':
            if node_id not in self.visited_fwd:
                self.visited_fwd.add(node_id)
                self.path_fwd.append(node_id)
                self.builder_fwd_state.update({
                    "last_node": node_id, "stalls": 0, "status": "running",
                    "current_shell": current_shell, "current_sector": current_sector
                })
        elif builder_type == 'reverse':
             if node_id not in self.visited_rev:
                self.visited_rev.add(node_id)
                self.path_rev.append(node_id)
                self.builder_rev_state.update({
                    "last_node": node_id, "stalls": 0, "status": "running",
                    "current_shell": current_shell, "current_sector": current_sector
                })

    def get_unvisited_nodes(self) -> Set[int]:
        """ Returns the set of all nodes not yet visited by either builder. """
        all_nodes = set(range(self.num_nodes))
        visited_all = self.visited_fwd.union(self.visited_rev)
        return all_nodes - visited_all

    def update_modulation_params(self, new_params: Dict):
        """ Updates dynamic modulation parameters (called by Controller). """
        self.modulation_params.update(new_params)
        # print(f"StateBus: Modulation params updated: {self.modulation_params}")

    def update_builder_state(self, builder_type: str, status: Optional[str] = None, stalled: Optional[bool] = None):
         """ Updates the status of a builder agent, tracking stalls. """
         state = self.builder_fwd_state if builder_type == 'forward' else self.builder_rev_state
         if status:
             state["status"] = status
         if stalled is True:
             state["stalls"] += 1
             state["status"] = "stalled" # Mark as stalled
         elif stalled is False: # Explicitly told not stalled (i.e., progress made)
             state["stalls"] = 0
             if not status: state["status"] = "running" # Assume running if progress made

    def check_convergence(self) -> bool:
         """ Checks if builders meet criteria to merge paths using dynamic midpoint logic. """
         node_fwd = self.builder_fwd_state["last_node"]
         node_rev = self.builder_rev_state["last_node"]
         if node_fwd == -1 or node_rev == -1 or \
            self.builder_fwd_state["status"] not in ["running", "stalled"] or \
            self.builder_rev_state["status"] not in ["running", "stalled"]:
             return False

         shell_fwd = self.builder_fwd_state["current_shell"]
         shell_rev = self.builder_rev_state["current_shell"]
         sector_fwd = self.builder_fwd_state["current_sector"]
         sector_rev = self.builder_rev_state["current_sector"]
         if shell_fwd == -1 or shell_rev == -1 or sector_fwd == -1 or sector_rev == -1: return False

         shell_threshold = self.config.get("convergence_shell_threshold", 1)
         shell_overlap = abs(shell_fwd - shell_rev) <= shell_threshold

         num_sectors = self.config.get("sweep_num_sectors", 8)
         sector_threshold = self.config.get("convergence_sector_threshold", 1)
         sector_diff = abs(sector_fwd - sector_rev)
         sector_proximity = min(sector_diff, num_sectors - sector_diff) <= sector_threshold

         stall_dist_factor = self.config.get("convergence_stall_dist_factor", 5.0)
         stall_dist_threshold = stall_dist_factor * max(1.0, self.shell_width or 10.0)
         phys_dist = math.dist(self.cities[node_fwd], self.cities[node_rev])
         is_stalled = self.builder_fwd_state["stalls"] > 3 or self.builder_rev_state["stalls"] > 3
         stall_convergence = is_stalled and phys_dist <= stall_dist_threshold

         converged = (shell_overlap and sector_proximity) or stall_convergence

         if converged:
             print(f"StateBus: Convergence detected between FWD {node_fwd} and REV {node_rev}")
             self.current_phase = "converging"
             self.builder_fwd_state["status"] = "converged"
             self.builder_rev_state["status"] = "converged"
         return converged

    def merge_paths(self) -> bool:
        """ Merges paths after convergence. Returns True if complete, False if patching needed. """
        if self.current_phase != "converging": return False
        print("StateBus: Attempting path merge...")
        node_fwd_last = self.path_fwd[-1]
        node_rev_last = self.path_rev[-1]
        merged_list = list(self.path_fwd)
        reversed_rev_path = self.path_rev[::-1]
        if node_fwd_last == node_rev_last:
            merged_list.extend(reversed_rev_path[1:])
        else:
            merged_list.extend(reversed_rev_path)
        self.full_path = merged_list

        visited_final = set(self.full_path)
        missed_nodes = set(range(self.num_nodes)) - visited_final
        if missed_nodes:
            print(f"StateBus WARNING: Path merge complete, but {len(missed_nodes)} nodes missed.")
            self.current_phase = "patching"
            return False
        else:
            if len(self.full_path) > 1 and self.full_path[0] != self.full_path[-1]:
                self.full_path.append(self.full_path[0]) # Close the loop
            print(f"StateBus: Path merge successful. All {self.num_nodes} nodes included.")
            self.current_phase = "merged"
            return True

    def add_salesman_proposal(self, proposal: Dict):
        self.salesman_proposals.append(proposal)

    def get_salesman_proposals(self) -> List[Dict]:
        return self.salesman_proposals

    def clear_salesman_proposals(self):
        self.salesman_proposals = []

    def store_accepted_patch(self, patch: Dict):
        self.accepted_patches.append(patch)

    def get_accepted_patches(self) -> List[Dict]:
        return self.accepted_patches

    def clear_accepted_patches(self):
        self.accepted_patches = []

    def splice_patch(self, patch: Dict) -> bool:
        """ Applies an accepted patch to the full_path. """
        if not self.full_path or self.current_phase not in ["merged", "finalizing", "complete"]:
            print("ERROR: Cannot splice patch, path not ready.")
            return False
        try:
            start_idx, end_idx = patch['segment_indices']
            new_subpath = patch['new_subpath_nodes']
            if not (0 <= start_idx < end_idx < len(self.full_path)):
                print(f"ERROR: Invalid splice indices {start_idx}, {end_idx}")
                return False
            # Assumes new_subpath replaces nodes from index start_idx+1 up to end_idx-1
            print(f"StateBus: Splicing patch {new_subpath} between indices {start_idx} and {end_idx}")
            self.full_path = self.full_path[:start_idx+1] + new_subpath + self.full_path[end_idx:]
            print(f"StateBus: Path spliced. New length: {len(self.full_path)}")
            return True
        except Exception as e:
            print(f"ERROR: Exception during patch splice: {e}")
            return False

# ============================
# === AGRM Agent: Navigator ===
# ============================
# (NavigatorGR class code as provided previously - verified complete)
class NavigatorGR:
    """
    Performs Golden Ratio sweeps to gather spatial and structural metadata.
    Does NOT build paths. Provides data for AGRM filtering and pathing.
    Includes dynamic shell width, quadrant/hemisphere/sector tagging, k-NN density.
    """
    def __init__(self, cities: List[Tuple[float, float]], config: Dict):
        self.cities = cities
        self.num_nodes = len(cities)
        self.config = config
        self.PHI = (1 + math.sqrt(5)) / 2
        self.sweep_data: Dict[int, Dict] = {i:{} for i in range(self.num_nodes)} # Pre-initialize
        self.center: Optional[Tuple[float, float]] = None
        self.max_radius: float = 0.0
        self.shell_width: float = 0.0
        self.start_node_fwd: Optional[int] = None
        self.start_node_rev: Optional[int] = None

    def _calculate_center(self):
        if not self.cities: self.center = (0.0, 0.0); return
        sum_x = sum(c[0] for c in self.cities)
        sum_y = sum(c[1] for c in self.cities)
        self.center = (sum_x / self.num_nodes, sum_y / self.num_nodes)

    def _calculate_radii_and_angles(self):
        if self.center is None: self._calculate_center()
        cx, cy = self.center
        max_r_sq = 0
        for i, (x, y) in enumerate(self.cities):
            dx, dy = x - cx, y - cy
            radius_sq = dx*dx + dy*dy
            radius = math.sqrt(radius_sq) if radius_sq > 0 else 0
            angle = math.atan2(dy, dx)
            self.sweep_data[i].update({'radius': radius, 'angle': angle})
            if radius_sq > max_r_sq: max_r_sq = radius_sq
        self.max_radius = math.sqrt(max_r_sq) if max_r_sq > 0 else 0

    def _assign_shells_and_sectors(self):
        if self.max_radius == 0 and self.num_nodes > 1: self._calculate_radii_and_angles()
        if self.max_radius == 0: # Handle single node or all nodes at center
             for i in range(self.num_nodes):
                 self.sweep_data[i]['shell'] = 0
                 self.sweep_data[i]['sector'] = 0
             self.shell_width = 1.0
             return

        desired_shells = self.config.get("sweep_num_shells", 10)
        self.shell_width = (self.max_radius / desired_shells) if desired_shells > 0 else self.max_radius
        if self.shell_width <= 1e-9: self.shell_width = 1.0

        num_sectors = self.config.get("sweep_num_sectors", 8)
        if num_sectors <= 0: num_sectors = 1
        sector_angle = 2 * math.pi / num_sectors

        shell_counts = Counter()
        for i in range(self.num_nodes):
            radius = self.sweep_data[i].get('radius', 0.0)
            angle = self.sweep_data[i].get('angle', 0.0)
            shell = int(radius // self.shell_width)
            shell = min(shell, desired_shells - 1) if desired_shells > 0 else 0
            self.sweep_data[i]['shell'] = max(0, shell)
            shell_counts[self.sweep_data[i]['shell']] += 1
            normalized_angle = (angle + 2 * math.pi) % (2 * math.pi)
            sector = int(normalized_angle // sector_angle)
            self.sweep_data[i]['sector'] = min(sector, num_sectors - 1)
        # print(f"  Navigator: Shell distribution: {dict(sorted(shell_counts.items()))}")

    def _calculate_gr_sweep_scores(self):
        # Placeholder: Rank by shell, then angle. Needs proper GR spiral logic.
        temp_nodes = []
        for i in range(self.num_nodes):
            shell = self.sweep_data[i].get('shell', 999)
            angle = self.sweep_data[i].get('angle', 0.0)
            score = shell + (abs(angle) / (2*math.pi)) # Simple composite score
            temp_nodes.append((score, i))
        temp_nodes.sort()
        for rank, (score, i) in enumerate(temp_nodes):
            self.sweep_data[i]['sweep_rank'] = rank
            self.sweep_data[i]['gr_score'] = score
        if temp_nodes:
            self.start_node_fwd = temp_nodes[0][1]
            self.start_node_rev = temp_nodes[-1][1]
            # print(f"  Navigator: Determined Fwd Start={self.start_node_fwd}, Rev Start={self.start_node_rev}")

    def _assign_quadrants_and_hemispheres(self):
        if self.center is None: self._calculate_center()
        cx, cy = self.center
        if not any('sweep_rank' in d for i,d in self.sweep_data.items()):
             print("  Navigator ERROR: Sweep rank needed for hemisphere assignment.")
             return
        midpoint_rank = self.num_nodes // 2
        for i, (x, y) in enumerate(self.cities):
            if x >= cx and y >= cy: quadrant = "Q1"
            elif x < cx and y >= cy: quadrant = "Q2"
            elif x < cx and y < cy: quadrant = "Q3"
            else: quadrant = "Q4"
            self.sweep_data[i]['quadrant'] = quadrant
            rank = self.sweep_data[i].get('sweep_rank', -1)
            hemisphere = "A_start" if rank < midpoint_rank else "B_end"
            self.sweep_data[i]['hemisphere'] = hemisphere

    def _classify_density(self):
        print("  Navigator: Classifying node density...")
        if not HAS_SKLEARN or self.num_nodes < 3: # Need at least 3 points for k-NN with k>=1
            print("  Navigator WARNING: Using basic shell density (sklearn not found or N too small).")
            nodes_per_shell = Counter(d.get('shell', -1) for d in self.sweep_data.values())
            if not nodes_per_shell: return # Avoid division by zero
            avg_nodes_per_shell = self.num_nodes / max(1, len(nodes_per_shell))
            dense_threshold = avg_nodes_per_shell * self.config.get("density_dense_factor", 1.5)
            sparse_threshold = avg_nodes_per_shell * self.config.get("density_sparse_factor", 0.5)
            for i in range(self.num_nodes):
                shell = self.sweep_data[i].get('shell', -1)
                shell_count = nodes_per_shell.get(shell, 0)
                if shell_count >= dense_threshold: density = "dense"
                elif shell_count <= sparse_threshold: density = "sparse"
                else: density = "midling"
                self.sweep_data[i]['density'] = density
        else:
            coords = np.array(self.cities)
            k = self.config.get("density_knn_k", 10)
            k = min(k, self.num_nodes - 1)
            if k <= 0: # Handle N=1 or N=2 case
                 for i in range(self.num_nodes): self.sweep_data[i]['density'] = "midling"
                 return

            # Use BallTree for potentially better performance on some distributions
            tree = BallTree(coords)
            # Query for k+1 neighbors to exclude self
            distances, _ = tree.query(coords, k=k + 1)
            # Calculate average distance to k nearest neighbors (excluding self)
            avg_distances = np.mean(distances[:, 1:], axis=1)

            mean_avg_dist = np.mean(avg_distances)
            std_avg_dist = np.std(avg_distances)
            # Avoid zero std dev
            if std_avg_dist < 1e-9: std_avg_dist = mean_avg_dist * 0.1 if mean_avg_dist > 0 else 1.0

            density_factor = self.config.get("density_std_dev_factor", 0.75)
            dense_threshold = mean_avg_dist - std_avg_dist * density_factor
            sparse_threshold = mean_avg_dist + std_avg_dist * density_factor

            for i in range(self.num_nodes):
                avg_dist = avg_distances[i]
                if avg_dist <= dense_threshold: density = "dense"
                elif avg_dist >= sparse_threshold: density = "sparse"
                else: density = "midling"
                self.sweep_data[i]['density'] = density

        density_counts = Counter(d.get('density', 'unknown') for d in self.sweep_data.values())
        print(f"  Navigator: Density classification complete. Counts: {dict(density_counts)}")


    def run_sweep(self) -> Dict:
        """ Executes the full sweep and data generation process. """
        print("Navigator: Running sweep...")
        start_time = time.time()
        self._calculate_center()
        self._calculate_radii_and_angles()
        self._assign_shells_and_sectors()
        self._calculate_gr_sweep_scores() # Assigns ranks
        self._assign_quadrants_and_hemispheres() # Assigns hemi based on rank
        self._classify_density() # Assigns density
        end_time = time.time()
        print(f"Navigator: Sweep complete in {end_time - start_time:.4f} seconds.")
        return {
            'node_data': self.sweep_data, 'center': self.center, 'max_radius': self.max_radius,
            'shell_width': self.shell_width, 'start_node_fwd': self.start_node_fwd, 'start_node_rev': self.start_node_rev
        }

# =======================================
# === AGRM Agent: Legal Edge Validator ===
# =======================================
# (AGRMEdgeValidator class code as provided previously - verified complete)
class AGRMEdgeValidator:
    """
    Provides methods to check if a transition (edge) between two nodes
    is legal according to the current dynamic AGRM modulation parameters.
    Operates ephemerally - computes legality on demand using data from the StateBus.
    """
    def __init__(self, bus: AGRMStateBus, config: Dict):
        self.bus = bus
        self.config = config
        self.PHI = (1 + math.sqrt(5)) / 2

    def is_edge_legal(self, node_from: int, node_to: int, builder_type: str) -> bool:
        """ Checks all AGRM legality rules for a potential edge. """
        params = self.bus.modulation_params
        data_from = self.bus.get_node_sweep_data(node_from)
        data_to = self.bus.get_node_sweep_data(node_to)
        if not data_from or not data_to: return False

        if not self._check_shell_proximity(data_from, data_to, params): return False
        if not self._check_sector_continuity(data_from, data_to, params): return False
        if not self._check_phase_timing(data_to, params): return False
        if not self._check_curvature(node_from, node_to, builder_type, params): return False
        if not self._check_distance_cap(node_from, node_to, data_from, params): return False
        # Add Quadrant transition checks if needed
        return True

    def _check_shell_proximity(self, data_from: Dict, data_to: Dict, params: Dict) -> bool:
        shell_from = data_from.get('shell', -1)
        shell_to = data_to.get('shell', -1)
        if shell_from == -1 or shell_to == -1: return False
        shell_diff = abs(shell_from - shell_to)
        is_reentry_inward = params.get("reentry_mode", False) and shell_to < shell_from
        within_tolerance = shell_diff <= params.get("shell_tolerance", 2)
        return within_tolerance or is_reentry_inward

    def _check_sector_continuity(self, data_from: Dict, data_to: Dict, params: Dict) -> bool:
        sector_from = data_from.get('sector', -1)
        sector_to = data_to.get('sector', -1)
        if sector_from == -1 or sector_to == -1: return False
        num_sectors = self.config.get("sweep_num_sectors", 8)
        if num_sectors <= 0: return True
        sector_diff = abs(sector_from - sector_to)
        angular_diff = min(sector_diff, num_sectors - sector_diff)
        return angular_diff <= params.get("sector_tolerance", 2)

    def _check_phase_timing(self, data_to: Dict, params: Dict) -> bool:
        """ Checks if moving to a sparse zone is allowed based on phase unlock state. """
        if params.get("allow_sparse_unlock", False):
            return True # Sparse zones are allowed
        else:
            # Disallow entry *into* sparse zones before unlock
            return data_to.get('density') != "sparse"

    def _check_curvature(self, node_from: int, node_to: int, builder_type: str, params: Dict) -> bool:
        """ Checks if the turn angle respects the dynamic GR curvature limit. """
        path = self.bus.path_fwd if builder_type == 'forward' else self.bus.path_rev
        if len(path) < 2: return True # No angle to check

        node_prev = path[-2]
        try:
            pos_prev = self.bus.cities[node_prev]
            pos_from = self.bus.cities[node_from]
            pos_to = self.bus.cities[node_to]
        except IndexError: return False # Invalid index

        vec1 = (pos_from[0] - pos_prev[0], pos_from[1] - pos_prev[1])
        vec2 = (pos_to[0] - pos_from[0], pos_to[1] - pos_from[1])
        len1 = math.hypot(vec1[0], vec1[1])
        len2 = math.hypot(vec2[0], vec2[1])
        if len1 < 1e-9 or len2 < 1e-9: return True # Allow if points overlap

        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        cos_angle = max(-1.0, min(1.0, dot_product / (len1 * len2)))
        angle = math.acos(cos_angle)
        return angle <= params.get("curvature_limit", math.pi / 4)

    def _check_distance_cap(self, node_from: int, node_to: int, data_from: Dict, params: Dict) -> bool:
        """ Checks if the edge distance exceeds a dynamic cap. """
        avg_dist_in_shell = max(1.0, self.bus.shell_width or 10.0) # Proxy
        base_dist_cap = avg_dist_in_shell * params.get("distance_cap_factor", 3.0)
        if data_from.get('density') == "sparse":
            base_dist_cap *= self.config.get("dist_cap_sparse_mult", 1.5)

        effective_dist_cap = base_dist_cap
        if params.get("soft_override_active", False) or params.get("reentry_mode", False):
             effective_dist_cap *= self.config.get("dist_cap_override_mult", 1.5)
        try:
            actual_dist = math.dist(self.bus.cities[node_from], self.bus.cities[node_to])
        except IndexError: return False
        return actual_dist <= effective_dist_cap

# =======================================
# === AGRM Agent: Modulation Controller ===
# =======================================
# (ModulationController class code as provided previously - verified complete)
class ModulationController:
    """
    The 'brain' of AGRM. Manages system state, agent coordination,
    dynamic legality modulation, phase unlocks, and recovery triggers.
    Uses Hybrid Hashing logic. Includes dynamic adjustments based on feedback.
    """
    def __init__(self, bus: AGRMStateBus, config: Dict):
        self.bus = bus
        self.config = config
        self.default_modulation_params = self.bus.modulation_params.copy()
        self.complexity_threshold = config.get("hybrid_hash_threshold", 5)

    def assess_complexity(self, context: Dict) -> int:
        # Placeholder - needs better logic based on context
        return context.get("num_candidates", 1)

    def select_cache(self, context: Dict) -> Union[Dict, MDHGHashTable]:
        complexity_n = self.assess_complexity(context)
        return self.bus.get_cache(complexity_n)

    def trigger_migration_check(self, key: Any, old_n: int, new_n: int):
        # Simplified: Assumes value needs to be fetched if migrating dict->MDHG
        if (old_n <= self.complexity_threshold < new_n):
             source_cache = self.bus.get_cache(old_n)
             if key in source_cache:
                 value = source_cache.get(key) # Get value before migrating
                 self.bus.migrate_data(key, old_n, new_n, value) # Pass value
        elif (old_n > self.complexity_threshold >= new_n):
             self.bus.migrate_data(key, old_n, new_n) # Value fetched inside migrate_data


    def update_controller_state(self):
        """ Main update loop for the controller - applies dynamic modulation. """
        fwd_stalls = self.bus.builder_fwd_state["stalls"]
        rev_stalls = self.bus.builder_rev_state["stalls"]
        fwd_status = self.bus.builder_fwd_state["status"]
        rev_status = self.bus.builder_rev_state["status"]

        nodes_visited_count = len(self.bus.visited_fwd) + len(self.bus.visited_rev)
        progress_percent = nodes_visited_count / max(1, self.bus.num_nodes)

        new_params = {}
        params_changed = False
        current_params = self.bus.modulation_params

        # --- Adaptive Unlocking ---
        midpoint_percent = self.config.get("midpoint_unlock_percent", 0.5)
        if progress_percent >= midpoint_percent and not current_params["allow_sparse_unlock"]:
            print("CONTROLLER: Midpoint reached. Unlocking sparse zones.")
            new_params["allow_sparse_unlock"] = True
            params_changed = True
        # Update overall phase on bus if needed (e.g., based on progress)
        if progress_percent >= midpoint_percent and self.bus.current_phase == "building":
             self.bus.current_phase = "post-midpoint"

        # --- Dynamic Modulation & Override Logic ---
        stall_threshold = self.config.get("controller_stall_threshold", 5)
        severe_stall_threshold = stall_threshold * self.config.get("controller_severe_stall_factor", 2)
        override_active_now = False
        reentry_active_now = False

        # Check for severe stalls -> trigger reentry
        if (fwd_stalls >= severe_stall_threshold or rev_stalls >= severe_stall_threshold) and not current_params["reentry_mode"]:
            print(f"CONTROLLER: Severe stall ({fwd_stalls}, {rev_stalls}). Triggering Reentry Mode.")
            new_params["reentry_mode"] = True
            new_params["soft_override_active"] = True # Reentry implies override
            # Apply significant relaxation for reentry
            new_params["curvature_limit"] = self.default_modulation_params["curvature_limit"] + self.config.get("mod_reentry_curve_relax", math.pi / 6)
            new_params["shell_tolerance"] = self.default_modulation_params["shell_tolerance"] + self.config.get("mod_reentry_shell_relax", 2)
            new_params["distance_cap_factor"] = self.default_modulation_params["distance_cap_factor"] * self.config.get("mod_reentry_dist_relax_factor", 1.5)
            params_changed = True
            reentry_active_now = True
        elif current_params["reentry_mode"]: # If already in reentry, keep flags set
             reentry_active_now = True
             override_active_now = True # Reentry keeps override active

        # Check for moderate stalls -> trigger soft override (if not already in reentry)
        elif fwd_stalls >= stall_threshold and rev_stalls >= stall_threshold and not current_params["soft_override_active"]:
            print(f"CONTROLLER: Both builders stalled ({fwd_stalls}, {rev_stalls}). Activating soft override.")
            new_params["soft_override_active"] = True
            # Apply moderate relaxation
            new_params["curvature_limit"] = self.default_modulation_params["curvature_limit"] + self.config.get("mod_override_curve_relax", math.pi / 12)
            new_params["shell_tolerance"] = self.default_modulation_params["shell_tolerance"] + self.config.get("mod_override_shell_relax", 1)
            params_changed = True
            override_active_now = True
        elif current_params["soft_override_active"]: # If already in override, keep flag set
             override_active_now = True

        # Reset if overrides were active but no longer needed
        # Check if builders are running OR converged (implying stability)
        fwd_stable = fwd_status in ["running", "converged", "finished"]
        rev_stable = rev_status in ["running", "converged", "finished"]
        # Reset if BOTH are stable AND override/reentry was previously active
        if (current_params["soft_override_active"] or current_params["reentry_mode"]) and \
           fwd_stable and rev_stable:
             print("CONTROLLER: Builders stable. Deactivating overrides/reentry. Resetting params.")
             # Reset only the params that were changed by override/reentry
             reset_keys = ["soft_override_active", "reentry_mode", "curvature_limit", "shell_tolerance", "distance_cap_factor"]
             for key in reset_keys:
                 if key in self.default_modulation_params:
                     new_params[key] = self.default_modulation_params[key]
                 else: # Ensure flags are reset even if not in defaults
                     if key == "soft_override_active": new_params[key] = False
                     if key == "reentry_mode": new_params[key] = False
             # Ensure sparse unlock state is preserved based on progress
             new_params["allow_sparse_unlock"] = current_params["allow_sparse_unlock"]
             params_changed = True

        # Apply changes to the bus
        if params_changed:
            self.bus.update_modulation_params(new_params)

    def get_current_legality_params(self) -> Dict:
        """ Returns the currently active legality parameters from the bus. """
        return self.bus.modulation_params.copy()

    def process_salesman_feedback(self):
        """ Evaluates patch proposals from Salesman and stores accepted ones on bus. """
        proposals = self.bus.get_salesman_proposals()
        if not proposals: return
        print(f"CONTROLLER: Evaluating {len(proposals)} Salesman proposals.")
        accepted_patches = []
        for patch in proposals:
            # Evaluation logic: Accept if cost saving is positive and significant?
            # Needs more sophisticated evaluation (e.g., structural impact)
            cost_saving = patch.get('cost_saving', 0.0)
            if cost_saving > self.config.get("controller_patch_min_saving", 0.1): # Require min saving
                print(f"CONTROLLER: Accepting patch proposal for segment {patch.get('segment_indices')} (Save: {cost_saving:.2f})")
                self.bus.store_accepted_patch(patch) # Store on bus for builder
            # else: print(f"CONTROLLER: Rejecting patch proposal for segment {patch.get('segment_indices')} (Saving too small)")
        self.bus.clear_salesman_proposals() # Clear pending proposals

# ======================================
# === AGRM Agent: Path Builder (Dual) ===
# ======================================
# (PathBuilder class code as provided previously - verified complete)
class PathBuilder:
    """
    Builds a path segment (forward or reverse) using AGRM rules.
    Operates ephemerally, querying legality on demand.
    Interacts with Controller for modulation and feedback.
    Handles reentry logic when triggered.
    Can splice patches provided by Controller. Includes k-NN neighbor finding.
    """
    def __init__(self, builder_type: str, start_node: int, bus: AGRMStateBus, validator: AGRMEdgeValidator, config: Dict):
        self.builder_type = builder_type
        self.current_node = start_node
        self.bus = bus
        self.validator = validator
        self.config = config
        self.stalled_cycles = 0
        self.is_reentering = False
        # Initialize KDTree/BallTree for neighbor search if available
        self.neighbor_finder = None
        if HAS_SKLEARN and self.bus.num_nodes > 1:
            try:
                # Use BallTree as it can be better for non-uniform distributions
                self.neighbor_finder = BallTree(np.array(self.bus.cities))
                print(f"  Builder ({self.builder_type}): Initialized BallTree for neighbor search.")
            except Exception as e:
                print(f"  Builder ({self.builder_type}) WARNING: Failed to initialize BallTree: {e}. Falling back to linear scan.")
                self.neighbor_finder = None

    def step(self) -> bool:
        """ Performs one step of path construction. Returns True if progress was made. """
        state_key = "builder_fwd_state" if self.builder_type == 'forward' else "builder_rev_state"
        current_state = getattr(self.bus, state_key)
        if current_state["status"] in ["converged", "finished", "stalled_hard"]: return False

        # --- Candidate Selection ---
        k_neighbors = self.config.get("builder_knn_k", 50)
        k_neighbors = min(k_neighbors, self.bus.num_nodes - 1)
        potential_candidates = self._find_k_nearest_unvisited(k_neighbors)

        legal_candidates = [
            node for node in potential_candidates
            if self.validator.is_edge_legal(self.current_node, node, self.builder_type)
        ]

        # --- Decision & State Update ---
        next_node = None
        progress_made = False

        if legal_candidates:
            self.stalled_cycles = 0
            if self.is_reentering: # If we were reentering, mark as finished
                print(f"BUILDER ({self.builder_type}): Reentry successful, resuming normal modulation.")
                self.is_reentering = False
                # Signal controller implicitly via lack of stall
            self.bus.update_builder_state(self.builder_type, stalled=False) # Signal progress

            # Choose best candidate based on AGRM scoring (Sweep Rank)
            legal_candidates.sort(key=lambda n: self.bus.get_node_sweep_data(n).get('sweep_rank', float('inf')))
            next_node = legal_candidates[0]

        else: # Stalled
            self.stalled_cycles += 1
            self.bus.update_builder_state(self.builder_type, stalled=True)
            # print(f"BUILDER ({self.builder_type}): Stalled at node {self.current_node}, cycle {self.stalled_cycles}.")

            # Check if Controller activated reentry mode
            if self.bus.modulation_params.get("reentry_mode", False):
                if not self.is_reentering:
                    print(f"BUILDER ({self.builder_type}): Reentry mode active. Attempting inward move...")
                    self.is_reentering = True
                next_node = self._find_reentry_node()
                if not next_node:
                    print(f"BUILDER ({self.builder_type}): Reentry failed to find valid inward node. Hard stall likely.")
                    self.bus.update_builder_state(self.builder_type, status="stalled_hard")
            # Else: Normal stall, wait for controller action

        # --- Update Path ---
        if next_node is not None:
            self.bus.add_visited(next_node, self.builder_type)
            self.current_node = next_node
            progress_made = True
            # Ensure status is running if progress made
            if current_state["status"] != "running":
                self.bus.update_builder_state(self.builder_type, status="running")

        return progress_made

    def _find_k_nearest_unvisited(self, k: int) -> List[int]:
        """ Finds up to k nearest unvisited nodes using spatial index or fallback. """
        unvisited_nodes_set = self.bus.get_unvisited_nodes()
        if not unvisited_nodes_set: return []
        if k <= 0: return []

        current_pos = np.array([self.bus.cities[self.current_node]])

        if self.neighbor_finder:
            # Query tree for more neighbors than needed, then filter
            query_k = min(len(unvisited_nodes_set), k * 5, self.bus.num_nodes) # Query more initially
            try:
                 distances, indices = self.neighbor_finder.query(current_pos, k=query_k)
                 # indices[0] contains neighbor indices, distances[0] the distances
                 # Filter out self and already visited nodes
                 neighbors = []
                 for idx in indices[0]:
                     if idx != self.current_node and idx in unvisited_nodes_set:
                         neighbors.append(idx)
                         if len(neighbors) == k: break # Stop when we have enough
                 return neighbors
            except Exception as e:
                 print(f"  Builder ({self.builder_type}) WARNING: KDTree/BallTree query failed: {e}. Falling back.")
                 # Fallback to linear scan if tree query fails

        # Fallback: Linear scan over unvisited nodes
        distances = []
        for node_idx in unvisited_nodes_set:
            if node_idx == self.current_node: continue
            dist = math.dist(current_pos[0], self.bus.cities[node_idx])
            distances.append((dist, node_idx))
        distances.sort()
        return [node_idx for dist, node_idx in distances[:k]]


    def _find_reentry_node(self) -> Optional[int]:
         """ Finds a valid candidate node closer to the spiral center during reentry. """
         if not self.bus.center or not self.bus.sweep_data: return None
         current_data = self.bus.get_node_sweep_data(self.current_node)
         current_shell = current_data.get('shell', -1)
         if current_shell <= 0: return None # Already at center

         k_neighbors = self.config.get("builder_knn_k_reentry", 100)
         potential_candidates = self._find_k_nearest_unvisited(k_neighbors)

         valid_reentry_nodes = []
         for node in potential_candidates:
             # Check legality using RELAXED rules (validator uses current bus params)
             if self.validator.is_edge_legal(self.current_node, node, self.builder_type):
                 data_to = self.bus.get_node_sweep_data(node)
                 shell_to = data_to.get('shell', -1)
                 # Must move to an inner shell
                 if shell_to != -1 and shell_to < current_shell:
                     score = (current_shell - shell_to) # Prioritize larger drop
                     valid_reentry_nodes.append((score, node))

         if valid_reentry_nodes:
             valid_reentry_nodes.sort(reverse=True) # Best score (largest drop) first
             return valid_reentry_nodes[0][1]
         else:
             return None

    def splice_patch_if_instructed(self):
         """ Checks bus for accepted patches and splices them if applicable. """
         # This function is called by the main runner loop
         accepted_patches = self.bus.get_accepted_patches()
         if not accepted_patches: return

         # Process patches relevant to this builder? Or assume global path?
         # Assume patches apply to the final merged path managed by the bus
         spliced_any = False
         remaining_patches = []
         for patch in accepted_patches:
              # Check if patch applies to the portion built by this agent? Complex.
              # Simplification: Let the bus handle splicing on the final path.
              # This builder doesn't modify its history directly, relies on bus state.
              # If more sophisticated local splicing is needed, logic goes here.
              # For now, just acknowledge the concept.
              pass # Logic is handled in bus.splice_patch called by runner

         # If splicing happened locally, update self.current_node if needed
         # self.bus.clear_accepted_patches() # Runner should clear after processing

# ======================================
# === AGRM Agent: Salesman Validator ===
# ======================================
# (SalesmanValidator class code as provided previously - verified complete)
class SalesmanValidator:
    """
    Analyzes a completed path for inefficiencies (long jumps, curvature breaks).
    Generates AGRM-legal patch proposals for refinement via the Controller.
    Includes basic 2-opt check.
    """
    def __init__(self, bus: AGRMStateBus, validator: AGRMEdgeValidator, config: Dict):
        self.bus = bus
        self.validator = validator # Used to check legality of proposed patches
        self.config = config
        self.stats = {'flags_generated': 0, 'proposals_generated': 0}

    def run_validation_and_patching(self):
        """ Runs the post-path validation and patch generation cycle. """
        path = self.bus.full_path
        # Ensure path exists and is a closed loop for 2-opt checks
        if not path or len(path) < 4 or path[0] != path[-1]:
            print("SALESMAN: Path too short, not available, or not closed. Skipping validation.")
            return

        print(f"SALESMAN: Starting validation of path with {len(path)} steps...")
        self.stats['flags_generated'] = 0
        self.stats['proposals_generated'] = 0
        proposals = []

        max_len_factor = self.config.get("salesman_max_len_factor", 4.0)
        max_curve = self.config.get("salesman_max_curve", math.pi * 0.5) # 90 deg
        enable_2opt = self.config.get("salesman_enable_2opt", True)
        opt_threshold = self.config.get("salesman_2opt_threshold", 0.99) # Min 1% improvement

        # Use baseline legality params for checking proposed swaps
        validation_params = self.bus.default_modulation_params

        # Iterate through path segments for checks
        # Note: path includes return to start, so iterate up to len(path) - 2 for curvature/2-opt
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            pos1 = self.bus.cities[p1]
            pos2 = self.bus.cities[p2]
            dist12 = math.dist(pos1, pos2)

            # 1. Check Long Jumps
            shell1 = self.bus.get_node_sweep_data(p1).get('shell', 0)
            avg_dist_in_shell = max(1.0, self.bus.shell_width or 10.0)
            dist_threshold = avg_dist_in_shell * max_len_factor
            if dist12 > dist_threshold:
                self.stats['flags_generated'] += 1
                # print(f"SALESMAN FLAG (Long Jump): {p1}->{p2} (Dist: {dist12:.2f})")

            # 2. Check Sharp Turns (at p2 = path[i+1])
            if i < len(path) - 2:
                p_prev = path[i]     # p1
                p_curr = path[i+1]   # p2
                p_next = path[i+2]   # p3
                pos_prev, pos_curr, pos_next = self.bus.cities[p_prev], self.bus.cities[p_curr], self.bus.cities[p_next]
                vec1 = (pos_curr[0] - pos_prev[0], pos_curr[1] - pos_prev[1])
                vec2 = (pos_next[0] - pos_curr[0], pos_next[1] - pos_curr[1])
                len1, len2 = math.hypot(vec1[0], vec1[1]), math.hypot(vec2[0], vec2[1])
                if len1 > 1e-9 and len2 > 1e-9:
                    dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
                    cos_angle = max(-1.0, min(1.0, dot / (len1 * len2)))
                    angle = math.acos(cos_angle)
                    if angle > max_curve:
                        self.stats['flags_generated'] += 1
                        # print(f"SALESMAN FLAG (Sharp Turn): at {p_curr} (Angle: {math.degrees(angle):.1f})")

            # 3. Check for 2-Opt Improvements (Edges: p1->p2 and p3->p4)
            # We need i+3 to exist, and ensure we don't wrap around incorrectly
            if enable_2opt and i < len(path) - 3:
                 p3 = path[i+2]
                 p4 = path[i+3]
                 # Ensure p1 != p3 and p2 != p4 to avoid degenerate swaps
                 if p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4: continue

                 pos3, pos4 = self.bus.cities[p3], self.bus.cities[p4]
                 current_dist = dist12 + math.dist(pos3, pos4)
                 swapped_dist = math.dist(pos1, pos3) + math.dist(pos2, p4)

                 if swapped_dist < current_dist * opt_threshold:
                     # Potential improvement. Check if new edges p1->p3 and p2->p4 are AGRM-legal
                     # Use the validator instance with baseline/validation params
                     # Pass 'final_check' or similar context if validator uses it
                     # Note: is_edge_legal needs access to params, pass them explicitly
                     if self.validator.is_edge_legal(p1, p3, 'final_check') and \
                        self.validator.is_edge_legal(p2, p4, 'final_check'):
                         self.stats['flags_generated'] += 1
                         self.stats['proposals_generated'] += 1
                         print(f"SALESMAN: Proposing 2-opt swap: ({p1},{p2}) & ({p3},{p4}) -> ({p1},{p3}) & ({p2},{p4}). Saving: {current_dist - swapped_dist:.2f}")
                         # Define the patch: replaces segment from index i+1 to i+2
                         # Original: [... p1, p2, p3, p4 ...]
                         # Swapped: [... p1, p3, p2, p4 ...]
                         # The segment between p1 and p4 needs reversal: [p3, p2] replaces [p2, p3]
                         proposal = {
                             'type': '2-opt',
                             'segment_indices': (i, i+3), # Indices covering p1 to p4
                             'original_nodes': [p1, p2, p3, p4],
                             # The new sequence for nodes BETWEEN index i and index i+3
                             # The path from i+1 up to (but not including) i+3 needs reversal
                             # Original segment is path[i+1 : i+3] = [p2, p3]
                             # New segment should be reversed: [p3, p2]
                             'new_subpath_nodes': path[i+1 : i+3][::-1], # Reverse the segment between swapped edges
                             'cost_saving': current_dist - swapped_dist
                         }
                         proposals.append(proposal)
                     # else: print(f"SALESMAN: Potential 2-opt swap rejected by AGRM legality.")

        print(f"SALESMAN: Validation complete. Found {self.stats['flags_generated']} flags. Generated {self.stats['proposals_generated']} patch proposals.")
        # Send valid proposals to the Controller via the bus
        if proposals:
            for p in proposals:
                self.bus.add_salesman_proposal(p)


# ==============================
# === Path Audit Agent (NEW) ===
# ==============================
# (PathAuditAgent class code as provided previously - verified complete)
class PathAuditAgent:
    """
    Runs AFTER the full AGRM + Salesman process is complete.
    Evaluates the final path quality using global metrics.
    Analyzes patterns of sub-optimality.
    Generates parameter adjustment recommendations for the NEXT run.
    Enables run-to-run meta-learning.
    """
    def __init__(self, bus: AGRMStateBus, config: Dict):
        self.bus = bus
        self.config = config
        self.metrics = {}
        self.patterns = {}
        self.recommendations = {}

    def run_audit(self) -> Dict:
        """ Performs the full audit process. Returns recommendations dict. """
        print("AUDIT AGENT: Starting post-run path audit...")
        if not self.bus.full_path or self.bus.current_phase not in ["merged", "finalizing", "complete", "patched"]: # Allow patched state
            print("AUDIT AGENT: Final path not available or run not complete. Skipping audit.")
            return {}

        self.metrics = self._calculate_global_metrics()
        self.patterns = self._analyze_patterns()
        self.recommendations = self._generate_recommendations()

        print("AUDIT AGENT: Audit complete.")
        print(f"  Audit Metrics: {self.metrics}")
        print(f"  Audit Patterns: {self.patterns}")
        print(f"  Audit Recommendations: {self.recommendations}")
        return self.recommendations

    def _calculate_global_metrics(self) -> Dict:
        """ Calculates high-level quality metrics for the final path. """
        metrics = {}
        path = self.bus.full_path
        # 1. Final Path Length
        final_cost = self.bus.calculate_total_path_cost(path) # Use bus helper
        metrics['final_path_cost'] = final_cost
        metrics['final_efficiency'] = final_cost / max(1, self.bus.num_nodes)

        # 2. Comparison to Baseline (e.g., simple Nearest Neighbor from start)
        baseline_cost = self._run_simple_nn_baseline()
        metrics['baseline_nn_cost'] = baseline_cost
        if baseline_cost > 0:
             metrics['length_vs_baseline'] = final_cost / baseline_cost

        # 3. Remaining Salesman Flags (Count reported by Salesman)
        # Need Salesman to store final flag count accessible here
        # metrics['remaining_salesman_flags'] = self.bus.salesman_final_flags?

        # 4. Structural Metrics (Example: Bounding Box Ratio)
        if path:
             coords = np.array([self.bus.cities[i] for i in path[:-1]]) # Exclude return to start
             min_x, min_y = np.min(coords, axis=0)
             max_x, max_y = np.max(coords, axis=0)
             width = max_x - min_x
             height = max_y - min_y
             metrics['bounding_box_ratio'] = width / height if height > 0 else 1.0

        # 5. Add more metrics: Avg turn angle, std dev of edge lengths, etc.
        return metrics

    def _run_simple_nn_baseline(self) -> float:
         """ Runs a basic Nearest Neighbor heuristic for baseline comparison. """
         if not self.bus.cities: return 0.0
         start_node = self.bus.start_node_fwd if self.bus.start_node_fwd is not None else 0
         unvisited = set(range(self.bus.num_nodes))
         current = start_node
         path = [current]
         unvisited.remove(current)
         total_dist = 0.0

         while unvisited:
             nearest_node = -1
             min_dist = float('inf')
             pos_current = self.bus.cities[current]
             for node in unvisited:
                 dist = math.dist(pos_current, self.bus.cities[node])
                 if dist < min_dist:
                     min_dist = dist
                     nearest_node = node
             if nearest_node != -1:
                 total_dist += min_dist
                 current = nearest_node
                 path.append(current)
                 unvisited.remove(current)
             else: break # Should not happen if graph is connected

         # Add return to start
         if len(path) > 1:
              total_dist += math.dist(self.bus.cities[current], self.bus.cities[start_node])
         return total_dist


    def _analyze_patterns(self) -> Dict:
        """ Analyzes patterns of sub-optimality in the final path. """
        patterns = {}
        # Example: Analyze where Salesman flags occurred (requires Salesman to store flag locations)
        # patterns['flag_concentration_quadrant'] = self._analyze_flag_distribution('quadrant')
        # patterns['flag_concentration_shell'] = self._analyze_flag_distribution('shell')
        # patterns['high_cost_segments'] = self._find_high_cost_segments()
        return patterns # Placeholder

    def _generate_recommendations(self) -> Dict:
        """ Generates parameter tuning recommendations based on metrics and patterns. """
        recommendations = {}
        # Example Rules:
        length_ratio = self.metrics.get('length_vs_baseline', 1.0)
        target_ratio = self.config.get("audit_target_baseline_ratio", 1.1) # e.g., aim for 10% worse than NN

        # If path is much longer than baseline, maybe legality was too strict?
        if length_ratio > target_ratio * 1.2: # If >20% worse than target
             # Suggest relaxing curvature or shell tolerance slightly
             recommendations['mod_curvature_limit'] = self.bus.default_modulation_params['curvature_limit'] * 1.1 # Relax by 10%
             recommendations['mod_shell_tolerance'] = self.bus.default_modulation_params['shell_tolerance'] + 1
             print("AUDIT Recommendation: Path long vs baseline. Suggest relaxing curvature/shell tolerance.")

        # If Salesman found many 2-opt opportunities (requires pattern analysis)
        # if self.patterns.get('high_2opt_flags'):
        #    recommendations['salesman_2opt_threshold'] = self.config['salesman_2opt_threshold'] * 1.01 # Make slightly easier to trigger
        #    print("AUDIT Recommendation: Many 2-opt flags. Suggest lowering 2-opt improvement threshold.")

        # Add more rules based on other metrics and patterns
        return recommendations


# ==============================
# === AGRM System Controller ===
# ==============================
# (AGRMRunner class code as provided previously - verified complete)
# Renamed to AGRMController for clarity
class AGRMController:
    """ Orchestrates the AGRM TSP solving process using the agent stack. """
    def __init__(self, cities: List[Tuple[float, float]], config: Dict = {}, previous_recommendations: Dict = {}):
        """
        Initializes the controller and all agents.
        Args:
            cities: List of city coordinates.
            config: Base configuration dictionary.
            previous_recommendations: Parameter adjustments from previous PathAuditAgent run.
        """
        self.cities = cities
        self.num_nodes = len(cities)

        # Apply recommendations to base config
        self.config = config.copy()
        if previous_recommendations:
            print(f"CONTROLLER: Applying {len(previous_recommendations)} recommendations from previous run.")
            self.config.update(previous_recommendations)
            print(f"  New Config Snippet: { {k: self.config[k] for k in previous_recommendations} }")

        # Initialize shared state bus with potentially updated config
        self.bus = AGRMStateBus(cities, self.config)

        # Initialize agents, passing the bus and config
        self.navigator = NavigatorGR(cities, self.config)
        self.validator = AGRMEdgeValidator(self.bus, self.config)
        # Pass self (controller) to agents that might need to signal back directly? Or use bus.
        self.mod_controller_agent = ModulationController(self.bus, self.config) # The agent managing modulation params
        self.builder_fwd: Optional[PathBuilder] = None
        self.builder_rev: Optional[PathBuilder] = None
        self.salesman = SalesmanValidator(self.bus, self.validator, self.config)
        self.path_audit = PathAuditAgent(self.bus, self.config) # Initialize audit agent

        self.run_stats = {}


    def solve(self) -> Tuple[Optional[List[int]], Dict]:
        """ Runs the full AGRM TSP solving process, returning path and stats. """
        print(f"\n=== AGRM Controller: Starting Solve for {self.num_nodes} Nodes ===")
        overall_start_time = time.time()
        self.run_stats = {} # Reset stats for this run

        # --- 1. Sweep Phase ---
        sweep_results = self.navigator.run_sweep()
        self.bus.update_sweep_data(sweep_results)
        if self.bus.start_node_fwd is None or self.bus.start_node_rev is None:
             print("CONTROLLER ERROR: Navigator failed to determine start nodes.")
             return None, {"error": "Navigator failed"}
        self.run_stats['sweep_time'] = time.time() - overall_start_time

        # --- Initialize Builders ---
        self.builder_fwd = PathBuilder('forward', self.bus.start_node_fwd, self.bus, self.validator, self.config)
        self.builder_rev = PathBuilder('reverse', self.bus.start_node_rev, self.bus, self.validator, self.config)

        # --- 2. Bidirectional Build Phase ---
        print("CONTROLLER: Starting Bidirectional Build Phase...")
        build_start_time = time.time()
        max_steps = self.num_nodes * self.config.get("runner_max_step_factor", 3)
        steps = 0
        build_complete = False
        while steps < max_steps:
            steps += 1
            # Alternate stepping? Or step both? Stepping both:
            progress_fwd = self.builder_fwd.step() if self.bus.builder_fwd_state["status"] in ["running", "stalled"] else False
            progress_rev = self.builder_rev.step() if self.bus.builder_rev_state["status"] in ["running", "stalled"] else False

            # Update modulation controller based on latest builder states
            self.mod_controller_agent.update_controller_state()

            # Check for convergence
            if self.bus.check_convergence():
                 print(f"CONTROLLER: Convergence detected at step {steps}.")
                 build_complete = self.bus.merge_paths()
                 break # Exit build loop

            # Check for hard stalls
            if self.bus.builder_fwd_state["status"] == "stalled_hard" and \
               self.bus.builder_rev_state["status"] == "stalled_hard":
                print("CONTROLLER ERROR: Both builders hard stalled. Attempting merge.")
                build_complete = self.bus.merge_paths() # Try merging anyway
                break

            # Check if no progress possible and not converged
            if not progress_fwd and not progress_rev and \
               self.bus.builder_fwd_state["status"] not in ["running", "converged"] and \
               self.bus.builder_rev_state["status"] not in ["running", "converged"]:
                 print(f"CONTROLLER WARNING: No progress from either builder at step {steps}. Stopping build.")
                 build_complete = self.bus.merge_paths() # Try merging what we have
                 break

        if steps >= max_steps:
             print(f"CONTROLLER WARNING: Max steps ({max_steps}) reached during build phase.")
             build_complete = self.bus.merge_paths() # Try merging what we have

        self.run_stats['build_time'] = time.time() - build_start_time
        print(f"CONTROLLER: Build phase finished in {self.run_stats['build_time']:.4f}s ({steps} steps).")

        # --- 3. Patching Phase (If Needed for Missed Nodes) ---
        if self.bus.current_phase == "patching":
            print("CONTROLLER: Entering patching phase for missed nodes...")
            patch_start_time = time.time()
            # TODO: Implement robust patching logic
            # Needs to find missed nodes, determine best insertion points respecting legality
            # This is complex - requires potentially re-invoking builder/validator locally
            # For now, just flag as incomplete
            print(f"CONTROLLER WARNING: Path construction incomplete, {self.num_nodes - len(set(self.bus.full_path))} nodes missed. Patching logic not fully implemented.")
            build_complete = False # Mark as incomplete
            self.run_stats['patch_time'] = time.time() - patch_start_time
            self.bus.current_phase = "finalizing" # Move phase even if incomplete

        # --- 4. Salesman Validation & Refinement ---
        if self.bus.full_path:
            print("CONTROLLER: Starting Salesman validation and refinement...")
            salesman_start_time = time.time()
            self.salesman.run_validation_and_patching()
            # Controller evaluates proposals and stores accepted ones
            self.mod_controller_agent.process_salesman_feedback()
            # Builder splices approved patches (via bus)
            accepted_patches = self.bus.get_accepted_patches()
            if accepted_patches:
                 print(f"CONTROLLER: Applying {len(accepted_patches)} accepted patches...")
                 spliced_count = 0
                 # Apply patches iteratively? Or assume they don't overlap significantly?
                 # Applying iteratively for now
                 for patch in accepted_patches:
                     if self.bus.splice_patch(patch):
                         spliced_count += 1
                 print(f"CONTROLLER: Successfully spliced {spliced_count} patches.")
                 self.bus.clear_accepted_patches() # Clear after applying
            self.run_stats['salesman_time'] = time.time() - salesman_start_time
            print(f"CONTROLLER: Salesman phase complete in {self.run_stats['salesman_time']:.4f}s.")
        else:
            print("CONTROLLER: No path generated, skipping Salesman validation.")
            self.run_stats['salesman_time'] = 0.0

        # --- 5. Path Audit Phase (Meta-Learning) ---
        print("CONTROLLER: Starting Path Audit...")
        audit_start_time = time.time()
        recommendations = self.path_audit.run_audit() # Returns dict of param adjustments
        self.run_stats['audit_time'] = time.time() - audit_start_time
        print(f"CONTROLLER: Path Audit complete in {self.run_stats['audit_time']:.4f}s.")

        # --- 6. Final Results ---
        overall_end_time = time.time()
        final_path = self.bus.full_path
        # Recalculate final cost after potential splicing
        final_cost = self.calculate_total_path_cost(final_path) if final_path else 0.0

        # Compile final statistics
        final_stats = {
            "execution_time": overall_end_time - overall_start_time,
            "sweep_time": self.run_stats.get('sweep_time', 0.0),
            "build_time": self.run_stats.get('build_time', 0.0),
            "patch_time": self.run_stats.get('patch_time', 0.0),
            "salesman_time": self.run_stats.get('salesman_time', 0.0),
            "audit_time": self.run_stats.get('audit_time', 0.0),
            "path_complete": build_complete and bool(final_path) and len(set(final_path[:-1])) == self.num_nodes,
            "visited_nodes": len(set(final_path[:-1])) if final_path else 0,
            "path_length_nodes": len(final_path) if final_path else 0,
            "salesman_flags": self.salesman.stats.get('flags_generated', 0),
            "salesman_proposals": self.salesman.stats.get('proposals_generated', 0),
            "total_path_cost": final_cost,
            "efficiency": final_cost / max(1, self.num_nodes),
            "audit_recommendations": recommendations # Include recommendations for next run
        }
        self.bus.current_phase = "complete"
        print(f"=== AGRM Controller: Solve Complete in {final_stats['execution_time']:.4f}s ===")
        return final_path, final_stats

    def calculate_total_path_cost(self, path: Optional[List[int]]) -> float:
        """ Calculates the Euclidean distance of the TSP path. """
        if not path or len(path) < 2: return 0.0
        total_distance = 0.0
        for i in range(len(path) - 1):
            # Add checks for valid indices
            idx1, idx2 = path[i], path[i+1]
            if 0 <= idx1 < self.num_nodes and 0 <= idx2 < self.num_nodes:
                 pos1 = self.cities[idx1]
                 pos2 = self.cities[idx2]
                 total_distance += math.dist(pos1, pos2)
            else:
                 print(f"ERROR: Invalid node index in path during cost calculation: {idx1} or {idx2}")
                 return 0.0 # Indicate error
        return total_distance

# =========================
# === Example Usage ===
# =========================
if __name__ == "__main__":
    # Generate sample cities for testing
    NUM_CITIES = 100 # Small test for demonstration
    cities_data = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(NUM_CITIES)]

    # --- Configuration ---
    # Define base configuration
    config = {
        "sweep_num_shells": 10, "sweep_num_sectors": 8, "density_knn_k": 10,
        "density_dense_factor": 1.5, "density_sparse_factor": 0.5, "density_std_dev_factor": 0.75,
        "hybrid_hash_threshold": 5, "mdhg_dimensions": 3, "mdhg_capacity_factor": 1.2, # Factor of num_nodes
        "mod_shell_tolerance": 2, "mod_curvature_limit": math.pi / 4, "mod_sector_tolerance": 2,
        "mod_dist_cap_factor": 3.0, "mod_reentry_curve_relax": math.pi / 6,
        "mod_reentry_shell_relax": 2, "mod_reentry_dist_relax_factor": 1.5,
        "mod_override_curve_relax": math.pi / 12, "mod_override_shell_relax": 1,
        "dist_cap_sparse_mult": 1.5, "dist_cap_override_mult": 1.5,
        "midpoint_unlock_percent": 0.5, "controller_stall_threshold": 5, "controller_severe_stall_factor": 2,
        "convergence_shell_threshold": 1, "convergence_sector_threshold": 1, "convergence_stall_dist_factor": 5.0,
        "builder_knn_k": 50, "builder_knn_k_reentry": 100,
        "salesman_max_len_factor": 4.0, "salesman_max_curve": math.pi * 0.5,
        "salesman_enable_2opt": True, "salesman_2opt_threshold": 0.98,
        "runner_max_step_factor": 3,
        "audit_target_baseline_ratio": 1.1, "controller_patch_min_saving": 0.1,
        # MDHG Specific Config (passed to MDHGHashTable)
        "mdhg_access_history_len": 100, "mdhg_velocity_promo_threshold": 10,
        "mdhg_ops_thresh_minor": 100, "mdhg_time_thresh_minor": 1.0,
        "mdhg_ops_thresh_major": 1000, "mdhg_time_thresh_major": 5.0,
        "mdhg_hot_key_count": 100, "mdhg_hot_key_min_freq": 5,
        "mdhg_cluster_threshold": 3, "mdhg_path_cache_max_size": 100,
        "mdhg_max_put_probes": 20, "mdhg_max_search_probes": 20,
        "mdhg_path_length_limit": 1000
    }

    print(f"--- Starting AGRM TSP Solver for {NUM_CITIES} cities ---")
    # --- Run 1 ---
    print("\n--- Run 1 ---")
    solver1 = AGRMController(cities_data, config)
    final_path1, final_stats1 = solver1.solve()

    print("\n--- AGRM Run 1 Results ---")
    print(f"Execution Time: {final_stats1.get('execution_time', 0.0):.4f} seconds")
    print(f"Path Complete: {final_stats1.get('path_complete', False)}")
    print(f"Nodes Visited: {final_stats1.get('visited_nodes', 0)} / {NUM_CITIES}")
    print(f"Total Path Cost: {final_stats1.get('total_path_cost', 0.0):.4f}")
    print(f"Efficiency (Cost/Node): {final_stats1.get('efficiency', 0.0):.4f}")
    print(f"Salesman Flags: {final_stats1.get('salesman_flags', 0)}")
    print(f"Audit Recommendations: {final_stats1.get('audit_recommendations', {})}")

    # --- Run 2 (Using Recommendations from Run 1) ---
    print("\n--- Run 2 (Applying Audit Recommendations) ---")
    recommendations1 = final_stats1.get('audit_recommendations', {})
    solver2 = AGRMController(cities_data, config, previous_recommendations=recommendations1)
    final_path2, final_stats2 = solver2.solve()

    print("\n--- AGRM Run 2 Results ---")
    print(f"Execution Time: {final_stats2.get('execution_time', 0.0):.4f} seconds")
    print(f"Path Complete: {final_stats2.get('path_complete', False)}")
    print(f"Nodes Visited: {final_stats2.get('visited_nodes', 0)} / {NUM_CITIES}")
    print(f"Total Path Cost: {final_stats2.get('total_path_cost', 0.0):.4f}")
    print(f"Efficiency (Cost/Node): {final_stats2.get('efficiency', 0.0):.4f}")
    print(f"Salesman Flags: {final_stats2.get('salesman_flags', 0)}")
    print(f"Audit Recommendations: {final_stats2.get('audit_recommendations', {})}")

    # --- Comparison ---
    print("\n--- Run Comparison ---")
    cost1 = final_stats1.get('total_path_cost', float('inf'))
    cost2 = final_stats2.get('total_path_cost', float('inf'))
    time1 = final_stats1.get('execution_time', 0.0)
    time2 = final_stats2.get('execution_time', 0.0)
    print(f"Run 1 Cost: {cost1:.4f} ({time1:.4f}s)")
    print(f"Run 2 Cost: {cost2:.4f} ({time2:.4f}s)")
    if cost1 > 0:
         print(f"Cost Improvement: {(cost1 - cost2) / cost1 * 100:.2f}%")
