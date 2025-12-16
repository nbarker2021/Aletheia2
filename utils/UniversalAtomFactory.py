class UniversalAtomFactory:
    """Factory for creating universal atoms from any data"""
    
    def __init__(self):
        self.sacred_frequencies = {
            1: 174.0, 2: 285.0, 3: 396.0, 4: 417.0, 5: 528.0,
            6: 639.0, 7: 741.0, 8: 852.0, 9: 963.0
        }
        
        self.binary_patterns = {
            1: SacredBinaryPattern.UNITY_FOUNDATION,
            2: SacredBinaryPattern.DUALITY_BALANCE,
            3: SacredBinaryPattern.CREATIVE_SEED,
            4: SacredBinaryPattern.STABILITY_ANCHOR,
            5: SacredBinaryPattern.TRANSFORMATIVE_CYCLE,
            6: SacredBinaryPattern.OUTWARD_EXPANSION,
            7: SacredBinaryPattern.TRANSFORMATIVE_CYCLE,
            8: SacredBinaryPattern.STABILITY_ANCHOR,
            9: SacredBinaryPattern.INWARD_COMPRESSION
        }
        
        self.rotational_patterns = {
            9: "INWARD_ROTATIONAL",
            6: "OUTWARD_ROTATIONAL", 
            3: "CREATIVE_SEED",
            1: "TRANSFORMATIVE_CYCLE", 2: "TRANSFORMATIVE_CYCLE",
            4: "TRANSFORMATIVE_CYCLE", 5: "TRANSFORMATIVE_CYCLE",
            7: "TRANSFORMATIVE_CYCLE", 8: "TRANSFORMATIVE_CYCLE"
        }
    
    def create_atom_from_data(self, data: Any) -> UniversalAtom:
        """Create universal atom from arbitrary data"""
        
        # Step 1: Generate CQE properties
        e8_coords = self.generate_e8_coordinates(data)
        quad_encoding = self.generate_quad_encoding(data)
        parity_channels = self.generate_parity_channels(data)
        
        # Step 2: Generate Sacred Geometry properties
        digital_root = self.calculate_digital_root(data)
        sacred_frequency = self.sacred_frequencies[digital_root]
        binary_guidance = self.binary_patterns[digital_root].value
        rotational_pattern = self.rotational_patterns[digital_root]
        
        # Step 3: Generate Mandelbrot properties
        fractal_coord = self.generate_fractal_coordinate(data)
        fractal_behavior = self.determine_fractal_behavior(fractal_coord)
        compression_ratio = self.calculate_compression_ratio(fractal_coord, fractal_behavior)
        iteration_depth = self.calculate_iteration_depth(fractal_coord)
        
        # Create atom
        atom = UniversalAtom(
            e8_coordinates=e8_coords,
            quad_encoding=quad_encoding,
            parity_channels=parity_channels,
            digital_root=digital_root,
            sacred_frequency=sacred_frequency,
            binary_guidance=binary_guidance,
            rotational_pattern=rotational_pattern,
            fractal_coordinate=fractal_coord,
            fractal_behavior=fractal_behavior,
            compression_ratio=compression_ratio,
            iteration_depth=iteration_depth,
            bit_representation=b'',  # Will be calculated in __post_init__
            storage_size=0,          # Will be calculated in __post_init__
            combination_mask=0,      # Will be calculated in __post_init__
            creation_timestamp=np.random.random(),  # Placeholder
            access_count=0,
            combination_history=[]
        )
        
        return atom
    
    def generate_e8_coordinates(self, data: Any) -> np.ndarray:
        """Generate E₈ lattice coordinates from data"""
        # Convert data to hash for consistent coordinate generation
        data_hash = hashlib.sha256(str(data).encode()).digest()
        
        # Extract 8 coordinates from hash using integer approach
        coords = []
        for i in range(8):
            # Use 4 bytes per coordinate, convert to integer first
            byte_slice = data_hash[i*4:(i+1)*4]
            if len(byte_slice) == 4:
                int_value = struct.unpack('I', byte_slice)[0]
                coord_value = (int_value % 2000000 - 1000000) / 1000000.0  # Scale to [-1, 1]
            else:
                coord_value = 0.0
            coords.append(coord_value)
        
        coords = np.array(coords)
        
        # Handle potential NaN or inf values
        coords = np.nan_to_num(coords, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize to E₈ lattice scale
        norm = np.linalg.norm(coords)
        if norm > 0:
            coords = coords / norm
        else:
            # If all coordinates are zero, create a default pattern
            coords = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return coords
    
    def generate_quad_encoding(self, data: Any) -> Tuple[int, int, int, int]:
        """Generate 4D quadratic encoding from data"""
        data_hash = hashlib.md5(str(data).encode()).digest()
        
        # Extract 4 integers from hash
        quad = []
        for i in range(4):
            byte_slice = data_hash[i*4:(i+1)*4]
            if len(byte_slice) == 4:
                value = struct.unpack('I', byte_slice)[0] % 256  # Keep in reasonable range
            else:
                value = 0
            quad.append(value)
        
        return tuple(quad)
    
    def generate_parity_channels(self, data: Any) -> np.ndarray:
        """Generate 8-channel parity state from data"""
        data_str = str(data)
        channels = np.zeros(8)
        
        for i, char in enumerate(data_str[:8]):
            channels[i] = ord(char) % 2  # Binary parity
        
        # Fill remaining channels if data is short
        for i in range(len(data_str), 8):
            channels[i] = hash(data_str) % 2
        
        return channels
    
    def calculate_digital_root(self, data: Any) -> int:
        """Calculate Carlson's digital root from data"""
        # Convert data to numeric value
        if isinstance(data, (int, float)):
            n = abs(int(data * 1000))
        else:
            n = abs(hash(str(data))) % 1000000
        
        # Calculate digital root
        while n >= 10:
            n = sum(int(digit) for digit in str(n))
        
        return n if n > 0 else 9
    
    def generate_fractal_coordinate(self, data: Any) -> complex:
        """Generate Mandelbrot coordinate from data"""
        data_hash = hashlib.sha1(str(data).encode()).digest()
        
        # Extract real and imaginary parts using integer approach
        real_bytes = data_hash[:4]
        imag_bytes = data_hash[4:8]
        
        if len(real_bytes) == 4:
            real_int = struct.unpack('I', real_bytes)[0]
            real_part = (real_int % 4000000 - 2000000) / 1000000.0  # Scale to [-2, 2]
        else:
            real_part = 0.0
            
        if len(imag_bytes) == 4:
            imag_int = struct.unpack('I', imag_bytes)[0]
            imag_part = (imag_int % 3000000 - 1500000) / 1000000.0  # Scale to [-1.5, 1.5]
        else:
            imag_part = 0.0
        
        # Handle potential NaN or inf values
        real_part = np.nan_to_num(real_part, nan=0.0, posinf=1.5, neginf=-2.5)
        imag_part = np.nan_to_num(imag_part, nan=0.0, posinf=1.5, neginf=-1.5)
        
        # Ensure within Mandelbrot viewing region
        real_part = max(-2.5, min(1.5, real_part))
        imag_part = max(-1.5, min(1.5, imag_part))
        
        return complex(real_part, imag_part)
    
    def determine_fractal_behavior(self, c: complex, max_iter: int = 100) -> str:
        """Determine Mandelbrot fractal behavior"""
        z = complex(0, 0)
        
        for i in range(max_iter):
            if abs(z) > 2.0:
                if i < max_iter * 0.2:
                    return 'ESCAPING'
                else:
                    return 'BOUNDARY'
            z = z*z + c
        
        # Check for periodic behavior
        orbit = []
        for i in range(20):
            z = z*z + c
            orbit.append(z)
        
        # Simple periodicity check
        for period in [2, 3, 4, 5]:
            if len(orbit) >= 2 * period:
                is_periodic = True
                for j in range(period):
                    if abs(orbit[-(j+1)] - orbit[-(j+1+period)]) > 1e-6:
                        is_periodic = False
                        break
                if is_periodic:
                    return 'PERIODIC'
        
        return 'BOUNDED'
    
    def calculate_compression_ratio(self, c: complex, behavior: str) -> float:
        """Calculate compression/expansion ratio"""
        if behavior == 'BOUNDED':
            return 1.0 / (1.0 + abs(c))
        elif behavior == 'ESCAPING':
            return abs(c) / (1.0 + abs(c))
        else:
            return 0.5  # Balanced for boundary/periodic

    def calculate_iteration_depth(self, c: complex, max_iter: int = 100) -> int:
        """Calculate fractal iteration depth"""
        z = complex(0, 0)
        
        for i in range(max_iter):
            if abs(z) > 2.0:
                return i
            z = z*z + c
        
        return max_iter
