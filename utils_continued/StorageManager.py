class CQEStorageManager:
    """Universal storage manager using CQE principles"""
    
    def __init__(self, kernel: CQEKernel, config: StorageConfig):
        self.kernel = kernel
        self.config = config
        self.stats = StorageStats()
        
        # Storage backends
        self.memory_storage: Dict[str, CQEAtom] = {}
        self.file_storage_path = Path(config.base_path)
        self.db_connection: Optional[sqlite3.Connection] = None
        
        # Indices for fast retrieval
        self.indices: Dict[IndexType, Dict[Any, Set[str]]] = {
            index_type: defaultdict(set) for index_type in config.index_types
        }
        
        # Caching and performance
        self.access_cache: Dict[str, CQEAtom] = {}
        self.cache_size = 1000
        self.access_frequency: Dict[str, int] = defaultdict(int)
        
        # Threading and synchronization
        self.storage_lock = threading.RLock()
        self.background_tasks = []
        
        # Initialize storage backend
        self._initialize_storage()
        self._initialize_indices()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_storage(self):
        """Initialize the storage backend"""
        if self.config.storage_type in [StorageType.FILE_SYSTEM, StorageType.HYBRID]:
            self.file_storage_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.file_storage_path / "atoms").mkdir(exist_ok=True)
            (self.file_storage_path / "indices").mkdir(exist_ok=True)
            (self.file_storage_path / "backups").mkdir(exist_ok=True)
            (self.file_storage_path / "temp").mkdir(exist_ok=True)
        
        if self.config.storage_type in [StorageType.SQLITE, StorageType.HYBRID]:
            db_path = self.file_storage_path / "cqe_storage.db"
            self.db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
            self._initialize_database_schema()
    
    def _initialize_database_schema(self):
        """Initialize SQLite database schema"""
        if not self.db_connection:
            return
        
        cursor = self.db_connection.cursor()
        
        # Main atoms table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS atoms (
                id TEXT PRIMARY KEY,
                data BLOB,
                quad_encoding TEXT,
                e8_embedding BLOB,
                parity_channels TEXT,
                governance_state TEXT,
                timestamp REAL,
                parent_id TEXT,
                metadata TEXT,
                size_bytes INTEGER,
                created_at REAL,
                accessed_at REAL,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Quad index table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quad_index (
                quad_signature TEXT,
                atom_id TEXT,
                PRIMARY KEY (quad_signature, atom_id)
            )
        """)
        
        # E8 spatial index table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS e8_spatial_index (
                region_hash TEXT,
                atom_id TEXT,
                distance REAL,
                PRIMARY KEY (region_hash, atom_id)
            )
        """)
        
        # Content index table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_index (
                content_hash TEXT,
                atom_id TEXT,
                content_type TEXT,
                PRIMARY KEY (content_hash, atom_id)
            )
        """)
        
        # Metadata index table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata_index (
                key TEXT,
                value TEXT,
                atom_id TEXT,
                PRIMARY KEY (key, value, atom_id)
            )
        """)
        
        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_atoms_timestamp ON atoms(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_atoms_parent ON atoms(parent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_atoms_governance ON atoms(governance_state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_atoms_accessed ON atoms(accessed_at)")
        
        self.db_connection.commit()
    
    def _initialize_indices(self):
        """Initialize indices for fast retrieval"""
        # Load existing indices from storage
        if self.config.storage_type in [StorageType.FILE_SYSTEM, StorageType.HYBRID]:
            self._load_indices_from_disk()
        
        if self.config.storage_type in [StorageType.SQLITE, StorageType.HYBRID]:
            self._load_indices_from_database()
    
    def store_atom(self, atom: CQEAtom) -> bool:
        """Store an atom using the configured storage backend"""
        with self.storage_lock:
            try:
                # Update access statistics
                self.access_frequency[atom.id] += 1
                atom.metadata['access_count'] = self.access_frequency[atom.id]
                atom.metadata['last_accessed'] = time.time()
                
                # Store in appropriate backend(s)
                success = False
                
                if self.config.storage_type == StorageType.MEMORY:
                    success = self._store_in_memory(atom)
                
                elif self.config.storage_type == StorageType.FILE_SYSTEM:
                    success = self._store_in_file_system(atom)
                
                elif self.config.storage_type == StorageType.SQLITE:
                    success = self._store_in_database(atom)
                
                elif self.config.storage_type == StorageType.HYBRID:
                    # Store in memory for fast access
                    memory_success = self._store_in_memory(atom)
                    
                    # Store persistently
                    if len(self.memory_storage) < self.config.max_memory_size:
                        persistent_success = self._store_in_database(atom)
                    else:
                        persistent_success = self._store_in_file_system(atom)
                    
                    success = memory_success and persistent_success
                
                elif self.config.storage_type == StorageType.COMPRESSED:
                    success = self._store_compressed(atom)
                
                elif self.config.storage_type == StorageType.ENCRYPTED:
                    success = self._store_encrypted(atom)
                
                if success:
                    # Update indices
                    self._update_indices(atom)
                    
                    # Update statistics
                    self._update_storage_stats(atom, operation="store")
                    
                    # Add to cache
                    self._add_to_cache(atom)
                
                return success
            
            except Exception as e:
                print(f"Storage error: {e}")
                return False
    
    def retrieve_atom(self, atom_id: str) -> Optional[CQEAtom]:
        """Retrieve an atom by ID"""
        with self.storage_lock:
            try:
                # Check cache first
                if atom_id in self.access_cache:
                    atom = self.access_cache[atom_id]
                    self._update_access_stats(atom_id)
                    return atom
                
                # Retrieve from storage backend
                atom = None
                
                if self.config.storage_type == StorageType.MEMORY:
                    atom = self._retrieve_from_memory(atom_id)
                
                elif self.config.storage_type == StorageType.FILE_SYSTEM:
                    atom = self._retrieve_from_file_system(atom_id)
                
                elif self.config.storage_type == StorageType.SQLITE:
                    atom = self._retrieve_from_database(atom_id)
                
                elif self.config.storage_type == StorageType.HYBRID:
                    # Try memory first
                    atom = self._retrieve_from_memory(atom_id)
                    if not atom:
                        # Try database
                        atom = self._retrieve_from_database(atom_id)
                        if not atom:
                            # Try file system
                            atom = self._retrieve_from_file_system(atom_id)
                
                elif self.config.storage_type == StorageType.COMPRESSED:
                    atom = self._retrieve_compressed(atom_id)
                
                elif self.config.storage_type == StorageType.ENCRYPTED:
                    atom = self._retrieve_encrypted(atom_id)
                
                if atom:
                    # Update access statistics
                    self._update_access_stats(atom_id)
                    
                    # Add to cache
                    self._add_to_cache(atom)
                
                return atom
            
            except Exception as e:
                print(f"Retrieval error: {e}")
                return None
    
    def query_atoms(self, query: Dict[str, Any], limit: int = 100) -> List[CQEAtom]:
        """Query atoms based on various criteria"""
        with self.storage_lock:
            matching_atom_ids = set()
            
            # Use indices for efficient querying
            if 'quad_encoding' in query and IndexType.QUAD_INDEX in self.indices:
                quad_sig = self._quad_to_signature(query['quad_encoding'])
                matching_atom_ids.update(self.indices[IndexType.QUAD_INDEX].get(quad_sig, set()))
            
            if 'content_hash' in query and IndexType.CONTENT_INDEX in self.indices:
                matching_atom_ids.update(self.indices[IndexType.CONTENT_INDEX].get(query['content_hash'], set()))
            
            if 'metadata' in query and IndexType.METADATA_INDEX in self.indices:
                for key, value in query['metadata'].items():
                    meta_key = f"{key}:{value}"
                    matching_atom_ids.update(self.indices[IndexType.METADATA_INDEX].get(meta_key, set()))
            
            if 'e8_region' in query and IndexType.E8_SPATIAL_INDEX in self.indices:
                region_hash = self._e8_to_region_hash(query['e8_region'])
                matching_atom_ids.update(self.indices[IndexType.E8_SPATIAL_INDEX].get(region_hash, set()))
            
            if 'timestamp_range' in query and IndexType.TEMPORAL_INDEX in self.indices:
                start_time, end_time = query['timestamp_range']
                for timestamp, atom_ids in self.indices[IndexType.TEMPORAL_INDEX].items():
                    if start_time <= timestamp <= end_time:
                        matching_atom_ids.update(atom_ids)
            
            # If no specific indices used, scan all atoms (expensive)
            if not matching_atom_ids and not any(key in query for key in ['quad_encoding', 'content_hash', 'metadata', 'e8_region', 'timestamp_range']):
                matching_atom_ids = set(self._get_all_atom_ids())
            
            # Retrieve matching atoms
            matching_atoms = []
            for atom_id in list(matching_atom_ids)[:limit]:
                atom = self.retrieve_atom(atom_id)
                if atom and self._matches_query(atom, query):
                    matching_atoms.append(atom)
            
            return matching_atoms
    
    def delete_atom(self, atom_id: str) -> bool:
        """Delete an atom from storage"""
        with self.storage_lock:
            try:
                # Remove from all storage backends
                success = True
                
                if self.config.storage_type in [StorageType.MEMORY, StorageType.HYBRID]:
                    if atom_id in self.memory_storage:
                        del self.memory_storage[atom_id]
                
                if self.config.storage_type in [StorageType.FILE_SYSTEM, StorageType.HYBRID]:
                    file_path = self.file_storage_path / "atoms" / f"{atom_id}.atom"
                    if file_path.exists():
                        file_path.unlink()
                
                if self.config.storage_type in [StorageType.SQLITE, StorageType.HYBRID]:
                    if self.db_connection:
                        cursor = self.db_connection.cursor()
                        cursor.execute("DELETE FROM atoms WHERE id = ?", (atom_id,))
                        self.db_connection.commit()
                
                # Remove from indices
                self._remove_from_indices(atom_id)
                
                # Remove from cache
                if atom_id in self.access_cache:
                    del self.access_cache[atom_id]
                
                # Update statistics
                self.stats.total_atoms -= 1
                if atom_id in self.memory_storage:
                    self.stats.memory_atoms -= 1
                else:
                    self.stats.disk_atoms -= 1
                
                return success
            
            except Exception as e:
                print(f"Deletion error: {e}")
                return False
    
    def backup_storage(self, backup_path: Optional[str] = None) -> bool:
        """Create a backup of the storage"""
        if not self.config.backup_enabled:
            return True
        
        try:
            if backup_path is None:
                timestamp = int(time.time())
                backup_path = self.file_storage_path / "backups" / f"backup_{timestamp}"
            
            backup_path = Path(backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup atoms
            atoms_backup_path = backup_path / "atoms"
            atoms_backup_path.mkdir(exist_ok=True)
            
            for atom_id in self._get_all_atom_ids():
                atom = self.retrieve_atom(atom_id)
                if atom:
                    atom_file = atoms_backup_path / f"{atom_id}.json"
                    with open(atom_file, 'w') as f:
                        json.dump(atom.to_dict(), f, default=str)
            
            # Backup indices
            indices_backup_path = backup_path / "indices"
            indices_backup_path.mkdir(exist_ok=True)
            
            for index_type, index_data in self.indices.items():
                index_file = indices_backup_path / f"{index_type.value}.json"
                # Convert sets to lists for JSON serialization
                serializable_index = {
                    key: list(value) for key, value in index_data.items()
                }
                with open(index_file, 'w') as f:
                    json.dump(serializable_index, f)
            
            # Backup configuration and statistics
            config_file = backup_path / "config.json"
            with open(config_file, 'w') as f:
                json.dump(asdict(self.config), f, default=str)
            
            stats_file = backup_path / "stats.json"
            with open(stats_file, 'w') as f:
                json.dump(asdict(self.stats), f, default=str)
            
            self.stats.last_backup = time.time()
            
            return True
        
        except Exception as e:
            print(f"Backup error: {e}")
            return False
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore storage from a backup"""
        try:
            backup_path = Path(backup_path)
            
            if not backup_path.exists():
                return False
            
            # Clear current storage
            self._clear_storage()
            
            # Restore atoms
            atoms_backup_path = backup_path / "atoms"
            if atoms_backup_path.exists():
                for atom_file in atoms_backup_path.glob("*.json"):
                    with open(atom_file, 'r') as f:
                        atom_dict = json.load(f)
                        atom = CQEAtom.from_dict(atom_dict)
                        self.store_atom(atom)
            
            # Restore indices
            indices_backup_path = backup_path / "indices"
            if indices_backup_path.exists():
                for index_file in indices_backup_path.glob("*.json"):
                    index_type_name = index_file.stem
                    try:
                        index_type = IndexType(index_type_name)
                        with open(index_file, 'r') as f:
                            index_data = json.load(f)
                            # Convert lists back to sets
                            self.indices[index_type] = defaultdict(set)
                            for key, value_list in index_data.items():
                                self.indices[index_type][key] = set(value_list)
                    except ValueError:
                        # Skip unknown index types
                        continue
            
            return True
        
        except Exception as e:
            print(f"Restore error: {e}")
            return False
    
    def optimize_storage(self) -> Dict[str, Any]:
        """Optimize storage performance and space usage"""
        optimization_results = {
            'atoms_moved': 0,
            'space_saved': 0,
            'indices_rebuilt': 0,
            'cache_optimized': False
        }
        
        try:
            with self.storage_lock:
                # Move frequently accessed atoms to memory
                if self.config.storage_type == StorageType.HYBRID:
                    frequent_atoms = sorted(
                        self.access_frequency.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:self.config.max_memory_size]
                    
                    for atom_id, _ in frequent_atoms:
                        if atom_id not in self.memory_storage:
                            atom = self._retrieve_from_database(atom_id)
                            if not atom:
                                atom = self._retrieve_from_file_system(atom_id)
                            
                            if atom:
                                self._store_in_memory(atom)
                                optimization_results['atoms_moved'] += 1
                
                # Rebuild indices for better performance
                old_index_sizes = {k: len(v) for k, v in self.indices.items()}
                self._rebuild_indices()
                new_index_sizes = {k: len(v) for k, v in self.indices.items()}
                
                optimization_results['indices_rebuilt'] = len(self.indices)
                
                # Optimize cache
                self._optimize_cache()
                optimization_results['cache_optimized'] = True
                
                # Compress old data if enabled
                if self.config.compression != CompressionType.NONE:
                    space_saved = self._compress_old_data()
                    optimization_results['space_saved'] = space_saved
        
        except Exception as e:
            print(f"Optimization error: {e}")
        
        return optimization_results
    
    def get_storage_statistics(self) -> StorageStats:
        """Get comprehensive storage statistics"""
        with self.storage_lock:
            # Update current statistics
            self.stats.total_atoms = len(self._get_all_atom_ids())
            self.stats.memory_atoms = len(self.memory_storage)
            self.stats.disk_atoms = self.stats.total_atoms - self.stats.memory_atoms
            
            # Calculate total size
            total_size = 0
            for atom_id in self._get_all_atom_ids():
                atom = self.retrieve_atom(atom_id)
                if atom:
                    total_size += len(pickle.dumps(atom))
            
            self.stats.total_size_bytes = total_size
            
            # Update index sizes
            self.stats.index_sizes = {
                index_type.value: len(index_data)
                for index_type, index_data in self.indices.items()
            }
            
            # Update access patterns
            self.stats.access_patterns = dict(self.access_frequency)
            
            return self.stats
    
    # Storage Backend Implementations
    def _store_in_memory(self, atom: CQEAtom) -> bool:
        """Store atom in memory"""
        self.memory_storage[atom.id] = atom
        return True
    
    def _store_in_file_system(self, atom: CQEAtom) -> bool:
        """Store atom in file system"""
        try:
            file_path = self.file_storage_path / "atoms" / f"{atom.id}.atom"
            
            # Serialize atom
            if self.config.compression == CompressionType.GZIP:
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(atom, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(atom, f)
            
            return True
        except Exception as e:
            print(f"File storage error: {e}")
            return False
    
    def _store_in_database(self, atom: CQEAtom) -> bool:
        """Store atom in SQLite database"""
        if not self.db_connection:
            return False
        
        try:
            cursor = self.db_connection.cursor()
            
            # Serialize complex data
            data_blob = pickle.dumps(atom.data)
            e8_blob = pickle.dumps(atom.e8_embedding)
            quad_str = json.dumps(atom.quad_encoding)
            parity_str = json.dumps(atom.parity_channels)
            metadata_str = json.dumps(atom.metadata)
            
            cursor.execute("""
                INSERT OR REPLACE INTO atoms 
                (id, data, quad_encoding, e8_embedding, parity_channels, governance_state, 
                 timestamp, parent_id, metadata, size_bytes, created_at, accessed_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                atom.id, data_blob, quad_str, e8_blob, parity_str, atom.governance_state,
                atom.timestamp, atom.parent_id, metadata_str, len(data_blob),
                time.time(), time.time(), self.access_frequency.get(atom.id, 0)
            ))
            
            self.db_connection.commit()
            return True
        
        except Exception as e:
            print(f"Database storage error: {e}")
            return False
    
    def _store_compressed(self, atom: CQEAtom) -> bool:
        """Store atom with compression"""
        try:
            file_path = self.file_storage_path / "atoms" / f"{atom.id}.atom.gz"
            
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(atom, f)
            
            return True
        except Exception as e:
            print(f"Compressed storage error: {e}")
            return False
    
    def _store_encrypted(self, atom: CQEAtom) -> bool:
        """Store atom with encryption"""
        # Placeholder for encryption implementation
        return self._store_in_file_system(atom)
    
    def _retrieve_from_memory(self, atom_id: str) -> Optional[CQEAtom]:
        """Retrieve atom from memory"""
        return self.memory_storage.get(atom_id)
    
    def _retrieve_from_file_system(self, atom_id: str) -> Optional[CQEAtom]:
        """Retrieve atom from file system"""
        try:
            file_path = self.file_storage_path / "atoms" / f"{atom_id}.atom"
            
            if not file_path.exists():
                # Try compressed version
                file_path = self.file_storage_path / "atoms" / f"{atom_id}.atom.gz"
                if file_path.exists():
                    with gzip.open(file_path, 'rb') as f:
                        return pickle.load(f)
                return None
            
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        except Exception as e:
            print(f"File retrieval error: {e}")
            return None
    
    def _retrieve_from_database(self, atom_id: str) -> Optional[CQEAtom]:
        """Retrieve atom from SQLite database"""
        if not self.db_connection:
            return None
        
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT * FROM atoms WHERE id = ?", (atom_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Deserialize data
            (id, data_blob, quad_str, e8_blob, parity_str, governance_state,
             timestamp, parent_id, metadata_str, size_bytes, created_at, accessed_at, access_count) = row
            
            data = pickle.loads(data_blob)
            quad_encoding = tuple(json.loads(quad_str))
            e8_embedding = pickle.loads(e8_blob)
            parity_channels = json.loads(parity_str)
            metadata = json.loads(metadata_str)
            
            # Reconstruct atom
            atom = CQEAtom(
                data=data,
                quad_encoding=quad_encoding,
                parent_id=parent_id,
                metadata=metadata
            )
            
            # Set computed properties
            atom.id = id
            atom.e8_embedding = e8_embedding
            atom.parity_channels = parity_channels
            atom.governance_state = governance_state
            atom.timestamp = timestamp
            
            return atom
        
        except Exception as e:
            print(f"Database retrieval error: {e}")
            return None
    
    def _retrieve_compressed(self, atom_id: str) -> Optional[CQEAtom]:
        """Retrieve compressed atom"""
        try:
            file_path = self.file_storage_path / "atoms" / f"{atom_id}.atom.gz"
            
            if not file_path.exists():
                return None
            
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        
        except Exception as e:
            print(f"Compressed retrieval error: {e}")
            return None
    
    def _retrieve_encrypted(self, atom_id: str) -> Optional[CQEAtom]:
        """Retrieve encrypted atom"""
        # Placeholder for decryption implementation
        return self._retrieve_from_file_system(atom_id)
    
    # Index Management
    def _update_indices(self, atom: CQEAtom):
        """Update all indices with new atom"""
        atom_id = atom.id
        
        # Quad index
        if IndexType.QUAD_INDEX in self.indices:
            quad_sig = self._quad_to_signature(atom.quad_encoding)
            self.indices[IndexType.QUAD_INDEX][quad_sig].add(atom_id)
        
        # E8 spatial index
        if IndexType.E8_SPATIAL_INDEX in self.indices:
            region_hash = self._e8_to_region_hash(atom.e8_embedding)
            self.indices[IndexType.E8_SPATIAL_INDEX][region_hash].add(atom_id)
        
        # Content index
        if IndexType.CONTENT_INDEX in self.indices:
            content_hash = self._compute_content_hash(atom.data)
            self.indices[IndexType.CONTENT_INDEX][content_hash].add(atom_id)
        
        # Temporal index
        if IndexType.TEMPORAL_INDEX in self.indices:
            time_bucket = int(atom.timestamp // 3600)  # Hour buckets
            self.indices[IndexType.TEMPORAL_INDEX][time_bucket].add(atom_id)
        
        # Metadata index
        if IndexType.METADATA_INDEX in self.indices:
            for key, value in atom.metadata.items():
                meta_key = f"{key}:{value}"
                self.indices[IndexType.METADATA_INDEX][meta_key].add(atom_id)
        
        # Hash index
        if IndexType.HASH_INDEX in self.indices:
            self.indices[IndexType.HASH_INDEX][atom_id].add(atom_id)
    
    def _remove_from_indices(self, atom_id: str):
        """Remove atom from all indices"""
        for index_type, index_data in self.indices.items():
            for key, atom_set in index_data.items():
                atom_set.discard(atom_id)
    
    def _rebuild_indices(self):
        """Rebuild all indices from scratch"""
        # Clear existing indices
        for index_type in self.indices:
            self.indices[index_type] = defaultdict(set)
        
        # Rebuild from all atoms
        for atom_id in self._get_all_atom_ids():
            atom = self.retrieve_atom(atom_id)
            if atom:
                self._update_indices(atom)
    
    # Utility Methods
    def _quad_to_signature(self, quad_encoding: Tuple[int, int, int, int]) -> str:
        """Convert quad encoding to string signature"""
        return f"{quad_encoding[0]}{quad_encoding[1]}{quad_encoding[2]}{quad_encoding[3]}"
    
    def _e8_to_region_hash(self, e8_embedding: np.ndarray) -> str:
        """Convert E8 embedding to spatial region hash"""
        # Quantize to regions for spatial indexing
        quantized = (e8_embedding // 0.5).astype(int)
        return hashlib.md5(quantized.tobytes()).hexdigest()[:8]
    
    def _compute_content_hash(self, data: Any) -> str:
        """Compute hash of content data"""
        content_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _get_all_atom_ids(self) -> List[str]:
        """Get all atom IDs from all storage backends"""
        atom_ids = set()
        
        # From memory
        atom_ids.update(self.memory_storage.keys())
        
        # From file system
        if self.file_storage_path.exists():
            atoms_dir = self.file_storage_path / "atoms"
            if atoms_dir.exists():
                for file_path in atoms_dir.glob("*.atom"):
                    atom_ids.add(file_path.stem)
                for file_path in atoms_dir.glob("*.atom.gz"):
                    atom_ids.add(file_path.stem.replace('.atom', ''))
        
        # From database
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT id FROM atoms")
            atom_ids.update(row[0] for row in cursor.fetchall())
        
        return list(atom_ids)
    
    def _matches_query(self, atom: CQEAtom, query: Dict[str, Any]) -> bool:
        """Check if atom matches query criteria"""
        for key, value in query.items():
            if key == 'quad_encoding':
                if atom.quad_encoding != tuple(value):
                    return False
            elif key == 'governance_state':
                if atom.governance_state != value:
                    return False
            elif key == 'parent_id':
                if atom.parent_id != value:
                    return False
            elif key == 'metadata':
                for meta_key, meta_value in value.items():
                    if atom.metadata.get(meta_key) != meta_value:
                        return False
            elif key == 'timestamp_range':
                start_time, end_time = value
                if not (start_time <= atom.timestamp <= end_time):
                    return False
        
        return True
    
    def _update_access_stats(self, atom_id: str):
        """Update access statistics for an atom"""
        self.access_frequency[atom_id] += 1
        
        # Update database if using it
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                UPDATE atoms 
                SET accessed_at = ?, access_count = access_count + 1 
                WHERE id = ?
            """, (time.time(), atom_id))
            self.db_connection.commit()
    
    def _update_storage_stats(self, atom: CQEAtom, operation: str):
        """Update storage statistics"""
        if operation == "store":
            self.stats.total_atoms += 1
            if atom.id in self.memory_storage:
                self.stats.memory_atoms += 1
            else:
                self.stats.disk_atoms += 1
    
    def _add_to_cache(self, atom: CQEAtom):
        """Add atom to access cache"""
        if len(self.access_cache) >= self.cache_size:
            # Remove least frequently accessed item
            lfa_atom_id = min(self.access_cache.keys(), 
                             key=lambda x: self.access_frequency.get(x, 0))
            del self.access_cache[lfa_atom_id]
        
        self.access_cache[atom.id] = atom
    
    def _optimize_cache(self):
        """Optimize the access cache"""
        # Keep only most frequently accessed atoms
        if len(self.access_cache) > self.cache_size // 2:
            frequent_atoms = sorted(
                self.access_cache.items(),
                key=lambda x: self.access_frequency.get(x[0], 0),
                reverse=True
            )[:self.cache_size // 2]
            
            self.access_cache = dict(frequent_atoms)
    
    def _compress_old_data(self) -> int:
        """Compress old data to save space"""
        space_saved = 0
        
        # Compress atoms older than 30 days
        cutoff_time = time.time() - (30 * 24 * 3600)
        
        for atom_id in self._get_all_atom_ids():
            atom = self.retrieve_atom(atom_id)
            if atom and atom.timestamp < cutoff_time:
                # Move to compressed storage if not already compressed
                file_path = self.file_storage_path / "atoms" / f"{atom_id}.atom"
                compressed_path = self.file_storage_path / "atoms" / f"{atom_id}.atom.gz"
                
                if file_path.exists() and not compressed_path.exists():
                    original_size = file_path.stat().st_size
                    
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    compressed_size = compressed_path.stat().st_size
                    space_saved += original_size - compressed_size
                    
                    file_path.unlink()  # Remove original
        
        return space_saved
    
    def _clear_storage(self):
        """Clear all storage (used for restore)"""
        self.memory_storage.clear()
        self.access_cache.clear()
        self.access_frequency.clear()
        
        # Clear indices
        for index_type in self.indices:
            self.indices[index_type] = defaultdict(set)
        
        # Clear database
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute("DELETE FROM atoms")
            cursor.execute("DELETE FROM quad_index")
            cursor.execute("DELETE FROM e8_spatial_index")
            cursor.execute("DELETE FROM content_index")
            cursor.execute("DELETE FROM metadata_index")
            self.db_connection.commit()
    
    def _load_indices_from_disk(self):
        """Load indices from disk files"""
        indices_dir = self.file_storage_path / "indices"
        if not indices_dir.exists():
            return
        
        for index_file in indices_dir.glob("*.json"):
            try:
                index_type = IndexType(index_file.stem)
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                    self.indices[index_type] = defaultdict(set)
                    for key, value_list in index_data.items():
                        self.indices[index_type][key] = set(value_list)
            except (ValueError, json.JSONDecodeError):
                continue
    
    def _load_indices_from_database(self):
        """Load indices from database"""
        if not self.db_connection:
            return
        
        cursor = self.db_connection.cursor()
        
        # Load quad index
        cursor.execute("SELECT quad_signature, atom_id FROM quad_index")
        for quad_sig, atom_id in cursor.fetchall():
            self.indices[IndexType.QUAD_INDEX][quad_sig].add(atom_id)
        
        # Load other indices similarly...
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Placeholder for background task implementation
        pass

# Export main classes
__all__ = [
    'CQEStorageManager', 'StorageConfig', 'StorageStats',
    'StorageType', 'IndexType', 'CompressionType'
]
"""
CQE Core System - Complete Implementation
========================================

The definitive implementation of the Cartan Quadratic Equivalence (CQE) system
that integrates all mathematical frameworks into a unified computational system.

This module provides the complete CQE system with:
- E₈ lattice operations for geometric processing
- Sacred geometry guidance for binary operations
- Mandelbrot fractal storage with bit-level precision
- Universal atomic operations for any data type
- Comprehensive validation and testing

Author: CQE Development Team
Version: 1.0.0 Master
"""

import numpy as np
import hashlib
import struct
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Setup logging
logger = logging.getLogger(__name__)
