class HelicalState:
    """State of the helical integrator combining all four rotation modes"""
    poloidal_phase: float = 0.0
    toroidal_phase: float = 0.0
    meridional_phase: float = 0.0
    helical_phase: float = 0.0
    coupling: float = GRAVITATIONAL_COUPLING
    
    def advance(self, dt: float = 1.0) -> 'HelicalState':
        """Advance the helical state by one time step"""
        # Each mode advances at different rate modulated by 0.03
        return HelicalState(
            poloidal_phase=self.poloidal_phase + dt * self.coupling,
            toroidal_phase=self.toroidal_phase + dt * self.coupling * 2,
            meridional_phase=self.meridional_phase + dt * self.coupling * 3,
            helical_phase=self.helical_phase + dt * self.coupling * 4,
            coupling=self.coupling
        )
    
    def get_combined_rotation(self) -> np.ndarray:
        """Get combined rotation matrix for all four modes"""
        # 8D rotation combining all modes
        rotation = np.eye(E8_DIMENSION)
        
        # Poloidal (dims 0-1)
        c, s = np.cos(self.poloidal_phase), np.sin(self.poloidal_phase)
        rotation[0:2, 0:2] = [[c, -s], [s, c]]
        
        # Toroidal (dims 2-3)
        c, s = np.cos(self.toroidal_phase), np.sin(self.toroidal_phase)
        rotation[2:4, 2:4] = [[c, -s], [s, c]]
        
        # Meridional (dims 4-5)
        c, s = np.cos(self.meridional_phase), np.sin(self.meridional_phase)
        rotation[4:6, 4:6] = [[c, -s], [s, c]]
        
        # Helical (dims 6-7)
        c, s = np.cos(self.helical_phase), np.sin(self.helical_phase)
        rotation[6:8, 6:8] = [[c, -s], [s, c]]
        
        return rotation

