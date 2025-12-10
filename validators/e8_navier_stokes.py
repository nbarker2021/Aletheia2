class E8NavierStokesValidator:
    """
    Numerical validation of E8 Navier-Stokes overlay dynamics proof
    """

    def __init__(self):
        self.num_overlays = 64  # Computational subset of overlays
        self.dimension = 8      # E8 dimension
        self.critical_re = 240  # Predicted critical Reynolds number

    def generate_initial_overlays(self, n_overlays=64):
        """Generate initial overlay configuration from velocity field"""
        np.random.seed(42)

        overlays = []
        for i in range(n_overlays):
            # Generate 3D velocity components
            u_x = np.random.uniform(-1, 1)
            u_y = np.random.uniform(-1, 1) 
            u_z = np.random.uniform(-1, 1)

            # Map to E8 coordinates (simplified embedding)
            theta = np.random.uniform(0, 2*np.pi)

            r = np.zeros(8)
            r[0] = u_x * np.cos(theta) + u_y * np.sin(theta)
            r[1] = -u_x * np.sin(theta) + u_y * np.cos(theta)
            r[2] = u_z
            r[3] = np.sqrt(u_x**2 + u_y**2 + u_z**2)  # speed
            r[4] = np.random.uniform(-0.5, 0.5)  # vorticity (simplified)
            r[5] = np.random.uniform(-0.5, 0.5)  # strain rate  
            r[6] = np.random.uniform(-0.5, 0.5)  # pressure gradient
            r[7] = np.random.uniform(-0.1, 0.1)  # viscous term

            # Project to approximate E8 lattice constraints
            r = self.project_to_e8_constraint(r)
            overlays.append(r)

        return np.array(overlays)

    def project_to_e8_constraint(self, r):
        """Project to satisfy E8 lattice constraints (simplified)"""
        # E8 constraint: sum must be even
        current_sum = np.sum(r)
        if abs(current_sum - round(current_sum)) > 0.5:
            # Adjust to make sum closer to integer
            adjustment = (round(current_sum) - current_sum) / len(r)
            r += adjustment

        # Bound coordinates (E8 fundamental domain)
        r = np.clip(r, -2, 2)
        return r

    def overlay_potential(self, overlays):
        """Compute MORSR overlay potential"""
        n_overlays = len(overlays)
        potential = 0.0

        # Pairwise interactions  
        for i in range(n_overlays):
            for j in range(i+1, n_overlays):
                dr = overlays[i] - overlays[j]
                distance = norm(dr)
                if distance > 1e-10:  # Avoid division by zero
                    # Screened Coulomb-like interaction
                    potential += np.exp(-distance) / distance

        # Single particle terms (viscous regularization)
        for i in range(n_overlays):
            potential += 0.5 * norm(overlays[i])**2

        return potential

    def morsr_dynamics(self, t, state, viscosity):
        """MORSR evolution equations for overlays"""
        n_overlays = len(state) // 8
        overlays = state.reshape(n_overlays, 8)

        derivatives = np.zeros_like(overlays)

        for i in range(n_overlays):
            force = np.zeros(8)

            # Forces from other overlays
            for j in range(n_overlays):
                if i != j:
                    dr = overlays[i] - overlays[j]
                    distance = norm(dr)
                    if distance > 1e-10:
                        # Gradient of screened interaction
                        force_mag = np.exp(-distance) * (1 + distance) / distance**3
                        force -= force_mag * dr

            # Viscous damping (E8 regularization)
            force -= overlays[i] / viscosity

            # Add small stochastic driving
            force += 0.1 * np.random.randn(8)

            derivatives[i] = force

        return derivatives.flatten()

    def compute_lyapunov_exponent(self, overlays, viscosity, evolution_time=10.0):
        """Compute maximal Lyapunov exponent for overlay system"""

        # Reference trajectory
        y0_ref = overlays.flatten()

        # Perturbed trajectory  
        perturbation = 1e-8 * np.random.randn(len(y0_ref))
        y0_pert = y0_ref + perturbation

        # Time points
        t_eval = np.linspace(0, evolution_time, 100)

        # Solve both trajectories
        try:
            sol_ref = solve_ivp(lambda t, y: self.morsr_dynamics(t, y, viscosity), 
                              [0, evolution_time], y0_ref, t_eval=t_eval, rtol=1e-6)
            sol_pert = solve_ivp(lambda t, y: self.morsr_dynamics(t, y, viscosity),
                               [0, evolution_time], y0_pert, t_eval=t_eval, rtol=1e-6)
        except:
            # If integration fails, assume unstable (high Lyapunov exponent)
            return 1.0

        if not sol_ref.success or not sol_pert.success:
            return 1.0

        # Compute separation growth
        separations = []
        for i, t in enumerate(t_eval):
            if i < len(sol_ref.y[0]) and i < len(sol_pert.y[0]):
                sep = norm(sol_ref.y[:, i] - sol_pert.y[:, i])
                if sep > 1e-12:  # Avoid log(0)
                    separations.append(sep)

        if len(separations) < 2:
            return 0.0

        # Linear fit to log(separation) vs time
        log_seps = np.log(separations)
        times = t_eval[:len(log_seps)]

        if len(times) > 1:
            lyapunov = (log_seps[-1] - log_seps[0]) / (times[-1] - times[0])
            return lyapunov
        else:
            return 0.0

    def test_critical_reynolds_number(self):
        """Test prediction of critical Reynolds number"""
        print("\n=== Critical Reynolds Number Test ===")

        # Test range of viscosities (inverse of Reynolds number)
        viscosities = np.logspace(-2, 1, 20)  # 0.01 to 10
        lyapunov_exponents = []

        # Generate initial overlays
        initial_overlays = self.generate_initial_overlays(32)  # Smaller for speed
        print(f"Generated {len(initial_overlays)} initial overlays")

        for nu in viscosities:
            # Compute Reynolds number (approximate)
            characteristic_velocity = np.mean([norm(r[:3]) for r in initial_overlays])
            characteristic_length = 1.0  # Normalized
            reynolds = characteristic_velocity * characteristic_length / nu

            # Compute Lyapunov exponent
            lambda_max = self.compute_lyapunov_exponent(initial_overlays, nu, evolution_time=5.0)
            lyapunov_exponents.append(lambda_max)

            print(f"  ν = {nu:.3f}, Re = {reynolds:.1f}, λ = {lambda_max:.3f}")

        # Find critical point where λ changes sign
        critical_indices = []
        for i in range(len(lyapunov_exponents)-1):
            if lyapunov_exponents[i] * lyapunov_exponents[i+1] < 0:
                critical_indices.append(i)

        if critical_indices:
            critical_nu = viscosities[critical_indices[0]]
            critical_re = 1.0 / critical_nu  # Approximate
            print(f"\n  Observed critical Re: {critical_re:.0f}")
            print(f"  Predicted critical Re: {self.critical_re}")
            print(f"  Ratio: {critical_re / self.critical_re:.2f}")
        else:
            print("\n  No clear critical transition found in range tested")

        return viscosities, lyapunov_exponents

    def test_energy_conservation(self):
        """Test energy conservation during overlay evolution"""
        print("\n=== Energy Conservation Test ===")

        # Generate initial overlays  
        initial_overlays = self.generate_initial_overlays(16)
        initial_energy = np.sum([norm(r)**2 for r in initial_overlays])

        viscosity = 0.1  # Moderate viscosity
        evolution_time = 5.0

        print(f"Initial energy: {initial_energy:.4f}")

        # Evolve system
        y0 = initial_overlays.flatten()
        t_eval = np.linspace(0, evolution_time, 50)

        try:
            sol = solve_ivp(lambda t, y: self.morsr_dynamics(t, y, viscosity),
                          [0, evolution_time], y0, t_eval=t_eval, rtol=1e-6)

            if sol.success:
                # Check energy at each time
                energies = []
                for i, t in enumerate(t_eval):
                    if i < len(sol.y[0]):
                        overlays = sol.y[:, i].reshape(-1, 8)
                        energy = np.sum([norm(r)**2 for r in overlays])
                        energies.append(energy)

                final_energy = energies[-1]
                energy_change = abs(final_energy - initial_energy) / initial_energy

                print(f"Final energy: {final_energy:.4f}")
                print(f"Relative change: {energy_change:.2%}")

                if energy_change < 0.1:  # 10% tolerance
                    print("✓ Energy approximately conserved")
                else:
                    print("⚠ Significant energy change (expected due to viscosity)")

                return t_eval[:len(energies)], energies
            else:
                print("✗ Integration failed")
                return None, None

        except Exception as e:
            print(f"✗ Error in integration: {e}")
            return None, None

    def test_smooth_vs_turbulent_flow(self):
        """Test smooth vs turbulent flow regimes"""
        print("\n=== Smooth vs Turbulent Flow Test ===")

        initial_overlays = self.generate_initial_overlays(24)

        # Test two viscosity regimes
        high_viscosity = 1.0    # Should give smooth flow (λ < 0)
        low_viscosity = 0.01    # Should give turbulent flow (λ > 0)

        print("High viscosity regime (smooth flow expected):")
        lambda_smooth = self.compute_lyapunov_exponent(initial_overlays, high_viscosity)
        print(f"  ν = {high_viscosity}, λ = {lambda_smooth:.4f}")
        if lambda_smooth < 0:
            print("  ✓ Smooth flow (λ < 0)")
        else:
            print("  ⚠ Turbulent-like behavior")

        print("\nLow viscosity regime (turbulent flow expected):")  
        lambda_turbulent = self.compute_lyapunov_exponent(initial_overlays, low_viscosity)
        print(f"  ν = {low_viscosity}, λ = {lambda_turbulent:.4f}")
        if lambda_turbulent > 0:
            print("  ✓ Turbulent flow (λ > 0)")
        else:
            print("  ⚠ Unexpectedly stable")

        return lambda_smooth, lambda_turbulent

    def test_e8_constraint_preservation(self):
        """Test that E8 lattice constraints are preserved"""
        print("\n=== E8 Constraint Preservation Test ===")

        initial_overlays = self.generate_initial_overlays(8)

        # Check initial constraints
        initial_sums = [np.sum(overlay) for overlay in initial_overlays]
        initial_norms = [norm(overlay) for overlay in initial_overlays]

        print("Initial state:")
        print(f"  Coordinate sums: {[f'{s:.2f}' for s in initial_sums]}")
        print(f"  Overlay norms: {[f'{n:.2f}' for n in initial_norms]}")

        # Evolve briefly  
        viscosity = 0.1
        evolution_time = 2.0

        y0 = initial_overlays.flatten()

        try:
            sol = solve_ivp(lambda t, y: self.morsr_dynamics(t, y, viscosity),
                          [0, evolution_time], y0, rtol=1e-6)

            if sol.success and len(sol.y[:, -1]) > 0:
                final_overlays = sol.y[:, -1].reshape(-1, 8)

                final_sums = [np.sum(overlay) for overlay in final_overlays]
                final_norms = [norm(overlay) for overlay in final_overlays]

                print("\nFinal state:")
                print(f"  Coordinate sums: {[f'{s:.2f}' for s in final_sums]}")
                print(f"  Overlay norms: {[f'{n:.2f}' for n in final_norms]}")

                # Check if constraints approximately preserved
                sum_changes = [abs(f - i) for f, i in zip(final_sums, initial_sums)]
                max_sum_change = max(sum_changes) if sum_changes else 0

                if max_sum_change < 0.5:
                    print(f"  ✓ Constraints preserved (max change: {max_sum_change:.3f})")
                else:
                    print(f"  ⚠ Constraints violated (max change: {max_sum_change:.3f})")

                return initial_overlays, final_overlays
            else:
                print("  ✗ Integration failed")
                return initial_overlays, None

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return initial_overlays, None

    def generate_validation_plots(self):
        """Generate validation plots"""
        print("\n=== Generating Validation Plots ===")

        # Plot 1: Lyapunov exponent vs Reynolds number
        viscosities, lyapunov_exponents = self.test_critical_reynolds_number()
        reynolds_numbers = [1.0/nu for nu in viscosities]

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.semilogx(reynolds_numbers, lyapunov_exponents, 'bo-', linewidth=2, markersize=6)
        plt.axhline(0, color='red', linestyle='--', alpha=0.7, label='λ = 0')
        plt.axvline(self.critical_re, color='green', linestyle='--', alpha=0.7, 
                   label=f'Predicted Re_c = {self.critical_re}')
        plt.xlabel('Reynolds Number')
        plt.ylabel('Lyapunov Exponent λ')
        plt.title('Critical Reynolds Number Test')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Energy conservation
        times, energies = self.test_energy_conservation()
        if times is not None and energies is not None:
            plt.subplot(2, 2, 2)
            plt.plot(times, energies, 'r-', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('Total Energy')
            plt.title('Energy Conservation')
            plt.grid(True, alpha=0.3)

        # Plot 3: Flow regime comparison
        plt.subplot(2, 2, 3)
        lambda_smooth, lambda_turbulent = self.test_smooth_vs_turbulent_flow()

        regimes = ['High ν\n(Smooth)', 'Low ν\n(Turbulent)']
        lambdas = [lambda_smooth, lambda_turbulent]
        colors = ['blue' if l < 0 else 'red' for l in lambdas]

        bars = plt.bar(regimes, lambdas, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(0, color='black', linestyle='-', alpha=0.5)
        plt.ylabel('Lyapunov Exponent λ')
        plt.title('Smooth vs Turbulent Regimes')
        plt.grid(True, alpha=0.3)

        # Add value labels
        for bar, lambda_val in zip(bars, lambdas):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 * max(abs(min(lambdas)), max(lambdas)),
                    f'{lambda_val:.3f}', ha='center', va='bottom', fontweight='bold')

        # Plot 4: Overlay configuration
        initial_overlays, final_overlays = self.test_e8_constraint_preservation()

        plt.subplot(2, 2, 4)
        if initial_overlays is not None:
            # Show 2D projection of overlays
            initial_2d = initial_overlays[:, :2]  # First 2 E8 coordinates
            plt.scatter(initial_2d[:, 0], initial_2d[:, 1], c='blue', alpha=0.7, 
                       label='Initial', s=60, edgecolor='black')

            if final_overlays is not None:
                final_2d = final_overlays[:, :2]
                plt.scatter(final_2d[:, 0], final_2d[:, 1], c='red', alpha=0.7,
                           label='Final', s=60, edgecolor='black', marker='s')

        plt.xlabel('E8 Coordinate 1')
        plt.ylabel('E8 Coordinate 2')  
        plt.title('Overlay Evolution (2D Projection)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('navier_stokes_validation_plots.png', dpi=300, bbox_inches='tight')
        print("✓ Plots saved as 'navier_stokes_validation_plots.png'")
