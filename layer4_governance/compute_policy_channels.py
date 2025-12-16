def compute_policy_channels(embedding):
    """Compute 8 policy channels from embedding using D8 harmonic basis"""
    v = np.array(embedding)
    
    # D8 harmonic basis (8 channels: DC, Nyquist, 3 cosine-sine pairs)
    channels = np.zeros(8)
    
    # Channel 0: DC (average)
    channels[0] = np.mean(v)
    
    # Channel 1: Nyquist (alternating pattern)
    channels[1] = np.mean([(-1)**i * v[i] for i in range(8)])
    
    # Channels 2-7: Fourier-like components
    for k in range(1, 4):  # 3 harmonic pairs
        cos_sum = sum(v[i] * np.cos(2 * np.pi * k * i / 8) for i in range(8))
        sin_sum = sum(v[i] * np.sin(2 * np.pi * k * i / 8) for i in range(8))
        channels[2*k] = cos_sum / 4
        channels[2*k+1] = sin_sum / 4
    
    return channels.tolist()

# Create overlay states for all test scenarios
