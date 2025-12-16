
from typing import Dict, Any, Tuple, Optional

def data_to_complex(s: str) -> complex:
    # Deterministic rolling-hash mapping to [-2,2]x[-2,2]
    h = 0
    for ch in s.encode("utf-8"):
        h = (h * 131 + ch) % (1<<32)
    real = ((h & 0xFFFF) / 0xFFFF)*4 - 2
    imag = (((h>>16) & 0xFFFF) / 0xFFFF)*4 - 2
    return complex(real, imag)

def mandelbrot_classify(c: complex, max_iter: int=512, period_max: int=12, tol: float=1e-6) -> Tuple[str, int, float]:
    """
    Return (behavior, escape_time_or_-1, lyapunov_estimate).
    behavior ∈ {"ESCAPING","PERIODIC","BOUNDED"} (BOUNDED includes boundary/chaotic undecided within budget).
    Lyapunov estimate uses λ ≈ (1/n) Σ log|f'(z_k)| with f'(z)=2z for z_{k+1}=z_k^2 + c.
    """
    z = 0j
    deriv = 1.0+0j  # derivative chain
    orbit = []
    for n in range(1, max_iter+1):
        deriv = 2*z * deriv  # f'(z) chain rule for z^2 + c
        z = z*z + c
        orbit.append(z)
        if (z.real*z.real + z.imag*z.imag) > 4.0:
            # ESCAPING
            lam = 0.0
            if n > 1:
                # Lyapunov estimate from derivative magnitude
                # protect against log(0)
                m = max(abs(deriv), 1e-12)
                lam = (1.0/n) * math.log(m)
            return "ESCAPING", n, lam
        # simple period detection up to period_max using near-equality
        if n > 10:  # wait a bit
            for p in range(1, min(period_max, n//2)+1):
                if abs(orbit[-1] - orbit[-1-p]) < tol:
                    lam = 0.0
                    if n > 1:
                        m = max(abs(deriv), 1e-12)
                        lam = (1.0/n) * math.log(m)
                    return "PERIODIC", p, lam
    # Did not escape and no short period detected within budget
    lam = 0.0
    if max_iter > 1:
        m = max(abs(deriv), 1e-12)
        lam = (1.0/max_iter) * math.log(m)
    return "BOUNDED", -1, lam

def analyze_string(s: str) -> Dict[str, Any]:
    c = data_to_complex(s)
    behavior, aux, lyap = mandelbrot_classify(c)
    return {"c_real": round(c.real,6), "c_imag": round(c.imag,6), "behavior": behavior, "aux": aux, "lyapunov": lyap}
