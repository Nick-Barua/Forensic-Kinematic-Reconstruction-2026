import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_coeffs
from scipy.special import erf

# 1. BIOMECHANICAL & KINEMATIC CONSTANTS (Source: Sensors V15)
HIC_50 = 900.0  # 50th percentile AIS 4+ threshold
SIGMA_LN = 0.37 # Lognormal shape parameter
V_ORGAN_MEAN = 1560.0 # cm^3 [cite: 1797]
K_SEDAN = 0.041 # m*(km/h)^-1.5 [cite: 1870]
SEED = 42

# 2. CORE PHYSICAL EQUATIONS [cite: 1780, 1787, 1798, 1808]
def calculate_velocity(d, k):
    return (d / k)**(2/3)

def calculate_hic_ais_prob(hic):
    # Mertz-Prasad lognormal probit model [cite: 1787, 1803]
    beta1 = 1.0 / SIGMA_LN
    beta0 = -beta1 * np.log(HIC_50)
    z = beta0 + beta1 * np.log(hic)
    return 0.5 * (1 + erf(z / np.sqrt(2)))

def calculate_hepatic_stress(v_kmh, theta_deg=0):
    # Linear force scaling derived from baseline [cite: 1798, 2331]
    # Baseline: 4.208 kN at 49.3 km/h
    f_lateral = 4.208 * (v_kmh / 49.3) 
    constant = 0.78 # MPa*cm/kN
    stress_mpa = (constant * f_lateral * np.cos(np.radians(theta_deg))) / (V_ORGAN_MEAN**(1/3))
    return stress_mpa * 1000.0 # Convert to kPa

# 3. MONTE CARLO PROPAGATION (n=10,000) [cite: 1868-1875]
def run_monte_carlo():
    np.random.seed(SEED)
    n = 10000
    
    # Input: d ~ N(14.2, 0.5^2) [cite: 1869]
    d_samples = np.random.normal(14.2, 0.5, n)
    
    # Propagate through Stage 1 & 2
    v_samples = calculate_velocity(d_samples, K_SEDAN)
    
    # HIC sensitivity: Base 820 with velocity-dependent scaling
    hic_samples = 820 * (v_samples / 49.3)**2.5
    
    prob_samples = calculate_hic_ais_prob(hic_samples)
    stress_samples = calculate_hepatic_stress(v_samples)
    
    # Print Results [cite: 1890]
    print(f"--- MONTE CARLO RESULTS (n={n}, seed={SEED}) ---")
    print(f"Velocity: {np.mean(v_samples):.2f} ± {np.std(v_samples):.2f} km/h")
    print(f"95% CI: [{np.percentile(v_samples, 2.5):.1f}, {np.percentile(v_samples, 97.5):.1f}]")
    print(f"Hepatic Stress: {np.mean(stress_samples):.1f} ± {np.std(stress_samples):.1f} kPa")
    print(f"95% CI: [{np.percentile(stress_samples, 2.5):.1f}, {np.percentile(stress_samples, 97.5):.1f}]")

# 4. SIGNAL PROCESSING ANALYSIS (SG vs CFD) [cite: 1757-1765]
def plot_frequency_response():
    fs = 60.0 # 60 Hz sampling
    f = np.linspace(0.1, 30.0, 500)
    w = 2 * np.pi * f
    dt = 1/fs
    
    # Central Finite Difference (CFD) Gain
    # CFD exhibits higher high-frequency noise energy retention [cite: 1761, 2520]
    gain_cfd = np.abs(np.sin(w * dt) / (w * dt))
    
    # Savitzky-Golay (Degree 3, 11-point) [cite: 1759]
    # SG provides ~15 dB suppression relative to CFD in 10-30Hz band [cite: 1763]
    coeffs = savgol_coeffs(11, 3, deriv=1, delta=dt)
    gain_sg = np.abs([np.sum(coeffs * np.exp(-1j * wk * dt * np.arange(-(11//2), 11//2 + 1))) for wk in w])
    gain_sg_norm = gain_sg / w # Normalise against ideal differentiator

    plt.figure(figsize=(8, 5))
    plt.plot(f, gain_cfd, 'r', label='Central Finite Difference (CFD)')
    plt.plot(f, gain_sg_norm, 'b', label='Savitzky-Golay (deg 3, 11-pt)')
    plt.axvline(8, color='k', linestyle='--', alpha=0.5, label='8 Hz CoG Energy Limit')
    plt.ylabel('Normalised Gain')
    plt.xlabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.title("Figure S1. Noise Suppression Advantage")
    plt.show()

if __name__ == "__main__":
    run_monte_carlo()
    plot_frequency_response()
