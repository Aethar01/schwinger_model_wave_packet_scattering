
import numpy as np
import math


def compute_wave_packet_coefficients(N, k_mean, sigma, mass, center_n):
    """
    Computes the wave packet coefficients in position space for the staggered model.
    Follows Eq. (13) of the paper.
    """
    coefficients = np.zeros(N, dtype=complex)

    start_idx = -int(np.floor(N/4))
    end_idx = int(np.ceil(N/4))

    print(f"DEBUG: N={N}, k_mean={k_mean:.3f}, sigma={
          sigma}, center={center_n}")
    print(f"DEBUG: Loop range: {start_idx} to {end_idx}")

    for j in range(start_idx, end_idx):
        k = (2 * np.pi / N) * j

        # Energy and v_k
        wk = np.sqrt(mass**2 + np.sin(k)**2)

        # Handle singularity for massless case at k=0
        if wk < 1e-9:
            vk = 0.0
            factor = 1.0
        else:
            vk = np.sin(k) / (mass + wk)
            factor = np.sqrt((mass + wk) / wk)

        arg = -(k - k_mean)**2 / (4 * sigma**2)
        phi_k = np.exp(arg) * np.exp(-1j * k * center_n)

        print(f"DEBUG: j={j}, k={k:.3f}, wk={wk:.3f}, vk={
              vk:.3f}, phi_k_mag={np.abs(phi_k):.3f}")

        for n in range(N):
            if n % 2 == 0:
                proj_factor = 1.0
            else:
                proj_factor = vk

            term = phi_k * factor * np.exp(1j * k * n) * proj_factor
            coefficients[n] += term

    # Normalize
    norm = np.linalg.norm(coefficients)
    if norm > 0:
        coefficients /= norm

    return coefficients


N = 8
mass = 0.0
sigma = 0.5
k_mag = np.pi/4
center_left = math.floor((N-1) * 0.25)

print("\n--- Computing Phi_C ---")
phi_c = compute_wave_packet_coefficients(N, k_mag, sigma, mass, center_left)
print("\nPhi_C:", np.round(phi_c, 3))
print("Abs:", np.round(np.abs(phi_c), 3))
