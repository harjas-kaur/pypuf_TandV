import numpy as np
from base import LTFArray
from temp_voltage import PhysicalFactors

def main():
    # Parameters
    k = 4  # Number of LTFs
    n = 8  # Challenge length
    weight_array = np.random.normal(0, 1, (k, n))
    bias = np.ones((k, 1)) * 0.5  # Initial bias

    # Challenges
    num_samples = 10
    challenges = np.random.choice([-1, 1], size=(num_samples, n))

    # Test different environmental conditions
    test_cases = [
        {"temperature": 20, "vdd": 1.35, "Tfactor": False, "Vfactor": False},
        {"temperature": 80, "vdd": 1.35, "Tfactor": True, "Vfactor": False},
        {"temperature": 20, "vdd": 1.0, "Tfactor": False, "Vfactor": True},
        {"temperature": 80, "vdd": 1.0, "Tfactor": True, "Vfactor": True},
    ]

    for i, params in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {params}")
        pf = PhysicalFactors(
            temperature=params["temperature"],
            vdd=params["vdd"],
            Tfactor=params["Tfactor"],
            Vfactor=params["Vfactor"]
        )
        factor = pf.process(params["Tfactor"], params["Vfactor"])
        print(f"Physical factor: {factor:.4f}")

        # Omit T_factor and V_factor to test auto-detection
        puf = LTFArray(
            weight_array=weight_array.copy(),
            transform=lambda x, k: np.broadcast_to(x[:, None, :], (x.shape[0], k, x.shape[1])),
            bias=bias.copy(),
            temperature=params["temperature"],
            vdd=params["vdd"]
            # T_factor and V_factor are omitted here!
        )

        # Print the actual bias used in the PUF
        print("PUF bias after scaling:", puf.bias.flatten())
        # Print auto-detected factors
        print(f"Auto-detected T_factor: {puf.T_factor}, V_factor: {puf.V_factor}")

        # Evaluate the PUF on the challenges
        responses = puf.eval(challenges)
        print("Sample responses:", responses.flatten()[:5])

if __name__ == "__main__":
    main()
    