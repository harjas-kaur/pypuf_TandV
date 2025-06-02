import numpy as np
import traceback
from base import LTFArray
from temp_voltage import PhysicalFactors
from bistable import XORBistableRingPUF
from delay import (
    XORArbiterPUF,
    FeedForwardArbiterPUF,
    XORFeedForwardArbiterPUF,
    ArbiterPUF,
    LightweightSecurePUF,
)

def print_physical_factors(temperature, vdd):
    factors = []
    for Tfactor, Vfactor, label in [
        (True, False, "Temperature only"),
        (False, True, "Voltage only"),
        (False, False, "No factor"),
    ]:
        try:
            pf = PhysicalFactors(
                temperature=temperature,
                vdd=vdd,
                Tfactor=Tfactor,
                Vfactor=Vfactor,
            )
            factor = pf.process(Tfactor, Vfactor)
            factors.append((label, factor))
        except Exception as e:
            factors.append((label, f"Exception: {e}"))
    return factors

def print_table(headers, rows):
    col_widths = [max(len(str(x)) for x in col) for col in zip(*([headers] + rows))]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-+-".join('-' * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(x) if isinstance(x, list) else x for x in row]))
        
def get_biases_and_factors(puf):
    if hasattr(puf, "simulations"):
        biases = []
        tfactors = []
        vfactors = []
        for sub in puf.simulations:
            bias = getattr(sub, "bias", "No bias")
            if isinstance(bias, np.ndarray):
                bias = np.array2string(bias.flatten(), precision=3, separator=',')
            biases.append(bias)
            tfactors.append(getattr(sub, "T_factor", "N/A"))
            vfactors.append(getattr(sub, "V_factor", "N/A"))
        return biases, tfactors, vfactors
    else:
        bias = getattr(puf, "bias", "No bias")
        if isinstance(bias, np.ndarray):
            bias = np.array2string(bias.flatten(), precision=3, separator=',')
        return [bias], [getattr(puf, "T_factor", "N/A")], [getattr(puf, "V_factor", "N/A")]
    
def test_puf(puf_name, puf_factory, extra_args=None):
    print(f"\n--- Testing {puf_name} ---")
    n = 8
    k = 4 if "XOR" in puf_name else 1
    challenges = np.random.choice([-1, 1], size=(10, n))
    test_cases = [
        {"temperature": 20, "vdd": 1.35},
        {"temperature": 80, "vdd": 1.35},
        {"temperature": 20, "vdd": 1.0},
        {"temperature": 80, "vdd": 1.0},
    ]
    headers = [
        "Test Case", "Temperature", "Vdd",
        "T_factor(s)", "V_factor(s)",
        "Dependency (T)", "Dependency (V)", "Dependency (None)",
        "PUF Bias(es)", "Sample Responses"
    ]
    rows = []
    for i, params in enumerate(test_cases):
        try:
            factors = print_physical_factors(params["temperature"], params["vdd"])
            if extra_args:
                puf = puf_factory(
                    n=n,
                    k=k,
                    challenges=challenges,
                    temperature=params["temperature"],
                    vdd=params["vdd"],
                    **extra_args
                )
            else:
                puf = puf_factory(
                    n=n,
                    k=k,
                    challenges=challenges,
                    temperature=params["temperature"],
                    vdd=params["vdd"]
                )
            # Try to get bias attribute
            biases, tfactors, vfactors = get_biases_and_factors(puf)
            responses = puf.eval(challenges)
            sample_responses = np.array2string(responses.flatten()[:5], separator=',')
            rows.append([
                f"{i+1}",
                params["temperature"],
                params["vdd"],
                tfactors,
                vfactors,
                factors[0][1],
                factors[1][1],
                factors[2][1],
                biases,
                sample_responses
            ])
        except Exception as e:
            rows.append([
                f"{i+1}",
                params["temperature"],
                params["vdd"],
                "-", "-", "-", "-", "-", "-", f"Exception: {e}"
            ])
    print_table(headers, rows)

if __name__ == "__main__":
    # XORBistableRingPUF
    def bistable_factory(n, k, challenges, temperature, vdd, **kwargs):
        weights = np.random.normal(0, 1, (k, n + 1))
        return XORBistableRingPUF(
            n=n,
            k=k,
            weights=weights,
            temperature=temperature,
            vdd=vdd
        )
    test_puf("XORBistableRingPUF", bistable_factory)

    # XORArbiterPUF
    def xorarbiter_factory(n, k, challenges, temperature, vdd, **kwargs):
        return XORArbiterPUF(
            n=n,
            k=k,
            temperature=temperature,
            vdd=vdd
        )
    test_puf("XORArbiterPUF", xorarbiter_factory)

    # FeedForwardArbiterPUF
    def ff_factory(n, k, challenges, temperature, vdd, **kwargs):
        ff = [(2, 5), (4, 7)]
        return FeedForwardArbiterPUF(
            n=n,
            ff=ff,
            temperature=temperature,
            vdd=vdd
        )
    test_puf("FeedForwardArbiterPUF", ff_factory)

    # XORFeedForwardArbiterPUF
    def xorff_factory(n, k, challenges, temperature, vdd, **kwargs):
        ff = [[(2, 5)], [(4, 7)], [(1, 6)]]
        return XORFeedForwardArbiterPUF(
            n=n,
            k=3,
            ff=ff,
            temperature=temperature,
            vdd=vdd
        )
    test_puf("XORFeedForwardArbiterPUF", xorff_factory)

    # ArbiterPUF
    def arbiter_factory(n, k, challenges, temperature, vdd, **kwargs):
        return ArbiterPUF(
            n=n,
            temperature=temperature,
            vdd=vdd
        )
    test_puf("ArbiterPUF", arbiter_factory)

   