from temp_voltage import PhysicalFactors

test_cases = [
    {"temperature": 20, "vdd": 1.35},
    {"temperature": 80, "vdd": 1.35},
    {"temperature": 20, "vdd": 1.0},
    {"temperature": 80, "vdd": 1.0},
]

for i, params in enumerate(test_cases):
    print(f"\nTest Case {i+1}: {params}")
    for Tfactor, Vfactor, label in [
        (True, False, "Temperature only"),
        (False, True, "Voltage only"),
        (False, False, "No factor"),
    ]:
        try:
            pf = PhysicalFactors(
                temperature=params["temperature"],
                vdd=params["vdd"],
                Tfactor=Tfactor,
                Vfactor=Vfactor,
            )
            factor = pf.process(Tfactor, Vfactor)
            print(f"  {label} dependency factor: {factor}")
        except Exception as e:
            print(f"  {label} dependency factor: Exception: {e}")