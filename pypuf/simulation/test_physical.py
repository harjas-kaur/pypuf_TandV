import numpy as np
import matplotlib.pyplot as plt
from temp_voltage import PhysicalFactors

def main():
    """
    Validates the PhysicalFactors class by modifying a normal distribution
    representing delays with temperature and voltage dependencies.
    """
    # Generate a normal distribution of delays (mean=1e-9, std=1e-10)
    mean_delay = 1e-9  
    std_delay = 1e-10  
    num_samples = 1000  
    np.random.seed(42)
    delays = np.random.normal(mean_delay, std_delay, num_samples)
    print(f"Original Delays: Mean={np.mean(delays):.2e}, Std={np.std(delays):.2e}")

    # Initialize the PhysicalFactors instance with specific temperature and voltage
    temperature = 50  
    vdd = 1.2  
    physical_factors = PhysicalFactors(temperature=temperature, vdd=vdd)

    # Apply temperature dependencies
    print(physical_factors.process(Tfactor=True, Vfactor=False))
    print(physical_factors.process(Tfactor=False, Vfactor=True))
    temperature_factor = physical_factors.process(Tfactor=True, Vfactor=True)
    modified_delays_temperature = delays * temperature_factor
    print(f"After Temperature Adjustment: Mean={np.mean(modified_delays_temperature):.2e}, Std={np.std(modified_delays_temperature):.2e}")

    # Apply voltage dependencies
    voltage_factor=physical_factors.process(Tfactor=False, Vfactor=True)
    modified_delays_voltage = delays * voltage_factor
    print(f"After Voltage Adjustment: Mean={np.mean(modified_delays_voltage):.2e}, Std={np.std(modified_delays_voltage):.2e}")

    # Visualize the results

    plt.figure(figsize=(10, 6))
    plt.hist(delays, bins=30, alpha=0.5, label='Original Delays', color='blue')
    plt.hist(modified_delays_temperature, bins=30, alpha=0.5, label='Temperature Adjusted', color='green')
    plt.hist(modified_delays_voltage, bins=30, alpha=0.5, label='Voltage Adjusted', color='red')
    plt.title(f"Delay Distribution Modification (Temp={temperature}°C, VDD={vdd}V)")
    plt.xlabel('Delay (s)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    main()
