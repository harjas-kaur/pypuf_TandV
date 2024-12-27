import numpy as np

class PhysicalFactors:
    """
    Implements variations in MUX delay due to environmental factors such as temperature and voltage.
    """

    def __init__(self, temperature: float = 20, vdd: float = 1.35, Tfactor=False, Vfactor=False):
        """
        Initializes the PhysicalFactors instance.

        Parameters:
        - temperature (float): Environmental temperature in °C (default 20).
        - vdd (float): Supply voltage in volts (default 1.35).
        """
        if not (0 <= temperature <= 150):
            raise ValueError("Temperature should be between 0°C and 150°C.")
        if not (0.5 <= vdd <= 5):
            raise ValueError("VDD should be between 0.5V and 5V.")
        self.temperature = temperature
        self.vdd = vdd

    def temperature_dependencies(self) -> float:
        """
        Calculates temperature dependency based on the formula provided.
        Formula is for vdd=1.35v
        Returns:
        - float: Calculated temperature dependency factor.
        """
        temperature_value = np.power(
            (0.92 - 0.4 * np.sqrt(np.abs(0.03 * self.temperature + 1)) +
             0.4 * np.sqrt(np.abs(0.03 * self.temperature))),
            2
        )
        return temperature_value

    def voltage_dependencies(self) -> float:
        """
        Calculates voltage dependency based on the formula provided.
        Formula is for T= 20 deg Celcius
        Returns:
        - float: Calculated voltage dependency factor.
        """
        denominator = (self.vdd - 0.4548)
        fun=(1-0.0466*self.vdd)
        if denominator == 0:
            raise ValueError("Division by zero encountered in voltage dependencies calculation.")

        numerator = self.vdd* np.power(fun, 2)
        divisor = self.vdd * (1 - 0.0466 * self.vdd)

        return numerator / divisor

    def process(self, Tfactor=True, Vfactor=False) -> float:
        if Tfactor and not Vfactor:
            return self.temperature_dependencies()
        elif not Tfactor and Vfactor:
            return self.voltage_dependencies()
        elif Tfactor and Vfactor:
            print("Error: Combined function not yet defined.")
            return 1
        elif not Tfactor and not Vfactor:
            return 1  

        
