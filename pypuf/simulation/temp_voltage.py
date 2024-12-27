import numpy as np

class PhysicalFactors:
    """
    Implements variations in MUX delay due to environmental factors such as temperature and voltage.
    """

    def __init__(self, temperature: float = 20, vdd: float = 1.35):
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

    def temperature_dependencies(self, k: float = 1, td: float = 1e-9, return_k: bool = False) -> float:
        """
        Calculates delay or proportional constant based on temperature.

        Parameters:
        - k (float): Proportional constant (default 1).
        - td (float): Initial delay in seconds (default 1ns).
        - return_k (bool): If True, return proportional constant; otherwise, return delay.

        Returns:
        - float: Calculated delay or proportional constant.
        """
        temperature_value = np.power(
            (0.92 - 0.4 * np.sqrt(np.abs(0.03 * self.temperature + 1)) +
             0.4 * np.sqrt(np.abs(0.03 * self.temperature))),
            2
        )
        return (td * temperature_value) if return_k else (k / temperature_value)

    def voltage_dependencies(self, k: float = 1, td: float = 1e-9, return_k: bool = False) -> float:
        """
        Calculates delay or proportional constant based on voltage.

        Parameters:
        - k (float): Proportional constant (default 1).
        - td (float): Initial delay in seconds (default 1ns).
        - return_k (bool): If True, return proportional constant; otherwise, return delay.

        Returns:
        - float: Calculated delay or proportional constant.
        """
        denominator = (self.vdd - 0.6257)
        if denominator == 0:
            raise ValueError("Division by zero encountered in voltage dependencies calculation.")

        numerator = td * np.power(denominator, 2)
        divisor = self.vdd * (1 - 0.0466 * self.vdd)

        return numerator / divisor if return_k else (k * divisor / np.power(denominator, 2))
