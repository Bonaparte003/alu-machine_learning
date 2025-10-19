#!/usr/bin/env python3


"""Class that represents a normal distribution"""


class Normal:
    """Class that represents a normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution

        Args:
            data: list of data to estimate the distribution
            mean: mean of the distribution
            stddev: standard deviation of the distribution
        """
        if data is None:
            # Use provided mean and stddev
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            # Calculate mean and stddev from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean
            self.mean = sum(data) / len(data)

            # Calculate standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value

        Args:
            x: x-value

        Returns:
            z-score of x
        """
        # Z-score formula: z = (x - μ) / σ
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score

        Args:
            z: z-score

        Returns:
            x-value of z
        """
        # X-value formula: x = μ + z × σ
        x = self.mean + z * self.stddev
        return x

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value

        Args:
            x: x-value

        Returns:
            PDF value for x
        """
        # Normal PDF formula: f(x) = (1/σ√(2π)) × e^(-(x-μ)²/(2σ²))
        coefficient = 1 / (self.stddev * (2 * 3.141592653589793) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)

        # Calculate e^exponent using Taylor series
        exp_value = 1.0
        term = 1.0
        for i in range(1, 200):
            term *= exponent / i
            exp_value += term
            if abs(term) < 1e-15:
                break
        pdf_value = coefficient * exp_value

        return pdf_value

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value

        Args:
            x: x-value

        Returns:
            CDF value for x
        """
        # Convert to standard normal (z-score)
        z = self.z_score(x)

        # Use approximation for standard normal CDF
        # This is the Abramowitz and Stegun approximation
        if z < -6:
            return 0.0
        if z > 6:
            return 1.0

        # Abramowitz and Stegun approximation for Φ(z)
        t = 1 / (1 + 0.2316419 * abs(z))

        # Calculate e^(-z²/2) using Taylor series
        exp_arg = -z * z / 2
        exp_value = 1.0
        term = 1.0
        for i in range(1, 200):
            term *= exp_arg / i
            exp_value += term
            if abs(term) < 1e-15:
                break

        d = 0.3989423 * exp_value
        prob = d * t * (0.3193815 + t * (
            -0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))))

        if z > 0:
            return 1 - prob
        else:
            return prob
