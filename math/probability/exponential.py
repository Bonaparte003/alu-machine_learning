#!/usr/bin/env python3

import math


"""Class that represents an exponential distribution"""


class Exponential:
    """Class that represents an exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Exponential distribution
        
        Args:
            data: list of data to estimate the distribution
            lambtha: rate parameter (expected number of occurrences)
        """
        if data is None:
            # Use provided lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Calculate lambtha from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # For exponential distribution, λ = 1 / mean
            mean = sum(data) / len(data)
            self.lambtha = 1.0 / mean
    
    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        
        Args:
            x: time period
            
        Returns:
            PDF value for x
        """
        # If x is out of range (negative), return 0
        if x < 0:
            return 0
        
        # Exponential PDF formula: f(x) = λ × e^(-λx)
        pdf_value = self.lambtha * math.exp(-self.lambtha * x)
        
        return pdf_value
    
    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        
        Args:
            x: time period
            
        Returns:
            CDF value for x
        """
        # If x is out of range (negative), return 0
        if x < 0:
            return 0
        
        # Exponential CDF formula: F(x) = 1 - e^(-λx)
        cdf_value = 1 - math.exp(-self.lambtha * x)
        
        return cdf_value
