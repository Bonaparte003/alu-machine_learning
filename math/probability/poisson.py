#!/usr/bin/env python3

import math


"""Class that represents a poisson distribution"""


class Poisson:
    """Class that represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
    
    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes"
        
        Args:
            k: number of "successes"
            
        Returns:
            PMF value for k
        """
        # Convert k to integer if it's not already
        k = int(k)
        
        # If k is out of range (negative), return 0
        if k < 0:
            return 0
        
        # Poisson PMF formula: P(X = k) = (λ^k × e^(-λ)) / k!
        # λ^k
        lambda_power_k = self.lambtha ** k
        
        # e^(-λ)
        exp_neg_lambda = math.exp(-self.lambtha)
        
        # k!
        k_factorial = math.factorial(k)
        
        # Calculate the PMF
        pmf_value = (lambda_power_k * exp_neg_lambda) / k_factorial
        
        return pmf_value
    
    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of "successes"
        
        Args:
            k: number of "successes"
            
        Returns:
            CDF value for k
        """
        # Convert k to integer if it's not already
        k = int(k)
        
        # If k is out of range (negative), return 0
        if k < 0:
            return 0
        
        # CDF is the sum of PMF values from 0 to k
        # P(X ≤ k) = Σ(i=0 to k) P(X = i)
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        
        return cdf_value
