#!/usr/bin/env python3


"""Class that represents a binomial distribution"""


class Binomial:
    """Class that represents a binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize Binomial distribution
        
        Args:
            data: list of data to estimate the distribution
            n: number of Bernoulli trials
            p: probability of a "success"
        """
        if data is None:
            # Use provided n and p
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            # Calculate n and p from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            
            # Calculate p first (proportion of successes)
            # For binomial data, we need to estimate p from the data
            # Assuming data contains 0s and 1s (or success/failure counts)
            # p = mean of data
            p_estimate = sum(data) / len(data)
            
            # Calculate n (total number of trials)
            # For binomial, n is typically the maximum value in the data
            # or we can estimate it from the variance
            max_val = max(data)
            n_estimate = max_val
            
            # Round n to nearest integer
            self.n = round(n_estimate)
            
            # Recalculate p based on the rounded n
            self.p = p_estimate
    
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
        
        # If k is out of range, return 0
        if k < 0 or k > self.n:
            return 0
        
        # Binomial PMF formula: P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
        # where C(n,k) = n! / (k! × (n-k)!)
        
        # Calculate binomial coefficient C(n,k)
        if k == 0 or k == self.n:
            binomial_coeff = 1
        else:
            # Use the more efficient formula to avoid large factorials
            binomial_coeff = 1
            for i in range(min(k, self.n - k)):
                binomial_coeff = binomial_coeff * (self.n - i) // (i + 1)
        
        # Calculate p^k
        p_power_k = self.p ** k
        
        # Calculate (1-p)^(n-k)
        one_minus_p_power_n_minus_k = (1 - self.p) ** (self.n - k)
        
        # Calculate the PMF
        pmf_value = binomial_coeff * p_power_k * one_minus_p_power_n_minus_k
        
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
        
        # If k is out of range, return appropriate value
        if k < 0:
            return 0
        if k >= self.n:
            return 1
        
        # CDF is the sum of PMF values from 0 to k
        # P(X ≤ k) = Σ(i=0 to k) P(X = i)
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        
        return cdf_value
