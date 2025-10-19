#!/usr/bin/env python3

import numpy as np
from poisson import Poisson

# Test Task 0: Initialize Poisson
print("=== Testing Task 0: Initialize Poisson ===")
np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('Lambtha:', p1.lambtha)

p2 = Poisson(lambtha=5)
print('Lambtha:', p2.lambtha)
print()

# Test Task 1: Poisson PMF
print("=== Testing Task 1: Poisson PMF ===")
print('P(9):', p1.pmf(9))
print('P(9):', p2.pmf(9))
print()

# Test Task 2: Poisson CDF
print("=== Testing Task 2: Poisson CDF ===")
print('F(9):', p1.cdf(9))
print('F(9):', p2.cdf(9))
