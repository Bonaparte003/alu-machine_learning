#!/usr/bin/env python3

import numpy as np
from exponential import Exponential

# Test Task 3: Initialize Exponential
print("=== Testing Task 3: Initialize Exponential ===")
np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('Lambtha:', e1.lambtha)

e2 = Exponential(lambtha=2)
print('Lambtha:', e2.lambtha)
print()

# Test Task 4: Exponential PDF
print("=== Testing Task 4: Exponential PDF ===")
print('f(1):', e1.pdf(1))
print('f(1):', e2.pdf(1))
print()

# Test Task 5: Exponential CDF
print("=== Testing Task 5: Exponential CDF ===")
print('F(1):', e1.cdf(1))
print('F(1):', e2.cdf(1))
