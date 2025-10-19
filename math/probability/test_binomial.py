#!/usr/bin/env python3

import numpy as np
from binomial import Binomial

# Test Task 10: Initialize Binomial
print("=== Testing Task 10: Initialize Binomial ===")
np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('n:', b1.n, "p:", b1.p)

b2 = Binomial(n=50, p=0.6)
print('n:', b2.n, "p:", b2.p)
print()

# Test Task 11: Binomial PMF
print("=== Testing Task 11: Binomial PMF ===")
print('P(30):', b1.pmf(30))
print('P(30):', b2.pmf(30))
print()

# Test Task 12: Binomial CDF
print("=== Testing Task 12: Binomial CDF ===")
print('F(30):', b1.cdf(30))
print('F(30):', b2.cdf(30))
