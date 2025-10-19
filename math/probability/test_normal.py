#!/usr/bin/env python3

import numpy as np
from normal import Normal

# Test Task 6: Initialize Normal
print("=== Testing Task 6: Initialize Normal ===")
np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('Mean:', n1.mean, ', Stddev:', n1.stddev)

n2 = Normal(mean=70, stddev=10)
print('Mean:', n2.mean, ', Stddev:', n2.stddev)
print()

# Test Task 7: Normalize Normal (z_score and x_value)
print("=== Testing Task 7: Normalize Normal ===")
print('Z(90):', n1.z_score(90))
print('X(2):', n1.x_value(2))

print('Z(90):', n2.z_score(90))
print('X(2):', n2.x_value(2))
print()

# Test Task 8: Normal PDF
print("=== Testing Task 8: Normal PDF ===")
print('PSI(90):', n1.pdf(90))
print('PSI(90):', n2.pdf(90))
print()

# Test Task 9: Normal CDF
print("=== Testing Task 9: Normal CDF ===")
print('PHI(90):', n1.cdf(90))
print('PHI(90):', n2.cdf(90))
