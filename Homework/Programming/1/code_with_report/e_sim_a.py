import random 
import time
from decimal import Decimal, getcontext

NUM_ITERATIONS = 4_000_000

def monte_carlo_e_estimate():
    counter = 0
    getcontext().prec = 40
    start_time = time.time()
    for _ in range(NUM_ITERATIONS):
        total = 0
        while total <= 1:
            total += random.uniform(0, 1)
            counter += 1
    elapsed_time = time.time() - start_time
    e_estimate = Decimal(counter / NUM_ITERATIONS)
    print(f"Euler's Number Estimate: {e_estimate}")
    print(f"Elapsed Time: {elapsed_time} seconds")
if __name__ == '__main__':
    monte_carlo_e_estimate()
