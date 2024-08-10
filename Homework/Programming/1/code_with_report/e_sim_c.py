from mpi4py import MPI
import random
import time
from decimal import Decimal, getcontext

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_processes = comm.Get_size()

total_iterations = 4_000_000
local_counter = 0

getcontext().prec = 40

if my_rank == 0:
    start = time.time()
for _ in range(int(total_iterations // num_processes)):
    running_total = 0
    while running_total <= 1:
        running_total += random.uniform(0, 1)
        local_counter += 1
total_count = comm.reduce(local_counter, root=0)

if my_rank == 0:
    print(f"Euler: {Decimal(total_count / total_iterations)}")
    print(f"time: {time.time() - start} (s)")
