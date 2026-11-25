import os

# Gradient boosting (OpenMP) is ~7x faster with 4 threads than 8 on a laptop with
# 4 performance + 4 efficiency cores: with 8, threads wait at barriers for the slow cores.
# Must be set before scikit-learn is imported. Override by exporting OMP_NUM_THREADS.
os.environ.setdefault("OMP_NUM_THREADS", "4")
