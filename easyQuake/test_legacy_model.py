#!/usr/bin/env python3

import numpy as np
import sys
import os
sys.path.insert(0, '/home/jwalter/easyQuake/easyQuake')

# Import the GPD module
from gpd_predict import gpd_predict

print("Testing legacy model with the same data that produced 489 picks...")

# Run GPD with the legacy model on the same data
gpd_predict.main()

print("Finished running GPD test.")
