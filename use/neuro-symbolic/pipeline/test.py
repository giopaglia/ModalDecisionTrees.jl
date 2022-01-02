from pipeline import *
import numpy as np

load_model(dataset="racketsports",
    seed=1,
    attribute=2,
    code=1,
    limit=100)

o = str(np.random.randn(1,4).tolist())

validation(o)