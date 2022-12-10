import numpy as np
from source.env import CloudEnv
from source.synthetic import set_slo, set_base

slo = int(np.exp(np.random.randint(240,840)/100))
freq = int(1e6 / np.random.randint(int(slo*0.8), int(slo*1.2)))
knob = 0.05 # For over, under and near provisioning
print(f"SLO: {slo}, Freq: {freq}, Knob: {knob}")

env = CloudEnv("_", 2048, [120,480], slo, 0.2, 'synthetic', 25, 5)
set_slo(slo, freq, knob)
set_base(env.components)
for i in range(15):
    print(env.step(0)[2])
