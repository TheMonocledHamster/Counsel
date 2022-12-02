import numpy as np
from source.env import CustomEnv
from source.synthetic import set_slo, set_base

slo = int(np.exp(np.random.randint(240,840)/100))
freq = int(1e6 / np.random.randint(int(slo*0.8), int(slo*1.2)))
knob = 0.1 # For over, under and near provisioning
print(f"SLO: {slo}, Freq: {freq}, Knob: {knob}")

env = CustomEnv("_", 2048, [30,60], slo, 0.2, 'synthetic')
set_slo(slo, freq, knob)
set_base(env.components)
for i in range(15):
    print(env.step(0)[2])
