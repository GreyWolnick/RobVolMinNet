import matplotlib.pyplot as plt
import numpy as np

averages = {}
stds = {}

dataset = "cifar10"
loss_func = "sl"
vol_min = "True"
reg_type = "min"
alpha = "0.900000"
beta = "0.400000"
noise_rate = "0.500000"
instance_noise_rate = "0.100000"

best_accs = []
for i in range(5):

    file_path = f"saves/{dataset}/{loss_func}/vol_min={vol_min}/{reg_type}/alpha={alpha}/beta={beta}/noise_rate_{noise_rate}/instance_noise_rate_{instance_noise_rate}/lam=0.0001006_1/{i}/log.txt"

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Best Model Test Loss"):
                desired_line = line.strip()
                best_accs.append(float(desired_line[-8:]))
                break

print(f"{np.mean(best_accs)}±{np.std(best_accs)}")


