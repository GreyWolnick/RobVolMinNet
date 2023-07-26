import numpy as np

def calculate_accuracy(dataset, loss_func, vol_min, reg_type, lam, instance_noise_rate):
    alpha = "0.900000"
    beta = "0.400000"
    # q = "0.700000"
    # k = "0.500000"
    q = "0.700000"
    k = "0.500000"
    noise_rate = "0.300000"

    best_accs = []
    for i in range(5):
        if loss_func == "sl":
            file_path = f"saves/{dataset}/{loss_func}/vol_min={vol_min}/{reg_type}/alpha={alpha}/beta={beta}/noise_rate_{noise_rate}/instance_noise_rate_{instance_noise_rate}/lam={lam}06_1/{i}/log.txt"
        elif loss_func == "gce":
            file_path = f"saves/{dataset}/{loss_func}/vol_min={vol_min}/{reg_type}/q={q}/k={k}/noise_rate_{noise_rate}/instance_noise_rate_{instance_noise_rate}/lam={lam}006_1/{i}/log.txt"
        else:
            file_path = f"saves/{dataset}/{loss_func}/vol_min={vol_min}/{reg_type}/noise_rate_{noise_rate}/instance_noise_rate_{instance_noise_rate}/lam={lam}06_1/{i}/log.txt"

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("Best Model Test Loss"):
                    desired_line = line.strip()
                    best_accs.append(float(desired_line[-8:]))
                    break

    print(dataset, loss_func, vol_min, reg_type, instance_noise_rate)
    print(f"{np.mean(best_accs) * 100:.2f}±{np.std(best_accs) * 100:.3f}")


configurations = [
    {
        'dataset': "cifar10",
        'loss_func': "sl",
        'vol_min': "True",
        'reg_type': "min",
        'lam': "0.00001"
    },
#     {
#         'dataset': "cifar10",
#         'loss_func': "sl",
#         'vol_min': "True",
#         'reg_type': "max",
#         'lam': "0.0001"
#     },
#     {
#         'dataset': "cifar10",
#         'loss_func': "sl",
#         'vol_min': "False",
#         'reg_type': "min",
#         'lam': "0.0000"
#     },
# {
#         'dataset': "cifar10",
#         'loss_func': "ce",
#         'vol_min': "True",
#         'reg_type': "min",
#         'lam': "0.00001"
#     },
#     {
#         'dataset': "cifar10",
#         'loss_func': "ce",
#         'vol_min': "False",
#         'reg_type': "min",
#         'lam': "0.00000"
#     },
    {
        'dataset': "cifar10",
        'loss_func': "gce",
        'vol_min': "False",
        'reg_type': "min",
        'lam': "0.0000"
    },
]

percent_instance = ["0.100000", "0.200000"]

for config in configurations:
    for p in percent_instance:
        calculate_accuracy(config['dataset'], config['loss_func'], config['vol_min'], config['reg_type'], config['lam'], p)
