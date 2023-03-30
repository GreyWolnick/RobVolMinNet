import subprocess

datasets = ['mnist', 'cifar10', 'fashionmnist']
indep_noise_rate = 0.2
dep_noise_rate = 0.4
k = 0.4
q = 0.5
lam = 0.00001

for dataset in datasets:
    cmd = f'python3 gce.py --dataset {dataset} --indep_noise_rate {indep_noise_rate} --dep_noise_rate {dep_noise_rate} --q {q} --k {k} --lam {lam}'
    subprocess.run(cmd.split())

# q = 0.7
#
# for k in range(1, 5):
#     k = k / 10.0
#     cmd = f'python3 gce.py --dataset {dataset} --indep_noise_rate {indep_noise_rate} --dep_noise_rate {dep_noise_rate} --q {q} --k {k}'
#     subprocess.run(cmd.split())
#
# k = 0.2
#
# for lam in [0.0001, 0.001, 0.00001]:
#     cmd = f'python3 gce.py --dataset {dataset} --indep_noise_rate {indep_noise_rate} --dep_noise_rate {dep_noise_rate} --q {q} --k {k} --lam {lam}'
#     subprocess.run(cmd.split())

