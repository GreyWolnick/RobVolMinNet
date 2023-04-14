import subprocess

datasets = ['mnist', 'cifar10', 'fashionmnist']
indep_noise_rate = 0.2
dep_noise_rate = 0.4
k = 0.4
q = 0.5
lam = 0.00001

print("THIS IS A TEST")

for i in range(10):
    print(i)

# for dataset in datasets:
#     cmd = f'python3 gce.py --dataset {dataset} --indep_noise_rate {indep_noise_rate} --dep_noise_rate {dep_noise_rate} --q {q} --k {k} --lam {lam}'
#     subprocess.run(cmd.split())
#     cmd = f'python3 gce.py --loss_func ce --dataset {dataset} --indep_noise_rate {indep_noise_rate} --dep_noise_rate {dep_noise_rate} --q {q} --k {k} --lam {lam}'
#     subprocess.run(cmd.split())


