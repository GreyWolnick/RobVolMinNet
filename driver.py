import subprocess

# datasets = ['mnist', 'cifar10', 'fashionmnist']
# qs = [0.1, 0.9]
# ks = [0.1, 0.9]
# lams = [0.1, 0.01]

cmd = f'python3 gce.py --reg_type max --dataset cifar10'
subprocess.run(cmd.split())
cmd = f'python3 gce.py --reg_type max --dataset cifar10 --loss_func ce'
subprocess.run(cmd.split())

# for q in qs:
#     cmd = f'python3 gce.py --reg_type max --dataset cifar10 --q {q} --k 0.4 --lam 0.00001'
#     subprocess.run(cmd.split())
#
# for k in ks:
#     cmd = f'python3 gce.py --reg_type max --dataset cifar10 --q 0.5 --k {k} --lam 0.00001'
#     subprocess.run(cmd.split())
#
# for lam in lams:
#     cmd = f'python3 gce.py --reg_type max --dataset cifar10 --q 0.5 --k 0.4 --lam {lam}'
#     subprocess.run(cmd.split())


