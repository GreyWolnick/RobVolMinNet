import subprocess

# datasets = ['mnist', 'cifar10', 'fashionmnist']
qs = [0.1, 0.5, 0.7, 0.9]
ks = [0.1, 0.5, 0.9]
# lams = [0.1, 0.01]

# cmd = f'python3 gce.py --reg_type max --dataset cifar10'
# subprocess.run(cmd.split())
# cmd = f'python3 gce.py --reg_type max --dataset cifar10 --loss_func ce'
# subprocess.run(cmd.split())

# for i in range(3):  # VGCE fashion 0.5
#     cmd = f'python3 main.py --dataset fashionmnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.5 --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # VGCE mnist 0.2
#     cmd = f'python3 main.py --dataset mnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # VGCE fashion 0.2
#     cmd = f'python3 main.py --dataset fashionmnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # VolMinNet fashion 0.2
#     cmd = f'python3 main.py --dataset fashionmnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --loss_func ce --sess {i}'
#     subprocess.run(cmd.split())

# for i in range(5):  # GCE mnist 0.2
#     cmd = f'python3 main.py --dataset mnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --vol_min False --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # GCE mnist 0.5
#     cmd = f'python3 main.py --dataset mnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.5 --vol_min False --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # GCE fashion 0.2
#     cmd = f'python3 main.py --dataset fashionmnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --vol_min False --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # GCE fashion 0.5
#     cmd = f'python3 main.py --dataset fashionmnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.5 --vol_min False --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # GCE cifar10 0.2
#     cmd = f'python3 main.py --dataset cifar10 --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --vol_min False --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # GCE cifar10 0.5
#     cmd = f'python3 main.py --dataset cifar10 --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.5 --vol_min False --sess {i}'
#     subprocess.run(cmd.split())

# for i in range(5):  # CE mnist 0.2
#     cmd = f'python3 main.py --dataset mnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --vol_min False --loss_func ce --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # CE mnist 0.5
#     cmd = f'python3 main.py --dataset mnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.5 --vol_min False --loss_func ce --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # CE fashion 0.2
#     cmd = f'python3 main.py --dataset fashionmnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --vol_min False --loss_func ce --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # CE fashion 0.5
#     cmd = f'python3 main.py --dataset fashionmnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.5 --vol_min False --loss_func ce --sess {i}'
#     subprocess.run(cmd.split())
#
# for i in range(5):  # CE cifar10 0.2
#     cmd = f'python3 main.py --dataset cifar10 --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --vol_min False --loss_func ce --sess {i}'
#     subprocess.run(cmd.split())

for i in range(5):  # CE mnist 0.5
    cmd = f'python3 main.py --dataset mnist --q 0.2 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --loss_func ce --sess {i}'
    subprocess.run(cmd.split())

for i in range(5):  # CE fashion 0.5
    cmd = f'python3 main.py --dataset fashionmnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --loss_func ce --sess {i}'
    subprocess.run(cmd.split())

for i in range(5):  # CE cifar10 0.5
    cmd = f'python3 main.py --dataset cifar10 --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --loss_func ce --sess {i}'
    subprocess.run(cmd.split())

for i in range(5):  # VGCE mnist 0.5
    cmd = f'python3 main.py --dataset mnist --q 0.2 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --sess {i}'
    subprocess.run(cmd.split())

for i in range(5):  # VGCE fashion 0.5
    cmd = f'python3 main.py --dataset fashionmnist --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --sess {i}'
    subprocess.run(cmd.split())

for i in range(5):  # VGCE cifar10 0.5
    cmd = f'python3 main.py --dataset cifar10 --q 0.5 --k 0.4 --lam 0.0001 --reg_type min --noise_rate 0.2 --sess {i}'
    subprocess.run(cmd.split())



# for k in ks:
#     cmd = f'python3 main.py --dataset cifar10 --q 0.5 --k {k} --lam 0.0001'
#     subprocess.run(cmd.split())
#
# for lam in lams:
#     cmd = f'python3 gce.py --reg_type max --dataset cifar10 --q 0.5 --k 0.4 --lam {lam}'
#     subprocess.run(cmd.split())


