import subprocess

percent_instance = [0.1, 0.2]

for p in percent_instance:
    for i in range(5):  # Proposed
        cmd = f'python3 main.py --dataset fashionmnist --noise_rate 0.5 --lam 0.0001 --reg_type min --loss_func sl --alpha 0.9 --beta 0.4 --percent_instance_noise {p} --sess {i}'
        subprocess.run(cmd.split())

    for i in range(5):  # Proposed Max
        cmd = f'python3 main.py --dataset fashionmnist --noise_rate 0.5 --lam 0.0001 --reg_type max --loss_func sl --alpha 0.9 --beta 0.4 --percent_instance_noise {p} --sess {i}'
        subprocess.run(cmd.split())

    for i in range(5):  # Proposed no Regularization
        cmd = f'python3 main.py --dataset fashionmnist --noise_rate 0.5 --vol_min False --lam 0.0001 --reg_type min --loss_func sl --alpha 0.9 --beta 0.4 --percent_instance_noise {p} --sess {i}'
        subprocess.run(cmd.split())

    for i in range(5):  # VolMinNet
        cmd = f'python3 main.py --dataset fashionmnist --noise_rate 0.5 --lam 0.0001 --reg_type min --loss_func ce --percent_instance_noise {p} --sess {i}'
        subprocess.run(cmd.split())

    for i in range(5):  # CE
        cmd = f'python3 main.py --dataset fashionmnist --noise_rate 0.5 --vol_min False --lam 0.0001 --reg_type min --loss_func ce --percent_instance_noise {p} --sess {i}'
        subprocess.run(cmd.split())

    for i in range(5):  # GCE
        cmd = f'python3 main.py --dataset fashionmnist --noise_rate 0.5 --vol_min False --lam 0.0001 --reg_type min --loss_func gce --q 0.5 --k 0.4 --percent_instance_noise {p} --sess {i}'
        subprocess.run(cmd.split())



