import subprocess

percent_instance = [0.1, 0.2]

for i in range(1):  # Proposed
        cmd = f'python3 main.py --dataset cifar100 --noise_rate 0.5 --lam 0.0001 --reg_type min --loss_func sl --alpha 0.9 --beta 0.4 --percent_instance_noise 0.1 --sess {i}'
        subprocess.run(cmd.split())
        print(f"Finished iteration {i+1} of proposed")

for i in range(1):  # VolMinNet
    cmd = f'python3 main.py --dataset cifar100 --noise_rate 0.5 --lam 0.00001 --reg_type min --loss_func ce --percent_instance_noise 0.1 --sess {i}'
    subprocess.run(cmd.split())
    print(f"Finished iteration {i + 1} of volminnet")

#
# for p in percent_instance:
#     for i in range(5):  # Proposed
#         cmd = f'python3 main.py --dataset mnist --noise_rate 0.5 --lam 0.0001 --reg_type min --loss_func sl --alpha 0.9 --beta 0.4 --percent_instance_noise {p} --sess {i}'
#         subprocess.run(cmd.split())
#         print(f"Finished iteration {i+1} of proposed")
#
#     for i in range(5):  # Proposed Max
#         cmd = f'python3 main.py --dataset mnist --noise_rate 0.5 --lam 0.0001 --reg_type max --loss_func sl --alpha 0.9 --beta 0.4 --percent_instance_noise {p} --sess {i}'
#         subprocess.run(cmd.split())
#         print(f"Finished iteration {i + 1} of proposed max")
#
#     for i in range(5):  # Proposed no Regularization
#         cmd = f'python3 main.py --dataset mnist --noise_rate 0.5 --vol_min False --lam 0.0001 --reg_type min --loss_func sl --alpha 0.9 --beta 0.4 --percent_instance_noise {p} --sess {i}'
#         subprocess.run(cmd.split())
#         print(f"Finished iteration {i + 1} of proposed no reg")
#
#     for i in range(5):  # VolMinNet
#         cmd = f'python3 main.py --dataset mnist --noise_rate 0.5 --lam 0.00001 --reg_type min --loss_func ce --percent_instance_noise {p} --sess {i}'
#         subprocess.run(cmd.split())
#         print(f"Finished iteration {i + 1} of volminnet")
#
#     for i in range(5):  # CE
#         cmd = f'python3 main.py --dataset mnist --noise_rate 0.5 --vol_min False --lam 0.0001 --reg_type min --loss_func ce --percent_instance_noise {p} --sess {i}'
#         subprocess.run(cmd.split())
#         print(f"Finished iteration {i + 1} of ce")
#
#     for i in range(5):  # GCE
#         cmd = f'python3 main.py --dataset mnist --noise_rate 0.5 --vol_min False --lam 0.0001 --reg_type min --loss_func gce --q 0.7 --k 0.5 --percent_instance_noise {p} --sess {i}'
#         subprocess.run(cmd.split())
#         print(f"Finished iteration {i + 1} of gce")



