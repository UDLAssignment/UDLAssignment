import os
import sys

exp = int(sys.argv[1])

# KL coef sweep
if exp == 1:
    for kl_coef in [8,9,10,11,12]: 
        cmd = f"python main.py epochs=100 posterior_type=flow_meanfield name=flow_base_reg_{kl_coef} kl_loss_coef={10**kl_coef} num_kl_estimate_samples=10000"
        os.system(cmd)

# Accuracy imbalance
if exp == 2:
    for j in [0,1,2]:
        for i in range(10):
            cmd = f"python main.py epochs=100 posterior_type=meanfield name=acc_imbalance_{j}_{i} transform_swaps={j}:{i}"
            os.system(cmd)

# permuted mnist
## meanfield
if exp == 10:
    for iter in range(3):
        cmd = f"python main.py epochs=100 posterior_type=meanfield name=meanfield_iter{iter}"
        os.system(cmd)

## flow
if exp == 3:
    for iter in range(3):
        cmd = f"python main.py epochs=100 posterior_type=flow name=flow_iter{iter}"
        os.system(cmd)

## flow meanfield
if exp == 4:
    for iter in [2]:
        cmd = f"python main.py epochs=100 posterior_type=flow_meanfield name=flowmeanfield_iter{iter} kl_loss_coef=1000000000 num_kl_estimate_samples=10000 num_flow_layers=2"
        os.system(cmd)

## rank1
if exp == 5:
    for iter in range(3):
        cmd = f"python main.py epochs=100 posterior_type=meanfield_lowrankvar name=pm_rank1_{iter} postvar_weight_rank=1"
        os.system(cmd)

# permuted fmnist
## flow
if exp == 6:
    for iter in range(3):
        cmd = f"python main.py epochs=100 posterior_type=flow name=fmnist_flow_iter{iter} data=fashion"
        os.system(cmd)

## meanfield
if exp == 9:
    for iter in range(3):
        cmd = f"python main.py epochs=100 posterior_type=meanfield name=fmnist_meanfield_iter{iter} data=fashion"
        os.system(cmd)

## flow meanfield
if exp == 7:
    for iter in range(3):
        cmd = f"python main.py epochs=100 posterior_type=flow_meanfield name=fmnist_flowmeanfield_iter{iter} data=fashion kl_loss_coef=1000000000 num_kl_estimate_samples=10000 num_flow_layers=1"
        os.system(cmd)

## rank1
if exp == 8:
    for iter in range(3):
        cmd = f"python main.py epochs=100 posterior_type=meanfield_lowrankvar name=fmnist_pm_rank1_{iter} postvar_weight_rank=1 data=fashion"
        os.system(cmd)
