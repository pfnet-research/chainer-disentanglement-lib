# example 1
export DATASET_NAME=dsprites_full
export EVALUATION_NAME=beta_vae/${DATASET_NAME}/gpu
python3 train.py --vae BetaVAE --device 0 --training_steps 100 

# example 2
export DATASET_NAME=dsprites_full
export EVALUATION_NAME=factor_vae/${DATASET_NAME}/cpu
python3 train.py --vae FactorVAE --device -1 --training_steps 100000 

# example 3
export DATASET_NAME=mpi3d_toy
export EVALUATION_NAME=dip_vae_1/${DATASET_NAME}/gpu
python3 train.py --vae DIPVAE-1 --batch_size 128

# example 4
export DATASET_NAME=mpi3d_toy
export EVALUATION_NAME=dip_vae_2/${DATASET_NAME}/cpu/n_z_20/seed_2019
python3 train.py --vae DIPVAE-2 --device -1 --n_z 20 --seed 2019

# example 5
export DATASET_NAME=dsprites_full
export EVALUATION_NAME=joint_vae/${DATASET_NAME}/gpu/seed_10
python3 train.py --vae JointVAE --seed 10