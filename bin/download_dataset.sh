#!/bin/bash
mkdir -p ${DISENTANGLEMENT_LIB_DATA}
cd ${DISENTANGLEMENT_LIB_DATA}

echo "Downloading dSprites dataset."
if [[ ! -d "dsprites" ]]; then
  mkdir dsprites
  wget -O dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
fi
echo "Downloading dSprites completed!"

echo "Downloading mpi3d_toy dataset."
if [[ ! -d "mpi3d_toy" ]]; then
  mkdir mpi3d_toy
  wget -O mpi3d_toy/mpi3d_toy.npz https://storage.googleapis.com/disentanglement_dataset/data_npz/sim_toy_64x_ordered_without_heldout_factors.npz
fi
echo "Downloading mpi3d_toy completed!"