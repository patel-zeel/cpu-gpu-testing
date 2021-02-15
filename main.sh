### CPU Test
echo "Running CPU test"
cd ~/Ampere-CPU-GPU-Test/cpu/notebooks
jupyter nbconvert GP_RBF.ipynb --to python
ipython GP_RBF.py

### GPU Test
echo "Running GPU test"
cd ~/Ampere-CPU-GPU-Test/gpu
conda install tensorflow-gpu -y
python tensorflow_with_gpu.py
