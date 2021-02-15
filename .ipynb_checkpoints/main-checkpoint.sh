### CPU Test
echo "Running CPU test"
cd ~/cpu-gpu-testing/cpu/notebooks
python cpu_test.py
jupyter nbconvert GP_RBF.ipynb --to python
ipython GP_RBF.py

### GPU Test
#echo "Running GPU test"
#cd ~/cpu-gpu-testing/gpu
#conda install tensorflow-gpu -y
#python tensorflow_with_gpu.py
