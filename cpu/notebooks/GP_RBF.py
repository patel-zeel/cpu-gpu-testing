#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython import get_ipython
get_ipython().system('pip install -qq GPy scikit-learn')
import pandas as pd
import numpy as np
import sys
import os
from GPy.models import GPRegression
from GPy.kern import RBF
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from sklearn.metrics import mean_squared_error
from IPython.display import clear_output
from time import time
from glob import glob
import multiprocessing as mp


# ## Generate script

# In[6]:


get_ipython().run_cell_magic('writefile', 'gp_rbf.py', "import pandas as pd\nimport numpy as np\nfrom GPy.models import GPRegression\nfrom GPy.kern import RBF\nimport os\nimport sys\nnp.random.seed(0)\nf_name = sys.argv[1]\nrecompute = int(sys.argv[2])\ncompute=True\nbase_data_path = '../data/common/'\nbase_result_path = '../raw_results/GP_RBF/'\n\nif os.path.exists(f_name):\n    compute = True if recompute else False\n    \nif compute:\n    sub_data = pd.read_pickle(base_data_path+f_name)\n    kern = RBF(input_dim=2, active_dims=[0,1], ARD=True)\n    model = GPRegression(X = sub_data['trn_val_X'], \n                         Y = sub_data['trn_val_Y'],\n                         kernel = kern)\n    Yscaler = sub_data['Yscaler']\n#     model.kern.lengthscale.constrain_bounded(0.1,10)\n    model.optimize_restarts(5, robust=True, verbose=0)\n    pred_y = Yscaler.inverse_transform(model.predict(sub_data['test_X'])[0])\n    rmse = np.sqrt(np.mean(np.square(sub_data['test_Y'].squeeze() - pred_y.squeeze())))\n#     print(rmse, pred_y, model.kern.lengthscale)\n    pd.to_pickle({'rmse':rmse, 'test_y':sub_data['test_Y'].squeeze(),\n                  'pred_y':pred_y, 'lengthscales':model.kern.lengthscale.tolist(),'model':model}, \n                 base_result_path+f_name)")


# ### Compute

# In[7]:


i = 0
jobs = []
main_command = 'python gp_rbf.py'

def do_the_job(command):
    os.system(command)

for file in glob('../data/common/*'):
    clear_output(wait=True)
    print(i)
    jobs.append(main_command+' '+file.split('/')[-1]+' '+'1')
    i +=1
print('Appended')

workers = mp.Pool()
init = time()
workers.map(do_the_job, jobs)
end = time()-init
assert len(glob('../raw_results/GP_RBF/*')) == 840
os.system('python -V > python_version.txt')
with open(main_command+'_time.txt', 'w') as f:
    print('time in seconds:',end,file=f)
print('Job finished in',end,'seconds')


# In[3]:




