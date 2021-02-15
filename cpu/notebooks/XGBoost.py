#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython import get_ipython
get_ipython().system('pip install -qq GPy ')
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

# In[2]:


get_ipython().run_cell_magic('writefile', 'xgboost.py', "import pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import GradientBoostingRegressor\nimport os\nimport sys\nnp.random.seed(0)\nf_name = sys.argv[1]#'out_i_0_t_i_0'#\nrecompute = int(sys.argv[2])#1#\ncompute=True\nbase_data_path = '../data/common/'\nbase_result_path = '../raw_results/XGBoost/'\n\nif os.path.exists(f_name):\n    compute = True if recompute else False\n    \nif compute:\n    sub_data = pd.read_pickle(base_data_path+f_name)\n    model = GradientBoostingRegressor()\n    model.fit(sub_data['trn_val_X'], sub_data['trn_val_Y'].ravel())\n    Yscaler = sub_data['Yscaler']\n    pred_y = Yscaler.inverse_transform(model.predict(sub_data['test_X']))\n    rmse = np.sqrt(np.mean(np.square(sub_data['test_Y'].squeeze() - pred_y.squeeze())))\n#     print(rmse, pred_y)\n    pd.to_pickle({'rmse':rmse, 'test_y':sub_data['test_Y'].squeeze(),\n                  'pred_y':pred_y, 'model':model}, \n                 base_result_path+f_name)")


# ### Compute

# In[3]:


i = 0
jobs = []
main_command = 'python xgboost.py'

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
pd.to_pickle(end, main_command+'_seconds_time.pickle')
print('Job finished in',end,'seconds')

