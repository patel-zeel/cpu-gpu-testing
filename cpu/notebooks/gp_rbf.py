import pandas as pd
import numpy as np
from GPy.models import GPRegression
from GPy.kern import RBF
import os
import sys
np.random.seed(0)
f_name = sys.argv[1]
recompute = int(sys.argv[2])
compute=True
base_data_path = '../data/common/'
base_result_path = '../raw_results/GP_RBF/'

if os.path.exists(f_name):
    compute = True if recompute else False
    
if compute:
    sub_data = pd.read_pickle(base_data_path+f_name)
    kern = RBF(input_dim=2, active_dims=[0,1], ARD=True)
    model = GPRegression(X = sub_data['trn_val_X'], 
                         Y = sub_data['trn_val_Y'],
                         kernel = kern)
    Yscaler = sub_data['Yscaler']
#     model.kern.lengthscale.constrain_bounded(0.1,10)
    model.optimize_restarts(5, robust=True, verbose=0)
    pred_y = Yscaler.inverse_transform(model.predict(sub_data['test_X'])[0])
    rmse = np.sqrt(np.mean(np.square(sub_data['test_Y'].squeeze() - pred_y.squeeze())))
#     print(rmse, pred_y, model.kern.lengthscale)
    pd.to_pickle({'rmse':rmse, 'test_y':sub_data['test_Y'].squeeze(),
                  'pred_y':pred_y, 'lengthscales':model.kern.lengthscale.tolist(),'model':model}, 
                 base_result_path+f_name)
