import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import sys
np.random.seed(0)
f_name = sys.argv[1]
recompute = int(sys.argv[2])
compute=True
base_data_path = '../data/common/'
base_result_path = '../raw_results/RF/'

if os.path.exists(f_name):
    compute = True if recompute else False
    
if compute:
    sub_data = pd.read_pickle(base_data_path+f_name)
    model = RandomForestRegressor(random_state=0)
    Yscaler = sub_data['Yscaler']
    pred_y = Yscaler.inverse_transform(model.predict(sub_data['test_X']))
    rmse = np.sqrt(np.mean(np.square(sub_data['test_Y'].squeeze() - pred_y.squeeze())))
    pd.to_pickle({'rmse':rmse, 'test_y':sub_data['test_Y'].squeeze(),
                  'pred_y':pred_y, 'model':model}, 
                 base_result_path+f_name)
    pass
