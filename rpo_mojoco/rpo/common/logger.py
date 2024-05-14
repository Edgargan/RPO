import numpy as np
from datetime import datetime
import pickle
import os
import tensorflow as tf
from tensorboardX import SummaryWriter

class Logger:
    """Class for logging data throughout training"""

    def __init__(self, params):
        self.params = params
        self.param_dict = dict()
        self.train_dict = dict()
        self.final_dict = dict()
        # self.current_dict = dict()
        self.dir = params["log_path"]
        self.writer = SummaryWriter(self.dir)
        path_model = self.dir.split('/')
        path_model[4] = path_model[4]+"_Model"
        # self.filepath = self.params["log_path"] + '_Model'
        self.filepath = '/'.join(path_model)
        if not os.path.exists(self.filepath):
            # os.mkdirs(self.filepath)
            os.makedirs(self.filepath)

    # def S_W(self):
    #     SummaryWriter = SummaryWriter(self.dir)
    #     return SummaryWriter

    def add_scalar(self, tag, value, step):
        # with self.writer.as_default():
        #     tf.summary.scalar(tag, value, step)
        self.writer.add_scalar(tag, value, step)

    # def log_train(self,kv):
    #     for k,v in kv.items():
    #         if k in self.train_dict.keys():
    #             self.train_dict[k].append(v)
    #         else:
    #             self.train_dict[k] = [v]
    
    def log_params(self,kv):
        for k,v in kv.items():
            self.param_dict[k] = v

    def log_final(self,kv):
        for k,v in kv.items():
            self.final_dict[k] = v

    def dump(self):
        train_out = dict()
        for k,v in self.train_dict.items():
            train_out[k] = np.array(v)
        
        out = {
            'param': self.param_dict,
            # 'train': train_out,
            'final': self.final_dict
        }

        return out

    def save(self,log_path,log_name):
        out = self.dump()

        timestamp = datetime.today().strftime('%m%d%y_%H%M%S')

        os.makedirs(log_path,exist_ok=True)
        filename = 'log_%s_%s'%(log_name,timestamp)
        filefull = os.path.join(log_path,filename)
        with open(filefull,'wb') as f:
            pickle.dump(out,f)

    def log_current(self,kv):
        current_dict = dict()
        for k,v in kv.items():
            current_dict[k] = v
        return current_dict

    def save_current(self,kv, log_path,log_name):

        # os.makedirs(log_path, exist_ok=True)
        filename = 'log_%s' % (log_name)
        filefull = os.path.join(self.filepath, filename)
        with open(filefull, 'wb') as f:
            pickle.dump(kv, f)













