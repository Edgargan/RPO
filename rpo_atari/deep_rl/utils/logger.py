#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
from .misc import *

def get_logger(config, tag='default', log_level=0):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if tag is not None:
        if tag == 'PPO':
            fh = logging.FileHandler(config.base_path_log_txt + '/%s-ratio_clip-%s-seed-%s-%s.txt' %
                                     (tag, str(config.ppo_ratio_clip), str(config.seed), get_time_str()))
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
            return Logger(logger, config.base_path_tf_log + '/%s-ratio_clip-%s-seed-%s-%s' %
                          (tag, str(config.ppo_ratio_clip), str(config.seed), get_time_str()), log_level)
        elif tag == 'PPO_multi_ratio':
            fh = logging.FileHandler(config.base_path_log_txt + '/%s_seed_%s_%s.txt' %
                                     (config.tag_1, config.seed, get_time_str()))
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
            return Logger(logger, config.base_path_tf_log + '/%s_seed_%s_%s' %
                          (config.tag_1, config.seed, get_time_str()), log_level)
        else:
            fh = logging.FileHandler(config.base_path_log_txt + '/%s_seed_%s_%s.txt' %
                                     (config.tag, config.seed, get_time_str()))
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
            return Logger(logger, config.base_path_tf_log + '/%s_seed_%s_%s' %
                          (config.tag, config.seed, get_time_str()), log_level)
            # fh = logging.FileHandler(config.base_path + 'log_txt/%s/%s/%s-seed-%s-%s.txt' %
            #                          (config.env, tag, tag, str(config.seed), get_time_str()))
            # fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
            # fh.setLevel(logging.INFO)
            # logger.addHandler(fh)
            # return Logger(logger, config.base_path + 'tf_log/%s/%s/%s-seed-%s-%s' %
            #               (config.env, tag, tag, str(config.seed), get_time_str()), log_level)
    else:
        return Logger(logger, config.base_path + 'tf_log/%s/%s/%s-%s' % (config.env, tag, tag, get_time_str()), log_level)


# def get_logger(config, tag='default', log_level=0):
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     if tag is not None:
#         if tag == 'PPO':
#             fh = logging.FileHandler(config.base_path_log_txt + '/%s-ratio_clip-%s-seed-%s-%s.txt' %
#                                      (tag, str(config.ppo_ratio_clip), str(config.seed), get_time_str()))
#             fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
#             fh.setLevel(logging.INFO)
#             logger.addHandler(fh)
#             return Logger(logger, config.base_path_tf_log + '/%s-ratio_clip-%s-seed-%s-%s' %
#                           (tag, str(config.ppo_ratio_clip), str(config.seed), get_time_str()), log_level)
#         elif config.tag_0 is not None:
#             fh = logging.FileHandler(config.base_path_log_txt + '/%s_%s_%s-%s.txt' %
#                                  (config.tag_0, config.env, config.tag_1, get_time_str()))
#             fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
#             fh.setLevel(logging.INFO)
#             logger.addHandler(fh)
#             return Logger(logger, config.base_path_tf_log + '/%s_%s_%s-%s' %
#                           (config.tag_0, config.env, config.tag_1, get_time_str()), log_level)
#         else:
#             fh = logging.FileHandler(config.base_path + 'log_txt/%s/%s/%s-seed-%s-%s.txt' %
#                                      (config.env, tag, tag, str(config.seed), get_time_str()))
#             fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
#             fh.setLevel(logging.INFO)
#             logger.addHandler(fh)
#             return Logger(logger, config.base_path + 'tf_log/%s/%s/%s-seed-%s-%s' %
#                           (config.env, tag, tag, str(config.seed), get_time_str()), log_level)
#     else:
#         return Logger(logger, config.base_path + 'tf_log/%s/%s/%s-%s' % (config.env, tag, tag, get_time_str()), log_level)



class Logger(object):
    def __init__(self, vanilla_logger, log_dir, log_level=0):
        self.log_level = log_level
        self.writer = None
        if vanilla_logger is not None:
            self.info = vanilla_logger.info
            self.debug = vanilla_logger.debug
            self.warning = vanilla_logger.warning
        self.all_steps = {}
        self.log_dir = log_dir

    def lazy_init_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def add_scalar(self, tag, value, step=None, log_level=0):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None, log_level=0):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)
