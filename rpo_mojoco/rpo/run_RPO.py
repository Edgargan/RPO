import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from datetime import datetime
import pickle
import multiprocessing as mp
import numpy as np
import tensorflow as tf

from rpo.common.cmd_utils import create_parser
from rpo.common.initializers import init_seeds, init_env
from rpo.common.initializers import init_actor, init_critic
from rpo.common.initializers import import_params

from rpo.common.runner import Runner

from rpo.algs import rpo


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def gather_inputs(args):
    """Organizes inputs to prepare for simulations"""

    input_keys = ['ac_seed','sim_seed','env_name','s_normalize','r_normalize',
        'r_shift','s_t','s_mean','s_var','r_t','r_mean','r_var',
        'B','M_max','M_targ','uniform',
        'actor_layers','actor_activations',
        'actor_gain','actor_std_mult','actor_weights',
        'vf_layers','vf_activations','vf_gain',
        'r_min','r_max','gamma','critic_weights',
        'T','lam','n','is_trunc',
        'alg_name','sim_size','no_op_batches','ppo_adapt','rpo_noadapt',
        'log_path', 'vf_iters', 'cg_iter', 'residual_tol', 'damping', 'max_kl',
        'kl_verify', 'n_tr', 'alpha', 'eps', 'eval_sample_size'
    ]

    args_dict = vars(args)
    inputs_dict = dict()
    for key in input_keys:
        inputs_dict[key] = args_dict[key]

    if args.import_path and args.import_file:
        setup_dict = import_params(
            args.import_path,args.import_file,args.import_idx)
        for key in setup_dict.keys():
            inputs_dict[key] = setup_dict[key]

    ac_keys = ['vf_lr','adv_center','adv_scale',
        'actor_lr','actor_opt_type','update_it','nminibatch',
        'eps_ppo','max_grad_norm',
        'adapt_factor','adapt_minthresh','adapt_maxthresh',
        'early_stop','scaleinitlr', 'eps_ppo_next', 'eps_ppo_next_reg']
    ac_kwargs = dict()
    for key in ac_keys:
        ac_kwargs[key] = args_dict[key]
    inputs_dict['ac_kwargs'] = ac_kwargs

    return inputs_dict

def run(ac_seed,sim_seed,
    alg_name,env_name,s_normalize,r_normalize,r_shift,
    s_t,s_mean,s_var,r_t,r_mean,r_var,
    B,M_max,M_targ,uniform,
    actor_layers,actor_activations,actor_gain,actor_std_mult,actor_weights,
    vf_layers,vf_activations,vf_gain,r_min,r_max,gamma,critic_weights,
    T,lam,n,is_trunc,sim_size,no_op_batches,ppo_adapt,rpo_noadapt,
    ac_kwargs, log_path, vf_iters, cg_iter, residual_tol,
    damping, max_kl, kl_verify, n_tr, alpha, eps, eval_sample_size
    ):
    """Runs simulation on given seed"""

    # Save input parameters as dict
    params = locals()

    env = init_env(env_name,s_normalize,r_normalize,r_shift,
        s_t,s_mean,s_var,r_t,r_mean,r_var)
    
    init_seeds(ac_seed)
    actor = init_actor(env,actor_layers,actor_activations,actor_gain,
        actor_std_mult,actor_weights)

    critic = init_critic(env,vf_layers,vf_activations,vf_gain,
        r_min,r_max,gamma,critic_weights)

    polweights = np.ones(1)
    M = 1
    eps_mult = 1.0


    params['polweights'] = polweights
    params['M'] = M
    ac_kwargs['eps_mult'] = eps_mult

    if alg_name == 'rpo' :
        b_size = B * n   # 2 * 1024
        if ppo_adapt:
            ac_kwargs['adaptlr'] = True
        else:
            ac_kwargs['adaptlr'] = False


    params['b_size'] = b_size

    path = params["log_path"].split('/')[3]

    params["log_path"] = params["log_path"] + f'/{params["env_name"]}_eps_{params["ac_kwargs"]["eps_ppo"]}_mn_{params["ac_kwargs"]["nminibatch"]}_bs_{params["b_size"]}_early_stop_{int(params["ac_kwargs"]["early_stop"])}_' +\
                        f'RB_{params["b_size"]}_M_{params["M"]}_seed_{params["ac_seed"]}' +\
                        f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    print(params["log_path"])


    if alg_name == 'rpo':
        runner = Runner(T,gamma,lam,b_size,is_trunc,M,polweights)

    if params["alg_name"] == 'rpo':
        alg = rpo.RPO(sim_seed,env,actor,critic,runner,params)

    
    # Training
    alg.learn(sim_size,no_op_batches)   
    out = alg.dump(params)

    return out

def run1(**input):
    print('-------------')
    return 0

def run_wrapper(inputs_dict):
    return run(**inputs_dict)

def main(seeds, env_name='HalfCheetah-v3', alg_name='rpo', minibatch=32, rpo_noadapt=False,
         ppo_adapt=False, vf_lr=3e-4, vf_iters=10, cg_iter=10, residual_tol=1e-10, eps_ppo=0.2,
         damping=1e-2, max_kl=0.01, layers=[64, 64], kl_verify = 0.01, n_tr=4, n=1024, alpha=0.8,
         eps=[0.2, 0.15, 0.1, 0.1, 0.1], sim_size=1e6, actor_lr=3e-4, eval_sample_size=10, log_path='',
         eps_ppo_next=0.1, eps_ppo_next_reg=0.5):
    """Parses inputs, runs simulations, saves data"""
    parser = create_parser()

    args = parser.parse_args()
    args.eps_ppo_next = eps_ppo_next
    args.eps_ppo_next_reg = eps_ppo_next_reg
    args.alg_name = alg_name
    args.env_name = env_name
    args.minibatch = minibatch
    args.rpo_noadapt = rpo_noadapt
    args.ppo_adapt = ppo_adapt
    args.vf_lr = vf_lr
    args.vf_iters = vf_iters
    args.cg_iter = cg_iter
    args.residual_tol = residual_tol
    args.damping = damping
    args.max_kl = max_kl
    args.eps_ppo = eps_ppo
    args.actor_layers = layers
    args.vf_layers = layers
    args.kl_verify = kl_verify
    args.n_tr = n_tr
    args.n = n
    args.alpha = alpha
    args.eps = eps
    args.sim_size = sim_size
    args.actor_lr = actor_lr
    args.eval_sample_size = eval_sample_size
    args.runs = len(seeds)
    if log_path=='':
        args.log_path = './tf_log/'
    else:
        args.log_path = log_path

    if alg_name == 'rpo':
        args.log_path = args.log_path + f'{args.alg_name}/{args.alg_name}_next_clip_{args.eps_ppo_next}_next_reg_{args.eps_ppo_next_reg}_layers_{args.actor_layers[0]}_minibatch_{args.minibatch}_peppo_adapt_{int(args.ppo_adapt)}/{args.env_name}'



    inputs_dict = gather_inputs(args)

    ac_seeds = sim_seeds = seeds

    inputs_list = []
    for run in range(args.runs):
        if args.ac_seed is None:
            inputs_dict['ac_seed'] = int(ac_seeds[run])
        if args.sim_seed is None:
            inputs_dict['sim_seed'] = int(sim_seeds[run])

        inputs_list.append({**inputs_dict})


    with mp.Pool(args.cores) as pool:
        outputs = pool.map(run_wrapper,inputs_list)

    # Save data
    save_env = args.env_name.split('-')[0].lower()
    save_alg = args.alg_name.lower()
    save_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.save_file is None:
        save_file = '%s_%s_%s'%(save_env,save_alg,save_date)
    else:
        save_file = '%s_%s_%s_%s'%(save_env,save_alg,args.save_file,save_date)

    os.makedirs(args.save_path,exist_ok=True)
    save_filefull = os.path.join(args.save_path,save_file)

    with open(save_filefull,'wb') as f:
        pickle.dump(outputs,f)

if __name__=='__main__':
    mp.set_start_method('spawn')
    games = ['HalfCheetah-v3']
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main(seeds=[0], env_name=games[0], alg_name='rpo', eps_ppo_next=0.1, eps_ppo_next_reg=0.5)

