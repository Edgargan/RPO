from deep_rl import *

def ppo_pixel(seed, tag, **kwargs):
    generate_tag1(seed, tag, kwargs)
    kwargs.setdefault('skip', False)
    config = Config()

    config.base_path = './Result/DeepRL_result/'
    if not os.path.exists(config.base_path):
        os.makedirs(config.base_path)


    config.env = kwargs['game']
    config.seed = seed

    random_seed(seed=config.seed)
    config.merge(kwargs)
    config.ppo_ratio_clip = 0.1



    # save medels
    filename = config.tag + "_n_c_" + str(config.ppo_next_ratio_clip) + '_regu_' + str(config.regu)
    filename_path = os.path.join(config.base_path, 'Atari_Models', config.env, filename, config.tag_1
                                  + '_' + datetime.datetime.now().strftime(
                                      "%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(filename_path):
        mkdir(filename_path)
    config.base_path_models = filename_path
    filename_path = os.path.join(config.base_path, 'Atari_log_txt', config.env, filename)
    if not os.path.exists(filename_path):
        mkdir(filename_path)
    config.base_path_log_txt = filename_path
    filename_path = os.path.join(config.base_path, 'Atari_tf_log', config.env, filename)
    if not os.path.exists(filename_path):
        mkdir(filename_path)
    config.base_path_tf_log = filename_path


    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers, seed=config.seed)
    config.eval_env = Task(config.game)
    config.num_workers = 16     
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=2.5e-4)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 256      
    config.optimization_epochs = 4
    config.mini_batch_size = config.rollout_length * config.num_workers // 4
    config.log_interval = config.rollout_length * config.num_workers
    config.shared_repr = True
    config.max_steps = int(5e7)
    run_steps(RPO(config))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ["OMP_NUM_THREADS"] = '4'

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BreakoutNoFrameskip-v4")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()
    # mkdir('log')
    # mkdir('tf_log')
    random_seed(seed=args.seed)
    select_device(0)
    ppo_pixel(seed=args.seed, tag='RPO', game=args.env, regu = 3.0, ppo_next_ratio_clip=0.1)





