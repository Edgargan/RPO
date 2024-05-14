# Reflective Policy Optimization

This repository is the official implementation used to produce all experiments contained in the paper Reflective Policy Optimization (RPO). The RPO paper bas been accepted for ICML2024.



## Run an experiment

**Note that**: We provide Reflective Policy Optimization (RPO) code for both continuous and discrete environments (MuJoCo and Atari). 

Simulations can be run by calling `run` on the command line. For example, we can reproduce the results shown for RPO on HalfCheetah-v3 environment of MuJoCo as follows:

```
cd ./rpo_mujoco
python run_RPO.py
```

We can reproduce the results shown for PPO on BreakoutNoFrameskip-v4 environment of Atari as follows:

```
cd ./rpo_atari
python run_RPO.py
```



# Acknowledgements

- [GePPO](https://github.com/jqueeney/geppo)
- [DeepRL](https://github.com/ShangtongZhang/DeepRL)
