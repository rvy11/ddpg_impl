# DDPG Implementation

This repo contains an implementation of Deep Deterministic Policy Gradient, intended
for the D'Kitty quadruped robot.

### Prerequisites
Follow the instructions [here](https://github.com/google-research/robel) to set up MuJoCo and the ROBEL environment.

Additional packages needed (for Ubuntu 18.04 LTS): patchelf, libosmesa6-dev

We also discovered that we had to modify the LD_PRELOAD environment variable to allow MuJoCo to find the OpenGL Extension Wrangler library

```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so.2.0
```

### Run the program

```
$ python ./ddpg.py -h
usage: ddpg.py [-h] [-r RUN] [-f CONFIGFILE]

optional arguments:
  -h, --help            show this help message and exit
  -r RUN, --run RUN     Run simulation using provided trained model
  -f CONFIGFILE, --configfile CONFIGFILE
                        Use provided config (JSON) file for training

```

Train a policy with the given hyperparameters (see config file format below)

```
$ python ./ddpg.py -f test.json

```

Run pretrained policy

```
$ python ./ddpg.py -r saved_models/checkpoint_actor_dkitty_DDPG_noise_Normal_rand.pt

```

### Example config file

```
$ cat test.json 
{
    "policy":"DDPG",
    "env":"dkitty",
    "seed":0,
    "start_timesteps":1e4,
    "eval_freq":5e3,
    "max_timesteps":200000,
    "expl_noise":0.25,
    "replay_size":50000,
    "batch_size":256,
    "tau":0.005,
    "policy_noise":2,
    "action_noise":"Normal",
    "min_action":-1.0,
    "max_action":1.0
}
```
