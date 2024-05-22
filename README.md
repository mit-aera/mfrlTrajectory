## Multi-Fidelity Reinforcement Learning for Time-Optimal Quadrotor Re-planning

This repository contains the code for our recent paper entitled ["Multi-Fidelity Reinforcement Learning for Time-Optimal Quadrotor Re-planning"](https://arxiv.org/abs/2403.08152).

[![Video Link](res/real_world_exp_cam.png)](https://youtu.be/75AbKY3L5As)

### Installation
After cloning the repo, please build the docker container with the following script:
```bash
docker build -t mfrl .
./run_docker.sh
# The Docker image can also be downloaded from Google Drive:
# "https://drive.google.com/file/d/1COQMyMPrRllJp2Ljhu8yrxTUyvxTZCWK/view?usp=sharing"
```

### MFRL
The Multi-Fidelity Reinforcement Learning (MFRL) framework utilizes the minimum snap trajectory dataset during both the pretraining and reinforcement learning phases. The trajectory dataset can be either downloaded from our Google Drive or generated from scratch using the `gendata_minsnap_traj.py` script.
```bash
# Download minsnap dataset
mkdir -p dataset/mfrl_online
cd dataset
gdown --fuzzy "https://drive.google.com/file/d/1RBxDiPWAS_3tm2pTPYhxd405Iyn2LzOU/view?usp=sharing"
tar -xvf mfrl_dataset.tar.gz -C . --strip-components=1
# Generate minsnap dataset (instead of downloading the dataset)
cd scripts_datagen
nice -n 15 python3 gendata_minsnap_traj.py
```

The trajectory dataset is furtheraugmented and preprocessed using the following scripts:
```bash
# Augment and preprocess the waypoint sequence dataset
cd scripts_datagen
# Augment and preprocess train dataset
nice -n 15 python3 gendata_aug_tnorm.py -a
# Preprocess test dataset
nice -n 15 python3 gendata_aug_tnorm.py -t
# Slicing waypoint sequences
nice -n 15 python3 gendata_presample.py
```

To train the policy in simulated environments, execute `./schedule_exp_sim.sh`. This script employs a simple dynamics model, detailed in `pyTrajectoryUtils/pyTrajectoryUtils/quadModel.py`, for low-fidelity evaluations, and uses the [FlightGoggles](https://github.com/mit-aera/pyFlightGoggles) control simulation for high-fidelity evaluations. For real-world experiment training, run `./schedule_exp_real.sh`. This process requires inputting real-world experimental results during training, formatted as `logs/mfrl_robot_eval_data/ep_xx_res.txt.template`.

```bash
# Start MFRL training
cd scripts_mfrl
# MFRL training with simulated experiments
./schedule_exp_sim.sh
# MFRL training with robot experiments
./schedule_exp_robot.sh
```

### MFRL dataset structure
```bash
- "train"
    - num_wp
        - points_sta (num_data x num_wp x [x, y, z, yaw, eos, time, snapw]) # dataset for low-fidelity evaluation
        - points_sim (num_data_sim x num_wp x [x, y, z, yaw, eos, time, snapw]) # dataset for simulation
        - points_real (num_data_real x num_wp x [x, y, z, yaw, eos, time, snapw]) # dataset for real-world experiments
        - alpha_sta (num_data)
        - alpha_sim (num_data_sim)
        - alpha_real (num_data_real)
- "test"
    - num_wp
        - points (num_data x num_wp x [x, y, z, yaw, eos, time, snapw])
        - rand_idx (num_data_real) # Index of datapoints for real-world experiments
        - alpha_sta (num_data)
        - alpha_sim (num_data)
        - alpha_real (num_data_real)

# num_wp = 5, ... , 14
# download and save it under "dataset" directory
```


### MFBO
This repository also includes the code for optimizing a single trajectory using Bayesian optimization, as published in our previous paper: ["Multi-fidelity black-box optimization for time-optimal quadrotor manuevers"](https://journals.sagepub.com/doi/pdf/10.1177/02783649211033317).
```bash
cd scripts_mfbo

# Single-fidelity BO with simulation
nice -n 15 python3 run_mfbo_simple.py

# Multi-fidelity BO with simulation
nice -n 15 python3 run_mfbo.py

# Multi-fidelity BO with robot experiment
nice -n 15 python3 run_mfbo.py -r
```
Minimium-snap trajectory generation procedure can be found in the notebook `notebooks/run_mfbo.ipynb`.


### Citation
If you find this work useful for your research, please cite:
```bibtex
@article{ryou2024multi,
  title={Multi-Fidelity Reinforcement Learning for Time-Optimal Quadrotor Re-planning},
  author={Ryou, Gilhyun and Wang, Geoffrey and Karaman, Sertac},
  journal={arXiv preprint arXiv:2403.08152},
  year={2024}
}
```
