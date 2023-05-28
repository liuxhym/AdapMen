# Active adaptive expert involvement (AdapMen)

[AAMAS 23] Official implementation for AdapMen: How to Guide Your Learner: Imitation Learning with Active Adaptive Expert Involvement

## Installation

Before downloading the main code, you need to download the code from [unstable baselines](https://github.com/x35f/unstable_baselines/tree/AdapMen), which is required for the code to run properly.

```
git clone -b AdapMen https://github.com/x35f/unstable_baselines.git
cd unstable_baselines
pip install -e .
```

Then, download the main code from [AdapMen](https://github.com/liuxhym/AdapMen).

```
git clone https://github.com/liuxhym/AdapMen.git
cd AdapMen
pip install -e .
```



## Usage

To reproduce the experiments on MetaDrive, you should train a SAC expert using [unstable baselines](https://github.com/x35f/unstable_baselines/tree/AdapMen) first while for Atari, a DQN one.  

```
cd unstable_baselines/unstable_baselines/baselines/[sac,dqn]
python main.py config/[enviroment_name]/[task_name].py args(optional)
```

With the expert, experiments can be reproduced with the following:

```
cd AdapMen
python scripts/run_[algorithm_name].py scripts/config/[algorithm_name]/[enviroment_name]/[task_name].py args(optional)
```

If you want to conduct experiments with a human expert, please run algorithms as following:

```
python scripts/run_[algorithm_name].py scripts/config/[algorithm_name]/[enviroment_name]/[task_name]-human.py args(optional)
```

