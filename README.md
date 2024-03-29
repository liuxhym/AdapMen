# Active adaptive expert involvement (AdapMen)

[AAMAS 23] Official implementation for AdapMen: How to Guide Your Learner: Imitation Learning with Active Adaptive Expert Involvement

## Installation

Ddownload and install the main code from [AdapMen](https://github.com/liuxhym/AdapMen).

```
git clone https://github.com/liuxhym/AdapMen.git
cd AdapMen
pip install -e .
```

Install the [unstable_baselines](https://github.com/x35f/unstable_baselines)

```
git clone https://github.com/x35f/unstable_baselines -b AdapMen
cd unstable_baselines
pip install -e .
```

## Usage

An expert model for MetaDrive is provided. You can reproduce the experiments on MetaDrive with the following commands:
```
python scripts/run_[algorithm_name].py scripts/config/[algorithm_name]/metadrive/metadrive.py args(optional)
```

To reproduce your own expert on MetaDrive,train a SAC expert and for Atari, a DQN expert.  

```
cd unstable_baselines/unstable_baselines/baselines/[sac,dqn]
python main.py config/[enviroment_name]/[task_name].py args(optional)
```

With the trained experts, experiments can be reproduced with the following commands:

```
cd AdapMen
python scripts/run_[algorithm_name].py scripts/config/[algorithm_name]/[enviroment_name]/[task_name].py args(optional)
```

If you want to conduct experiments with a human expert, please run algorithms as following:

```
python scripts/run_[algorithm_name].py scripts/config/[algorithm_name]/[enviroment_name]/[task_name]_human.py args(optional)

