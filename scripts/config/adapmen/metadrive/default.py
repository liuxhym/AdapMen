default_args ={
  "task_name": 'AdapMen',
  "algo": 'AdapMen',
  "log_interval": 1000,  ###200
  "eval_interval": 1000,
  "snapshot_interval": 50000,
  "buffer_size": 2000,
  "teacher": "sac-btq",
  "horizon": 100,

  "sac":{
    "actor_network_params": [("mlp", 256), ("mlp", 256)]
    # "actor_network_params": [256, 256],
  },
  "env":{
    "env_name" : '',
    "discount" : 0.99,
    "num_traj": 0,
    "expert_data_path": '',
    "use_normalize_states": True,
    "subsample_rate": 1.0,
  },
  "btq":{
    "insert_all": False,
    "bc_type": "mse",
    "criterion_type": "q",
    "start_training_timesteps": 0,
    "actor_lr" : 1e-4,
    "batch_size": 128,
    "bc_step": 0,
    "max_timesteps": 30000,
    "update_interval": 50,#200,
    "updates_per_step": 50,
    "target_prob": 0.2,
    "update_p_freq": 200,
    "p_value": 1,
    "samplesteps": 2000,
    "smooth": 5,
  }
}