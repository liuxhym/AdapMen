overwrite_args ={
  "env":{
    "env_name" : 'MsPacman-v4'
  },
  "horizon": 50,
  "btq":{
    "insert_all": False,
    "bc_type": "mse",
    "criterion_type": "q",
    "start_training_timesteps": 0,
    "actor_lr" : 3e-4,
    "batch_size": 32,
    "bc_step": 0,
    "max_timesteps": 100000,
    "update_interval": 200,
    "updates_per_step": 200,
    "target_prob": 0.2,
    "update_p_freq": 200,
    "p_value": 10,
    "samplesteps": 2000,
    "smooth": 10,
  }
}