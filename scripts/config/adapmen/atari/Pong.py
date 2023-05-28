overwrite_args ={
  "env":{
    "env_name" : 'Pong-v4',
    "expert_data_path": '../data/Pong-v4.npz',
  },
  "horizon": 5,
  "btq":{
    "update_interval": 100,
    "updates_per_step": 100,
    "update_p_freq": 100,
    "p_value": 0.05,
    "max_timesteps": 300000,
  }
}
