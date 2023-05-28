overwrite_args ={
  "env":{
    "env_name" : 'Pong-v4',
    "expert_data_path": '../data/Pong-v4.npz',
  },
  "dagger":{
    "actor_lr" : 3e-4,
    "batch_size": 32,
    "bc_step": 0,
    "max_timesteps": 300000,
    "update_interval": 200,
    "updates_per_step": 200,
    "samplesteps": 2000,
  }
}
