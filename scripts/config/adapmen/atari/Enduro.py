overwrite_args ={
  "env":{
    "env_name" : 'Enduro-v4',
    "expert_data_path": '../data/Enduro-v4.npz',
  },
  
  "horizon": 50,
  "btq":{
    "update_interval": 100,
    "updates_per_step": 100,
    "update_p_freq": 100,
    "p_value": 1,
  }
}