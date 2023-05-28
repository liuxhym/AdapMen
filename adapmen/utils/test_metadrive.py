from this import d
import torch
import numpy as np
from unstable_baselines.common.util import load_config
from adapmen.model import Actor, DiscreteActor, NewActor
from adapmen.model.actor import EnsembleActor
import numpy as np
import os
from tqdm import tqdm
from adapmen.env.expert_guided_env import ExpertGuidedEnv
from adapmen.env.absorbing_wrapper import AbsorbingWrapper
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import munch

def update_dataset(experiment_result, reward, cost, length, success_rate, tot_steps, expert_intervention):
    experiment_result['reward'].append(reward.copy())
    experiment_result['cost'].append(cost.copy())
    experiment_result['length'].append(length.copy())
    experiment_result['success_rate'].append(success_rate)
    experiment_result['tot_steps'].append(tot_steps)
    experiment_result['expert_intervention'].append(expert_intervention)

def test_metadrive(cfg, save_dir, log_dir, device):
	envrionment_config=dict(
				vehicle_config=dict(
					use_saver=False,
					free_level=100),
				safe_rl_env=True,
				controller="keyboard",
				use_render=False,
				manual_control=False,
				random_traffic=False,
				environment_num=20,
				random_agent_model=False,
				random_lane_width=False,
				random_lane_num=False,
				map=4,  # seven block
				start_seed=200,
			)

	test_num = 20
	
	writer = SummaryWriter(log_dir)

	env = ExpertGuidedEnv(envrionment_config)
	#if "value_dice" in save_dir:
	#	env = AbsorbingWrapper(env)

	# actor = Actor(env.observation_space, env.action_space.shape[0], cfg.sac.actor_network_params)
	actor = Actor(env.observation_space, env.action_space.shape[0], cfg.sac.actor_network_params).to(device)
	files = os.listdir(save_dir)
	'''
	print('\n')
	print(files[0])
	print('\n')
	'''
	files.sort(key=lambda x:int(x.split('-')[0]))

	experiment_result = {'reward': [], 'cost': [], 'length': [], 'success_rate':[], 'tot_steps': [], 'expert_intervention': []}
	
	for model in files: 
		save_file = os.path.join(save_dir, model)

		file_name = model.split('-')     

		if 'AdapMen' in save_dir or 'DAgger' in save_dir:
			tot_steps, expert_intervention = int(file_name[0]), int(file_name[1])
		else:
			tot_steps = int(file_name[0])
			expert_intervention = tot_steps

		# if(tot_steps>30000):
		#     continue
		
		# ###
		# tot_steps, expert_intervention= model.split('1')
		# tot_steps = expert_intervention 
		# ###

		#actor.actor_model.load_state_dict(torch.load(save_file, map_location=torch.device('cpu')))
		actor.actor_model.load_state_dict(torch.load(save_file, map_location=device))

		reward_list = []
		cost_list = []
		length_list = []
		success_num = 0

		for i in tqdm(range(test_num)):
			done = False
			#if "value_dice" in save_dir:
			#	s = env.reset()
			#else:
			s = env.reset( envrionment_config["start_seed"] + i )
			# 
			# env.render()
			reward = 0
			cost = 0
			step = 0

			while not done:
				#a = actor(torch.from_numpy(np.array([s], dtype=np.float32)))['mean']
				a = actor(torch.from_numpy(np.array([s], dtype=np.float32)).to(device))['mean']
				#a = a.squeeze().detach().numpy()
				a = a.squeeze().detach().cpu().numpy()
				s_, r, done, info = env.step(a)
				reward += r
				cost += info['cost']
				s = s_
				step += 1
				if step > 2500:      # Avoid getting stuck in the same position
					break
				if done and info["arrive_dest"]:
					success_num +=1
				# env.render()
			reward_list.append(reward)
			cost_list.append(cost)
			length_list.append(step)

		success_rate = success_num/test_num

		writer.add_scalar('eva/reward', np.mean(reward_list), tot_steps)
		writer.add_scalar('eva/invention_reward', np.mean(reward_list), expert_intervention)
		writer.add_scalar('eva/cost', np.mean(cost_list), tot_steps)
		writer.add_scalar('eva/length', np.mean(length_list), tot_steps)
		writer.add_scalar('eva/success_rate',  success_rate , tot_steps)
		writer.add_scalar('eva/invention_success', success_rate, expert_intervention)

		print( np.mean(reward_list), np.mean(cost_list), np.mean(length_list), success_rate, tot_steps, expert_intervention)
		update_dataset(experiment_result, np.mean(reward_list), np.mean(cost_list), np.mean(length_list), success_rate, tot_steps, expert_intervention)

	for k, v in experiment_result.items():
		experiment_result[k] = np.array(v)

	# np.savez(log_dir, **experiment_result)

def test_ensemble(cfg, save_dir, log_dir, device):
	envrionment_config=dict(
				vehicle_config=dict(
					use_saver=False,
					free_level=100),
				safe_rl_env=True,
				controller="keyboard",
				use_render=False,
				manual_control=False,
				random_traffic=False,
				environment_num=20,
				random_agent_model=False,
				random_lane_width=False,
				random_lane_num=False,
				map=4,  # seven block
				start_seed=200,
			)

	test_num = 20

	# writer = SummaryWriter(log_dir)  # 存放log文件的目录

	# 存放log文件
	
	writer = SummaryWriter(log_dir)

	env = ExpertGuidedEnv(envrionment_config)
	#if "value_dice" in save_dir:
	#	env = AbsorbingWrapper(env)

	# actor = Actor(env.observation_space, env.action_space.shape[0], cfg.sac.actor_network_params)
	#actor = Actor(env.observation_space, env.action_space.shape[0], cfg.sac.actor_network_params).to(device)
	actor = EnsembleActor(env.observation_space.shape[0], env.action_space.shape[0], cfg.actor.actor_network_params, num_actors=cfg.actor.num_actors)
	actor.to(device)
	
	files = os.listdir(save_dir)

	files.sort(key=lambda x:int(x.split('-')[0]))

	experiment_result = {'reward': [], 'cost': [], 'length': [], 'success_rate':[], 'tot_steps': [], 'expert_intervention': []}
	
	for model in files: 
		save_file = os.path.join(save_dir, model)

		file_name = model.split('-')     
		'''
		if 'AdapMen' in save_dir or 'DAgger' in save_dir:
			tot_steps, expert_intervention = int(file_name[0]), int(file_name[1])
		else:
			tot_steps = int(file_name[0])
			expert_intervention = tot_steps
		'''
		tot_steps = int(file_name[0])
		expert_intervention = int(file_name[1].split('.')[0])


		#actor.actor_model.load_state_dict(torch.load(save_file, map_location=torch.device('cpu')))
		actor = torch.load(save_file, map_location=device)

		reward_list = []
		cost_list = []
		length_list = []
		success_num = 0

		for i in tqdm(range(test_num)):
			done = False
			#if "value_dice" in save_dir:
			#	s = env.reset()
			#else:
			s = env.reset( envrionment_config["start_seed"] + i )
			# 
			# env.render()
			reward = 0
			cost = 0
			step = 0

			while not done:
				#a = actor(torch.from_numpy(np.array([s], dtype=np.float32)))['mean']
				#print(actor(torch.from_numpy(np.array([s], dtype=np.float32)).to(device)))
				a = actor.act(torch.from_numpy(np.array([s], dtype=np.float32)).to(device))['mean']
				#a = a.squeeze().detach().numpy()
				a = a.squeeze()
				s_, r, done, info = env.step(a)
				reward += r
				cost += info['cost']
				s = s_
				step += 1
				if step > 2500:      # Avoid getting stuck in the same position
					break
				if done and info["arrive_dest"]:
					success_num +=1
				# env.render()
			reward_list.append(reward)
			cost_list.append(cost)
			length_list.append(step)

		success_rate = success_num/test_num

		writer.add_scalar('eva/reward', np.mean(reward_list), tot_steps)
		writer.add_scalar('eva/invention_reward', np.mean(reward_list), expert_intervention)
		writer.add_scalar('eva/cost', np.mean(cost_list), tot_steps)
		writer.add_scalar('eva/length', np.mean(length_list), tot_steps)
		writer.add_scalar('eva/success_rate',  success_rate , tot_steps)
		writer.add_scalar('eva/invention_success', success_rate, expert_intervention)

		print( np.mean(reward_list), np.mean(cost_list), np.mean(length_list), success_rate, tot_steps, expert_intervention)
		update_dataset(experiment_result, np.mean(reward_list), np.mean(cost_list), np.mean(length_list), success_rate, tot_steps, expert_intervention)

	for k, v in experiment_result.items():
		experiment_result[k] = np.array(v)

	# np.savez(log_dir, **experiment_result)

if __name__ == "__main__":
	#save_dir = 'logs/metadrive/bc/bc--20230515-184402/save'
	save_dir = 'logs/metadrive/Ensemble_DAgger/DAgger--num_traj-0--initialize-0--20230516-162342/save'
	config_path = 'scripts/config/ensemble_dagger/metadrive/metadrive.py'
	log_dir = 'testlogs/metadrive/ensemble/1'
	device = torch.device('cuda')
	hparams_dict = load_config(config_path)
	cfg = munch.munchify(hparams_dict)
	#test_metadrive(cfg, save_dir, log_dir, device)
	test_ensemble(cfg, save_dir, log_dir, device)