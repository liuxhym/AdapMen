import torch
import numpy as np
#from adapmen.env.safe_env import BenchmarkEnv

def evaluate(actor, env, device, num_episodes=10):
    if env.discrete:
        return discrete_evaluate(actor, env, device, num_episodes=num_episodes)
    else:
        return naive_evaluate(actor, env, device, num_episodes=num_episodes)

def naive_evaluate(actor, env, device, num_episodes=1):
    total_timesteps = 0
    total_returns = 0
    success_num = 0

    actor.eval()

    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = actor(torch.from_numpy(np.array([state], dtype=np.float32)).to(device))['mean']
                action = action[0].cpu().numpy()


                next_state, reward, done, info = env.step(action)
                total_returns += reward
                total_timesteps += 1
                state = next_state
                if done:
                    if info.get("arrive_dest",0):
                        success_num += 1
                    

    return {
        "average_return":total_returns / num_episodes,
        "average_c":  0,
        "average_length": total_timesteps / num_episodes,
        "success_rate": success_num / num_episodes
    }

def discrete_evaluate(actor, env, device, num_episodes=1):
    total_timesteps = 0
    total_returns = 0

    actor.eval()

    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            l = 0
            while not done:
                action = actor(torch.from_numpy(np.array([state], dtype=np.float32)).to(device))['mode']
                action = action[0].cpu().numpy()[0]
                #print(action)
                next_state, reward, done, _ = env.step(action)

                total_returns += reward
                total_timesteps += 1
                state = next_state
                l += 1
                if l >=2000:
                    break

    return {
        "average_return":total_returns / num_episodes,
        "average_c":  0,
        "average_length": total_timesteps / num_episodes
    }
