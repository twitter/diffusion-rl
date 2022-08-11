import argparse
import gym
import numpy as np
import os
import torch
import json

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler


def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args):
    # Load buffer
    dataset = d4rl.qlearning_dataset(env)
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    utils.print_banner('Loaded buffer')

    if args.algo == 'bc':
        from agents.bc_diffusion import BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      model_type=args.model,
                      lr=args.lr)
    elif args.algo == 'pcq':
        from agents.ql_diffusion import PCQ as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      model_type=args.model,
                      eta=args.eta,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs * args.num_steps_per_epoch,
                      mode=args.mode)
    elif args.algo == 'ql_cvae':
        from agents.ql_cvae import QL_CVAE as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      eta=args.eta,
                      lr=args.lr)
    elif args.algo == 'ed_pcq':
        from agents.ed_pcq import ED_PCQ as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      model_type=args.model,
                      eta=args.eta,
                      lr=args.lr,
                      num_qs=20,
                      q_eta=1.0)
    elif args.algo == 'adw_bc':
        from agents.adw_bc_diffusion import ADW_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      model_type=args.model,
                      quantile=args.quantile,
                      temp=args.temp,
                      lr=args.lr)
    elif args.algo == 'qgdp':
        from agents.qgdp import QGDP as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      model_type=args.model,
                      quantile=args.quantile)
    
    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    best_score = -100.
    while training_iters < max_timesteps:
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        agent.train(data_sampler,
                    iterations=iterations,
                    batch_size=args.batch_size)
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed,
                                                                               eval_episodes=args.eval_episodes)
        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std])
        np.save(os.path.join(output_dir, "eval"), evaluations)

        # record and save the best model
        if eval_norm_res >= best_score:
            if args.save_best_model: agent.save_model(output_dir)
            best_score = eval_norm_res
            best_res = {'epoch': curr_epoch, 'best normalized score avg': eval_norm_res,
                        'best normalized score std': eval_norm_res_std,
                        'best raw score avg': eval_res, 'best raw score std': eval_res_std}
            with open(os.path.join(output_dir, "best_score.txt"), 'w') as f:
                f.write(json.dumps(best_res))


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    return avg_reward, std_reward, avg_norm_score, std_norm_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)             # Experiment ID
    parser.add_argument('--device', default=0, type=int)                  # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="walker2d-expert-v2", type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="tests", type=str)  # Logging directory
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds

    parser.add_argument("--num_epochs", default=2000, type=int)
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--reward_tune", default='no', type=str)

    parser.add_argument('--save_best_model', action='store_true')
    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--lr_decay", action='store_true')
    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--max_q_backup", action='store_true')

    ### Diffusion Setting ###
    parser.add_argument("--T", default=100, type=int)
    parser.add_argument("--beta_schedule", default='linear', type=str)
    ### Algo Choice ###
    parser.add_argument("--model", default='MLP', type=str)  # ['MLP', MLP_Unet']
    parser.add_argument("--algo", default="bc", type=str)  # ['bc', 'pcq']
    # algo specific parameters
    parser.add_argument("--eta", default=0.25, type=float)
    parser.add_argument("--temp", default=3.0, type=float)
    parser.add_argument("--quantile", default=0.7, type=float)
    parser.add_argument("--num_qs", default=5, type=int)
    parser.add_argument("--q_eta", default=1.0, type=float)
    parser.add_argument("--mode", default='whole_grad', type=str, help="['last_few', 'whole_grad']")

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'results/{args.dir}'

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}|{args.beta_schedule}|T-{args.T}|{args.algo}|{args.model}|lr{args.lr:.5f}"
    if args.lr_decay: file_name += '|lr_decay'
    if args.algo != 'bc':
        file_name += f'|r_tune-{args.reward_tune}'
        if args.algo == 'pcq' or args.algo == 'ql_cvae':
            file_name += f'|eta{args.eta}|{args.mode}'
        elif args.algo == 'ed_pcq':
            file_name += f'|eta{args.eta}|q_eta{args.q_eta}|num_qs{args.num_qs}'
        elif args.algo == 'adw_bc':
            file_name += f'|temp{args.temp}|q{args.quantile}'
        elif args.algo == 'qgdp':
            file_name += f'|q{args.quantile}'
        if args.max_q_backup: file_name += '|max_q_backup'
    file_name += f'|{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    # if os.path.exists(os.path.join(results_dir, 'variant.json')):
    #     raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)
    variant.update(version=f"Diffusion-QL")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env,
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                args)
