import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".4"

import cv2
import time
import clip
import optax
import imageio
import ml_collections
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

from tqdm import trange
from models import SACAgent, FuRLAgent, RewardModel
from utils import (TASKS, DistanceBuffer, EmbeddingBuffer, log_git,
                   get_logger, make_env, load_liv,TASKS_TARGET)
import wandb
import jax.numpy as jnp


###################
# Utils Functions #
###################
def crop_center(config, image):
    x1, x2, y1, y2 = 32, 224, 32, 224
    return image[x1:x2, y1:y2, :]


def eval_policy(agent: SACAgent,
                env: gym.Env,
                eval_episodes: int = 10, success_reward: float = 0.0):
    t1 = time.time()
    eval_reward, eval_success, avg_step = 0, 0, 0
    for i in range(1, eval_episodes + 1):
        obs, _ = env.reset()
        while True:
            avg_step += 1
            action = agent.sample_action(obs, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            eval_reward += reward
            if terminated or truncated:
                eval_success += eval_reward > success_reward
                break

    eval_reward /= eval_episodes
    eval_success /= eval_episodes

    return eval_reward, eval_success, avg_step, time.time() - t1


def setup_logging(config):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # logging
    exp_prefix = f"furl_rho{config.rho}"
    exp_name = f"{exp_prefix}/{config.env_name}/s{config.seed}_{timestamp}"
    os.makedirs(f"logs/{exp_prefix}/{config.env_name}", exist_ok=True)
    exp_info = f"# Running experiment for: {exp_name} #"
    print("#" * len(exp_info) + f"\n{exp_info}\n" + "#" * len(exp_info))
    logger = get_logger(f"logs/{exp_name}.log")

    # Initialize wandb
    wandb.init(
        project="furl_mujoco",  # Change this to your wandb project name
        name=exp_name,
        config=config.to_dict(),
    )
    wandb.config.update({"timestamp": timestamp})  # Additional metadata

    # add git commit info
    log_git(config)
    logger.info(f"Config:\n{config}\n")

    # set random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    return exp_name, logger


def setup_exp(config):
    # liv
    transform = T.Compose([T.ToTensor()])
    liv = load_liv()

    # task description embedding
    with torch.no_grad():
        token = clip.tokenize([TASKS[config.env_name]])
        text_embedding = liv(input=token, modality="text")
    text_embedding = text_embedding.detach().cpu().numpy()
    if config.goal:
        data = np.load(f"data/oracle/{config.env_name}/s0_c{config.camera_id}.npz")
    else:
        data = np.load(f"data/oracle/door-open-v2-goal-hidden/s0_c2.npz")

    # goal_embedding / text_embedding
    oracle_images = data["images"]
    oracle_success = data["success"]
    oracle_traj_len = np.where(oracle_success)[0][0] + 1  # 84

    # initialize the environment
    env = make_env(config.env_name,
                   seed=config.seed,
                   camera_id=config.camera_id)
    eval_seed = config.seed if "hidden" in config.env_name else config.seed+100
    eval_env = make_env(config.env_name,
                        seed=eval_seed,
                        image_size=224,
                        camera_id=config.camera_id)

    # environment parameter
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    goal_image = data["images"][oracle_traj_len-1]
    goal_image = crop_center(config, goal_image)
    processed_goal_image = cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR)
    processed_goal_image = transform(processed_goal_image)
    goal_embedding = liv(input=processed_goal_image.to("cuda")[None], modality="vision")
    goal_embedding = goal_embedding.detach().cpu().numpy()

    # fixed LIV representation projection
    vlm_agent = FuRLAgent(obs_dim=obs_dim,
                          act_dim=act_dim,
                          max_action=max_action,
                          seed=config.seed,
                          tau=config.tau,
                          rho=config.rho,
                          margin=config.cosine_margin,
                          gamma=config.gamma,
                          lr=config.lr,
                          text_embedding=text_embedding,
                          goal_embedding=goal_embedding,
                          hidden_dims=config.hidden_dims)

    # SAC agent
    sac_agent = SACAgent(obs_dim=obs_dim,
                         act_dim=act_dim,
                         max_action=max_action,
                         seed=config.seed,
                         tau=config.tau,
                         gamma=config.gamma,
                         lr=config.lr,
                         hidden_dims=config.hidden_dims)

    # Initialize the reward model
    reward_model = RewardModel(seed=config.seed,
                               state_dim=obs_dim,
                               text_embedding=text_embedding,
                               goal_embedding=goal_embedding)

    # Replay buffer
    replay_buffer = DistanceBuffer(obs_dim=obs_dim,
                                   act_dim=act_dim,
                                   max_size=int(5e5))
    # Replay buffer
    replay_buffer_distill = DistanceBuffer(obs_dim=obs_dim,
                                   act_dim=act_dim,
                                   max_size=int(5e5))
    return (
        transform,
        liv,
        env,
        eval_env,
        vlm_agent,
        sac_agent,
        reward_model,
        replay_buffer,
        goal_image,
        replay_buffer_distill,
    )


#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()

    # logging setup
    exp_name, logger = setup_logging(config)
 
    # experiment setup
    (transform,
     liv,
     env,
     eval_env,
     vlm_agent,
     sac_agent,
     reward_model,
     replay_buffer,
     goal_image,
     replay_buffer_distill) = setup_exp(config)
    sucess_reward = TASKS_TARGET[config.env_name]

    # reward for untrained agent
    eval_episodes = 1 if "hidden" in config.env_name else 10
    eval_reward, eval_success, _, _ = eval_policy(vlm_agent,
                                                  eval_env,
                                                  eval_episodes,sucess_reward)
    logs = [{
        "step": 0,
        "eval_reward": eval_reward,
        "eval_success": eval_success,
    }]

    first_success_step = 0

    # trajectory embedding
    embedding_buffer = EmbeddingBuffer(emb_dim=1024,
                                       gap=config.gap,
                                       max_size=config.embed_buffer_size)
    traj_embeddings = np.zeros((1000, 1024))
    traj_success = np.zeros(1000)

    # relay freqs
    relay_freqs = [50, 100, 150, 200]
    relay_freq = np.random.choice(relay_freqs)
    logger.info(f"Relay freqs: {relay_freqs}\n")

    # start training
    obs, _ = env.reset()
    goal = obs[-3:]
    reward, ep_task_reward, ep_vlm_reward = 0, 0, 0
    success_cnt, ep_num, ep_step = 0, 0, 0
    lst_ep_step, lst_ep_task_reward, lst_ep_vlm_reward = 0, 0, 0
    sac_step, vlm_step = 0, 0
    lst_sac_step, lst_vlm_step = 0, 0
    policies = ["vlm", "sac"]
    use_relay = True
    pos_cosine = neg_cosine = lag_cosine = 0
    pos_cosine_max = neg_cosine_max = lag_cosine_max = 0
    pos_cosine_min = neg_cosine_min = lag_cosine_min = 0
    neg_num = neg_loss = neg_loss_max = 0
    pos_num = pos_loss = pos_loss_max = 0
    for t in trange(1, config.max_timesteps + 1):
        if t <= config.start_timesteps:
            action = env.action_space.sample()
        else:
            if use_relay:
                if policies[(ep_step//relay_freq)%2] == "vlm":
                    vlm_step += 1
                    action = vlm_agent.sample_action(obs)
                else:
                    sac_step += 1
                    action = sac_agent.sample_action(obs)
                    action_noise = np.random.normal(
                        0, sac_agent.max_action*config.expl_noise, size=sac_agent.act_dim)
                    action = (action + action_noise).clip(
                        -sac_agent.max_action, sac_agent.max_action)
            else:
                vlm_step += 1
                action = vlm_agent.sample_action(obs)
        next_obs, task_reward, terminated, truncated, info = env.step(action)

        # vision language model reward
        image = env.mujoco_renderer.render(
            render_mode="rgb_array",
            camera_id=config.camera_id).copy()
        image = image[::-1]
        image = crop_center(config, image)
        processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed_image = transform(processed_image)
        with torch.no_grad():
            if t%config.liv_freq == 0:
                image_embedding = liv(input=processed_image.to("cuda")[None], modality="vision") # torch.Size([1, 1024])
                image_embedding = image_embedding.detach().cpu().numpy()
                # add to buffer
                replay_buffer_distill.add(obs,
                                action,
                                next_obs,
                                reward,
                                terminated,
                                image_embedding,
                                l2_distance)
            else:
                image_embedding = reward_model.get_state_embeddings(reward_model.proj_state, jnp.array(next_obs)) # next_obs (151,)
        if config.goal:
            l2_distance = np.square(image_embedding - vlm_agent.goal_embedding).sum(-1)**0.5
        else:
            l2_distance = 0
        vlm_reward = reward_model.get_vlm_reward(reward_model.proj_state, image_embedding).item()*0.1

        reward = task_reward
        success = (ep_task_reward > sucess_reward)+0.0
        success_cnt += success

        traj_embeddings[ep_step] = image_embedding
        traj_success[ep_step] = success
        ep_step += 1

        if first_success_step == 0 and success:
            first_success_step = ep_step

        # add to buffer
        replay_buffer.add(obs,
                          action,
                          next_obs,
                          reward,
                          terminated,
                          image_embedding,
                          l2_distance)
        obs = next_obs
        ep_vlm_reward += vlm_reward
        ep_task_reward += task_reward

        # start a new trajectory
        if terminated or truncated:
            obs, _ = env.reset()
            goal = obs[-3:]
            lst_ep_step = ep_step
            lst_ep_task_reward = ep_task_reward
            lst_ep_vlm_reward = ep_vlm_reward
            lst_sac_step = sac_step
            lst_vlm_step = vlm_step
            ep_vlm_reward = 0
            ep_task_reward = 0
            sac_step = 0
            vlm_step = 0
            policies = policies[::-1]
            relay_freq = np.random.choice(relay_freqs)

            # save embedding
            if first_success_step == 0:
                for j in range(ep_step):
                    embedding_buffer.add(embedding=traj_embeddings[j],
                                         success=False)
            else:
                for j in range(first_success_step):
                    embedding_buffer.add(embedding=traj_embeddings[j],
                                         success=True,
                                         valid=j>=config.gap)

                for j in range(first_success_step, ep_step):
                    if traj_success[j]:
                        embedding_buffer.add(embedding=traj_embeddings[j],
                                             success=True,
                                             valid=j>=config.gap)
                    else:
                        break

            ep_step = 0
            ep_num += 1
            first_success_step = 0

            if use_relay and embedding_buffer.pos_size >= config.relay_threshold:
                use_relay = False

        # training
        if  t > config.start_timesteps:
            if t % config.train_freq == 0:
                for i in range(config.gradient_steps):
                    if (success_cnt > 0) and (embedding_buffer.valid_size > 0):
                        batch = replay_buffer.sample(config.batch_size)
                        batch_distill = replay_buffer_distill.sample(config.batch_size)
                        embedding_batch = embedding_buffer.sample(config.batch_size)
                        if t % (config.train_freq*10) == 0:
                            proj_log_info = reward_model.update_pos(embedding_batch)
                            loss_info = reward_model.update_state(batch_distill)
                        batch_vlm_rewards = reward_model.get_vlm_reward(reward_model.proj_state, batch.embeddings)
                        log_info = vlm_agent.update(batch, batch_vlm_rewards)

                    # # # collected zero successful trajectory
                    else:
                        batch = replay_buffer.sample_with_mask(config.batch_size, config.l2_margin)
                        batch_distill = replay_buffer_distill.sample(config.batch_size)
                        if t % (config.train_freq*10) == 0:
                            loss_info = reward_model.update_state(batch_distill)
                        proj_log_info = reward_model.update_neg(batch)
                        batch_vlm_rewards = proj_log_info.pop("vlm_rewards")
                        log_info = vlm_agent.update(batch, batch_vlm_rewards)
                #     # if t==config.start_timesteps+1:
                #     #     proj_log_info = reward_model.update_neg(batch)
                #     #     batch_vlm_rewards = proj_log_info.pop("vlm_rewards")
                #     #     log_info = vlm_agent.update(batch, batch_vlm_rewards)
                    #     pos_loss = proj_log_info["pos_loss"]

                    # update SAC agent
                    if use_relay: _ = sac_agent.update(batch)

        # eval
        if t % config.eval_freq == 0:
            eval_reward, eval_success, _, _ = eval_policy(vlm_agent,
                                                          eval_env,
                                                          eval_episodes,sucess_reward)

        # logging
        if t % config.log_freq == 0 and t % config.train_freq == 0:
            if t > config.start_timesteps:
                # print(loss_info)
                log_info_now = {
                    "step": t,
                    "success": success,
                    "task_reward": lst_ep_task_reward,
                    "vlm_reward": lst_ep_vlm_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "state_loss": loss_info['mse_loss'],
                    "time": (time.time() - start_time) / 60,
                    "global_step": t,
                }
                wandb.log(log_info_now)
            else:
                logs.append({
                    "step": t,
                    "task_reward": lst_ep_task_reward,
                    "vlm_reward": lst_ep_vlm_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "time": (time.time() - start_time) / 60,
                })
                logger.info(
                    f"\n[T {t//1000}K][{logs[-1]['time']:.2f} min] "
                    f"task_reward: {lst_ep_task_reward:.2f}, "
                    f"vlm_reward: {lst_ep_vlm_reward:.2f}\n"
                )
            # # save logs
            # log_df = pd.DataFrame(logs)
            # log_df.to_csv(f"logs/{exp_name}.csv")


    # # save logs
    # log_df = pd.DataFrame(logs)
    # log_df.to_csv(f"logs/{exp_name}.csv")

    wandb.finish()
    # close env
    env.close()
    eval_env.close()