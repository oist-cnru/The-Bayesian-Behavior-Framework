import argparse
import os
import time
import logging
import numpy as np
import scipy.io as sio
import torch
import multiprocessing
import warnings

from buffer import ReplayBuffer
from model import BayesianBehaviorAgent

# ================================ Macro ======================================
parser = argparse.ArgumentParser()

logging.basicConfig(level=logging.INFO)

# ----------- General properties -------------
parser.add_argument('--max_all_steps', type=int, default=650000, help="total environment steps")
parser.add_argument('--stage_3_start_step', type=int, default=500000, help="stage 3 start step")

parser.add_argument('--verbose', type=float, default=0, help="Verbose")
parser.add_argument('--n_seed', type=int, default=1, help="number of seeds")
parser.add_argument('--seed', type=int, default=0, help="starting seed")
parser.add_argument('--gui', type=int, default=0, help="whether to show Pybullet GUI")

parser.add_argument('--recording_steps', type=int, default=1000, help="num steps to record detailed behavior (in the end of training)")
parser.add_argument('--record_final_z_q', type=int, default=0, help="whether to record the final posterior z after full AIf")

# ----------- Network hyper-parameters ----------
parser.add_argument('--beta_z', type=float, default=0.1, help="coefficient of loss function of KLD of z")
parser.add_argument('--decision_precision_threshold', type=float, default=0.05, help="decision precision threshold")

# ----------- RL hyper-parameters -----------
parser.add_argument('--step_start', type=int, default=100000, help="steps starting training")

# ==================== arg parse & hyper-parameter setting ==================
savepath = './data/'
details_savepath = './details/'

args = parser.parse_args()

reward_scales_left = [1000, 500]
reward_scales_right = [0, 500]

def run_trial(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(seed % torch.cuda.device_count()))
        torch.cuda.manual_seed_all(0)
    else:
        device = torch.device("cpu")

    if os.path.exists(savepath):
        logging.info('{} exists (possibly so do data).'.format(savepath))
    else:
        try: os.makedirs(savepath)
        except: pass

    if os.path.exists(details_savepath):
        logging.info('{} exists (possibly so do data).'.format(details_savepath))
    else:
        try: os.makedirs(details_savepath)
        except: pass

    # ========================= T-Maze environment initialization ============================
    logging.info("Model-saved-at-{}".format(savepath))
    logging.info("Job-name-is-{}".format(os.getenv('AMLT_JOB_NAME')))

    from env.tmaze import TMazeEnv

    PyBulletClientMode = 'GUI' if args.gui else 'DIRECT'
    env = TMazeEnv(mode=PyBulletClientMode, obs='vision', seed=seed, reward_scales=reward_scales_left)
    task_name = "tmaze"

    max_all_steps = args.max_all_steps
    max_steps = 60  # maximum steps in one episode

    stage_3_start_step = args.stage_3_start_step

    # =============================== Hyperparameters ================================
    verbose = args.verbose

    rl_config = {"algorithm": "sac",
                 "gamma": 0.9,
                 "target_entropy": 0}
    train_interval = 5
    batch_size = 60
    seq_len = max_steps
    max_num_seq = int(2 ** 13)  # buffer size
    record_internal_states = True
    step_perf_eval = max(max_steps, int(max_all_steps / 2000)) # record performance every 0.05% of training
    recording_steps = args.recording_steps
    step_start = int(args.step_start)
    if step_start < 100000:
        warnings.warn("step_start should be > 50000 to collect enough experience for training")
    
    step_end = max_all_steps - recording_steps

    input_size = env.observation_space.shape
    action_size = env.action_space.shape[0]

    # ==================================== Initiliaze agent and replay buffer  =================================
    buffer = ReplayBuffer(env.observation_space.shape, env.action_space.shape,
                          device=device, batch_size=batch_size,
                          max_num_seq=max_num_seq, seq_len=seq_len, obs_uint8=True)

    agent = BayesianBehaviorAgent(input_size=input_size,
                                  action_size=action_size,
                                  beta_z=args.beta_z,
                                  rl_config=rl_config,
                                  record_final_z_q=args.record_final_z_q,
                                  decision_precision_threshold=args.decision_precision_threshold,
                                  device=device)

    agent.record_internal_states = record_internal_states

    # ====================================== Init recording data =======================
    performance_wrt_step = []
    global_steps = []
    steps_taken_wrt_step = []

    global_step = 0

    learned_behavior = []

    observations_rewarded_left = []
    observations_rewarded_right = []

    aif_iterations = []
    sig_priors = []
    sig_posts = []
    loss_all = []
    loss_v_all = []
    loss_q_all = []
    loss_a_all = []
    kld_all = []
    logp_x_all = []
    rewards_all = []

    episode = 0

    episode_record_before_devaluation = 0
    episode_record_after_devaluation = 0

    # ===================================== Experiment Start ============================================
    while global_step <= max_all_steps:

        sp = env.reset()
        if isinstance(sp, tuple):   # compatible with new Gym API
            sp = sp[0].astype(np.float32)
        else:
            sp = sp.astype(np.float32)

        observations = np.zeros([max_steps + 1, *env.observation_space.shape], dtype=np.float32)
        actions = np.zeros([max_steps, *env.action_space.shape], dtype=np.float32)
        rs = np.zeros([max_steps], dtype=np.float32)
        dones = np.zeros([max_steps], dtype=np.float32)
        infos = np.zeros([max_steps + 1, 2], dtype=np.float32)  # position infomation

        t = 0
        r = 0
        observations[0] = sp

        infos[0] = env.info['ob']  # specific for this task

        agent.init_states(- 4 * (global_step - step_start) / (max_all_steps - step_start)) # anneal motor noise by linearly decreasing target policy entropy

        if global_step >= step_start - max_steps - 1:
            if global_step >= stage_3_start_step:  # stage 3
                env.reward_scales = reward_scales_right
                sampled_idx = np.random.randint(0, len(observations_rewarded_right))
                s_goal = observations_rewarded_right[sampled_idx]
            else:  # stage 1, 2
                env.reward_scales = reward_scales_left
                sampled_idx = np.random.randint(0, len(observations_rewarded_left))
                s_goal = observations_rewarded_left[sampled_idx]

        for t in range(max_steps):
            
            start_time = time.time()

            if global_step % step_perf_eval == 0 and global_step >= step_start - 1:
                performance_wrt_step.append(episode_return)
                steps_taken_wrt_step.append(episode_length)
                global_steps.append(global_step)

            if global_step < step_start + max_steps: # random action at initial exploration stage
                sp, r, done, info, action = agent.step_with_env(env, sp, None, behavior="habitual")
            else:
                sp, r, done, info, action = agent.step_with_env(env, sp, s_goal, behavior="synergized")

            aif_iterations.append(agent.aif_iterations)
            sig_priors.append(agent.sigz_p_t.detach().cpu().numpy())
            sig_posts.append(agent.sigz_q_t.detach().cpu().numpy())
            
            observations[t + 1], infos[t + 1] = sp, info['ob']    # specific for this task
            actions[t], rs[t], dones[t] = action, r, done
            global_step += 1
            
            if global_step == max_all_steps + 1:
                break

            if global_step == stage_3_start_step: # stage 3 start, update the agent's reward memory
                buffer.rewards[buffer.rewards > reward_scales_left[1]] -= reward_scales_left[0] - reward_scales_right[0]

            # ---- training ----
            if global_step > step_start and global_step <= step_end and global_step % train_interval == 0:
                agent.learn(buffer) # training
                loss, loss_v, loss_q, loss_a, kld, logp_x = agent.record_loss(buffer)  # only for recording loss, no training, computed on latest experience
                loss_all.append(loss)
                loss_v_all.append(loss_v)
                loss_q_all.append(loss_q)
                loss_a_all.append(loss_a)
                kld_all.append(kld)
                logp_x_all.append(logp_x)
                rewards_all.append(np.sum(rs))

                if global_step - step_start < 50:
                    logging.info("model training one step takes {} s".format(time.time() - start_time))
                    start_time = time.time()

            if done or t == max_steps - 1:
                if r > 0:
                    if infos[t + 1, 0] < 0:
                        observations_rewarded_left.append(sp)
                    else:
                        observations_rewarded_right.append(sp)
                
                episode_return = np.sum(rs)
                episode_length = t + 1

                break
        # --------------------  Record Data to Buffer ----------------------
        dones[t] = True
        buffer.append_episode(observations, actions, rs, dones, episode_length)

        if verbose or episode % 100 == 0:
            logging.info(task_name + " seed {} -- episode {} (global step {}) : steps {}, total reward {}, reached position {}".format(
                        seed, episode, global_step, t, np.sum(rs), infos[t + 1, :2]))

        # ------------------------- testing after training ------------------------
        if max_all_steps - recording_steps < global_step <= max_all_steps:
            learned_behavior.append(infos)
        if episode % 100 == 0:
            if record_internal_states:
                agent.save_episode_data(details_savepath + task_name + "_habitization_{}_episode_{}.mat".format(seed, episode),
                                        info=infos[:episode_length + 1])

        # --------------  behavior right before and right after devaluation --------------------
        if  args.stage_3_start_step - 200 < global_step <= args.stage_3_start_step:
            if episode_record_before_devaluation >= 1:
                agent.save_episode_data(details_savepath + task_name + "_before_devaluation_{}_episode_{}.mat".format(
                    seed, episode_record_before_devaluation), info=infos[:episode_length + 1])
            episode_record_before_devaluation += 1
        elif args.stage_3_start_step < global_step <= args.stage_3_start_step + 200:
            if episode_record_after_devaluation >= 1:
                agent.save_episode_data(details_savepath + task_name + "_after_devaluation_{}_episode_{}.mat".format(
                    seed, episode_record_after_devaluation), info=infos[:episode_length + 1])
            episode_record_after_devaluation += 1
        
        episode += 1
    
    logging.info(" ^^^^^^^^  Finished, seed {}".format(seed))
    # save data
    performance_wrt_step_array = np.reshape(performance_wrt_step, [-1]).astype(np.float64)
    global_steps_array = np.reshape(global_steps, [-1]).astype(np.float64)
    steps_taken_wrt_step_array = np.reshape(steps_taken_wrt_step, [-1]).astype(np.float64)

    learned_behavior_np = np.stack(learned_behavior, axis=0)

    data = {"max_steps": max_steps,
            "learned_behavior": learned_behavior_np,
            "step_perf_eval": step_perf_eval,
            "beta_z": args.beta_z,
            "steps_taken_wrt_step": steps_taken_wrt_step_array,
            "performance_wrt_step": performance_wrt_step_array,
            "aif_iterations": np.array(aif_iterations),
            "sig_priors": np.array(sig_priors),
            "sig_posts": np.array(sig_posts),
            "loss_all": np.array(loss_all),
            "loss_v_all": np.array(loss_v_all),
            "loss_q_all": np.array(loss_q_all),
            "loss_a_all": np.array(loss_a_all),
            "kld_all": np.array(kld_all),
            "logp_x_all": np.array(logp_x_all),
            "rewards_all": np.array(rewards_all),
            "global_steps": global_steps_array}

    sio.savemat(savepath + task_name + "_habitization_{}.mat".format(seed), data, long_field_names=True)
    torch.save(agent, savepath + task_name + "_habitization_{}.model".format(seed))
    logging.info("@@@@@@@@ Saved, seed {}".format(seed))


if __name__ == "__main__":
    n_seed = int(args.n_seed)

    if n_seed == 1:
        run_trial(args.seed)
    elif n_seed > 1:
        program_list = [seed + args.seed for seed in range(n_seed)]

        # Create a process for each program
        processes = [multiprocessing.Process(target=run_trial, args=(program,)) for program in program_list]

        # Start each process
        for process in processes:
            process.start()

        # Join each process to wait for their completion
        for process in processes:
            process.join()
    else:
        raise ValueError('n_seed must be a positive integer')

