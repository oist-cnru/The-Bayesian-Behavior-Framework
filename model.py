import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dis
import numpy as np
import time
from base_modules import *
import scipy.io as sio
import colorednoise  # Use Pink Noise
import logging

logging.basicConfig(level=logging.INFO)

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 10
LOG_STD_MIN = -20
REG = 1e-3  # regularization of the actor

class BayesianBehaviorAgent(nn.Module):
    def __init__(self,
                 input_size,
                 action_size,
                 h_size=256,
                 z_size=4,
                 beta_z=0.1,
                 decision_precision_threshold=0.05,
                 max_iterations=16,
                 rl_config=None,
                 record_final_z_q=False,
                 device=None) -> None:
        super().__init__()

        self.input_size = input_size
        self.h_size = h_size
        self.z_size = z_size
        self.beta_z = beta_z
        self.action_size = action_size
        self.aif_max_iterations = max_iterations
        self.decision_precision_threshold = decision_precision_threshold

        self.record_final_z_q = record_final_z_q

        self.debug_mode = False 
        self.record_internal_states = True

        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ==============================  network definition ================================
        st_cnn_v, self.input_feature_size = make_cnn(self.input_size[0])  # for value function network
        st_cnn_h, self.input_feature_size = make_cnn(self.input_size[0])  # for h encoding network
        st_cnn_z, self.input_feature_size = make_cnn(self.input_size[0] * 2)  # for z_q encoding network

        self.f_x2phi_v = nn.Sequential(st_cnn_v, make_mlp(self.input_feature_size, [h_size], h_size, nn.ReLU, last_layer_linear=True))
        self.f_x2phi_h = nn.Sequential(st_cnn_h, make_mlp(self.input_feature_size, [h_size], h_size, nn.ReLU, last_layer_linear=True))
        self.f_x2phi_z = nn.Sequential(st_cnn_z, make_mlp(self.input_feature_size, [h_size], h_size, nn.ReLU, last_layer_linear=True))

        rnn_input_size = self.h_size
        self.rnn = nn.GRU(rnn_input_size, h_size, batch_first=True)

        self.h2muzp = make_mlp(h_size, [h_size, h_size], z_size, nn.ReLU, last_layer_linear=True)
        self.h2aspsigzp = make_mlp(h_size, [h_size, h_size], z_size, nn.ReLU, last_layer_linear=True)

        decoder = make_mlp(self.z_size, [h_size], h_size, nn.ReLU)
        self.pred_mux = nn.Sequential(decoder, UnsqueezeModule(-1), UnsqueezeModule(-1), make_dcnn(h_size, 6))

        self.hg2muz_q = make_mlp(self.input_feature_size, [h_size], z_size, nn.ReLU, last_layer_linear=True)
        self.hg2aspsigz_q = make_mlp(self.input_feature_size, [h_size], z_size, nn.ReLU, last_layer_linear=True)

        # ================================= RL part (SAC) ===================================

        self.algorithm = rl_config["algorithm"] if "algorithm" in rl_config else "sac"
        self.policy_layers = rl_config["policy_layers"] if ("policy_layers" in rl_config) else [h_size, h_size]
        self.value_layers = rl_config["value_layers"] if ("value_layers" in rl_config) else [h_size, h_size]
        self.motor_noise_beta = rl_config["motor_noise_beta"] if ("motor_noise_beta" in rl_config) else 1  # beta of colored motor noise. 0:Gaussian noise, 1:Pink noise 
        self.target_entropy = rl_config["target_entropy"] if ("target_entropy" in rl_config) else np.float32(
            - self.action_size)
        self.alg_type = 'actor_critic'
        self.lr_rl = rl_config["lr_rl"] if ("lr_rl" in rl_config) else 3e-4
        self.beta_h = rl_config["beta_h"] if ("beta_h" in rl_config) else 'auto'
        self.a_coef = rl_config["a_coef"] if ("a_coef" in rl_config) else 100000
        self.gamma = rl_config["gamma"] if ("gamma" in rl_config) else 0.9

        if isinstance(self.beta_h, str) and self.beta_h.startswith('auto'):
            # Default initial value of beta_h when learned
            init_value = 1.0
            if '_' in self.beta_h:
                init_value = float(self.beta_h.split('_')[1])
                assert init_value > 0., "The initial value of beta_h must be greater than 0"
            self.log_beta_h = torch.tensor(np.log(init_value).astype(np.float32), requires_grad=True)
        else:
            self.beta_h = float(self.beta_h)
        
        if isinstance(self.beta_h, str):
            self.optimizer_e = torch.optim.Adam([self.log_beta_h], lr=self.lr_rl)  # optimizer for beta_h

        value_input_size = self.h_size
        policy_input_size = self.z_size
        
        # policy network
        self.f_s2pi0 = ContinuousActionPolicyNetwork(policy_input_size, self.action_size, hidden_layers=self.policy_layers)
        # V network
        self.f_s2v = ContinuousActionVNetwork(value_input_size, hidden_layers=self.value_layers)
        # target V network
        self.f_s2v_tar = ContinuousActionVNetwork(value_input_size, hidden_layers=self.value_layers)
        # synchronize the target network with the main network
        self.f_s2v_tar.load_state_dict(self.f_s2v.state_dict())
        # Q network 1
        self.f_sa2q1 = ContinuousActionQNetwork(value_input_size, self.action_size, hidden_layers=self.value_layers) 
        # Q network 2
        self.f_sa2q2 = ContinuousActionQNetwork(value_input_size, self.action_size, hidden_layers=self.value_layers)
        
        # =================================== RL part end ==================================

        self.update_times = 0
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rl)
        self.to(self.device)
        self.init_states()

    def init_states(self, target_entropy=None):
        self.h_t = torch.zeros([1, self.h_size], dtype=torch.float32, device=self.device)
        self.muz_p_t = torch.zeros([1, self.z_size], dtype=torch.float32, device=self.device)
        self.sigz_p_t = torch.zeros([1, self.z_size], dtype=torch.float32, device=self.device)
        self.z_p_t = torch.zeros([1, self.z_size], dtype=torch.float32, device=self.device)
        self.z_q_t = torch.zeros([1, self.z_size], dtype=torch.float32, device=self.device)
        self.muz_q_t = torch.zeros([1, self.z_size], dtype=torch.float32, device=self.device)
        self.sigz_q_t = torch.zeros([1, self.z_size], dtype=torch.float32, device=self.device)
        self.z_s_t = torch.zeros([1, self.z_size], dtype=torch.float32, device=self.device)
        self.muz_s_t = torch.zeros([1, self.z_size], dtype=torch.float32, device=self.device)
        self.sigz_s_t = torch.zeros([1, self.z_size], dtype=torch.float32, device=self.device)

        self.mux_pred = None
        self.mux_pred_prior = None
        
        # colored noise generation
        self.colored_noise_episode = np.zeros([10000, self.action_size], dtype=np.float32)
        for i in range(self.colored_noise_episode.shape[-1]):
            self.colored_noise_episode[:, i] = colorednoise.powerlaw_psd_gaussian(self.motor_noise_beta, 10000, random_state=np.random.randint(0, 100000)).astype(np.float32)
        
        if target_entropy:
            self.target_entropy  = target_entropy

        self.env_step = 0
        self.init_recording_variables()
            
        self.aif_iterations = -1

    @staticmethod
    def compute_kl_divergence_gaussian(mu_1, sig_1, mu_2, sig_2):
        kl = torch.log(sig_2 / sig_1) + (sig_1 ** 2 + (mu_1 - mu_2) ** 2) / (2 * sig_2 ** 2) - 0.5
        return kl.sum(dim=-1)

    @staticmethod
    def reparameterize(muz, sigz):
        dist = torch.distributions.Normal(muz, sigz)
        z = dist.rsample()
        return z

    @staticmethod
    def compute_logp(mux, x_target):
        obs_dims = [-1, -2, -3]  # for vision
        logpx = - F.binary_cross_entropy_with_logits(mux, x_target, reduction='none').sum(dim=obs_dims)
        return logpx

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def infer_zq_es(self, z_p, muz_p, sigz_p, h, x, goal, early_stop: bool, mask=None, n_population=256, n_elite=32, goal_function=None):
        # using evolution strategy (CEM)
        n_generation = self.aif_max_iterations
        self.loss_curve = np.zeros([n_generation], dtype=np.float32)
        original_shape = h.shape[:-1]
        total_size = int(torch.prod(torch.tensor(original_shape)).item())

        z_s = None
        with torch.no_grad():

            mux_pred_prior = self.pred_mux(muz_p.reshape([-1, self.z_size])).reshape([*h.shape[:-1], 6, *self.input_size[1:]])
            
            x = x.reshape([-1, *x.shape[-3:]])
            goal = goal.reshape([-1, *goal.shape[-3:]])
            h = h.reshape([-1, *h.shape[-1:]])
            muz_p_expand = muz_p.reshape([-1, *muz_p.shape[-1:]])
            sigz_p_expand = sigz_p.reshape([-1, *sigz_p.shape[-1:]])

            if mask is not None:
                valid_index = mask.reshape([-1]) > 0
                x = x[valid_index]
                goal = goal[valid_index]
                h = h[valid_index]
                muz_p_expand = muz_p_expand[valid_index]
                sigz_p_expand = sigz_p_expand[valid_index]
            else:
                valid_index = 0

            t0 = time.time()

            x = x.repeat(n_population, 1, 1, 1)
            goal = goal.repeat(n_population, 1, 1, 1)
            
            h = h.repeat(n_population, 1)
            mask = mask.reshape([-1]).repeat(n_population, 1) if mask is not None else 1

            muz_p_expand = muz_p_expand.repeat(n_population, 1)
            sigz_p_expand = sigz_p_expand.repeat(n_population, 1)

            mean_m =  torch.zeros_like(muz_p_expand)
            std_m = 2 * torch.ones_like(mean_m)
            
            mean_s = torch.zeros_like(sigz_p_expand)
            std_s = 2 * torch.ones_like(mean_s)

            for generation in range(n_generation):

                if self.debug_mode:
                    t1 = time.time()
                
                muz_q = torch.normal(mean=mean_m, std=std_m)
                aspsigz_q = torch.normal(mean=mean_s, std=std_s)
                sigz_q = F.softplus(aspsigz_q) + 1e-3

                z_q = torch.normal(mean=muz_q, std=sigz_q)
                
                sigz_s = torch.sqrt(1 / ((1 / sigz_q) ** 2 + (1 / sigz_p_expand) ** 2))

                mux_pred = self.pred_mux(z_q)
                logpx = self.compute_logp(mux_pred[:, :3], x)
                if goal_function is None:
                    goal_achievement = self.compute_logp(mux_pred[:, -3:], goal)
                else:
                    goal_achievement = goal_function(mux_pred[:, -3:])
                logpx = logpx + goal_achievement

                
                kld_batch = torch.mean(self.compute_kl_divergence_gaussian(muz_q, sigz_q, muz_p_expand, sigz_p_expand).view([n_population, -1]), dim=-1)
                logpx_batch = torch.mean(logpx.view([n_population, -1]), dim=-1)

                free_energy_batch =  self.beta_z * kld_batch - logpx_batch
                
                idx_sort = torch.argsort(free_energy_batch, dim=0)
                idx = idx_sort[0]
                
                free_energy = torch.mean(free_energy_batch)

                std_m[:] = torch.sum(torch.abs(z_q[idx_sort[:n_elite]] - mean_m[:n_elite]), dim=0).expand(std_m.shape) / (n_elite - 1) 
                mean_m[:] = torch.mean(z_q[idx_sort[:n_elite]], dim=0).expand(mean_m.shape)

                std_s[:] = torch.sum(torch.abs(aspsigz_q[idx_sort[:n_elite]] - mean_s[:n_elite]), dim=0).expand(std_s.shape)  / (n_elite - 1)
                mean_s[:] = torch.mean(aspsigz_q[idx_sort[:n_elite]], dim=0).expand(mean_m.shape)

                self.loss_curve[generation] = free_energy.item()
                
                if self.debug_mode:
                    print('iter: {}, time use: {:.4f}s, free_energy: {:.3f}, kld: {:.3f}, goal_achievement: {:.3f}, sigz_p = {:.2f}, sigz_q = {:.2f}, mu_q = {:.2f}, z_q = {:2f}, early_stop: {}'.format(
                        generation + 1, time.time() - t1, free_energy.item(), kld_batch.mean().item(), goal_achievement.mean().item(), sigz_p_expand.mean().item(), F.softplus(mean_s).mean().item(), mean_m.mean().item(), z_q.mean().item(), early_stop))
                
                if (generation == n_generation - 1) or (early_stop and (sigz_s[idx_sort[0]].max().item() < self.decision_precision_threshold)):
                    
                    if z_s is None:  # compute z_s with early stopped z_q
                        z_q_ = z_q.view((n_population, -1, self.z_size,))[idx]
                        z_q_es = torch.zeros([total_size, self.z_size], dtype=torch.float32, device=self.device)
                        z_q_es[valid_index] = z_q_
                        z_q_es = z_q_es.view(original_shape + (-1,))

                        muz_q_ = muz_q.view((n_population, -1, self.z_size,))[idx]
                        muz_q_es = torch.zeros([total_size, self.z_size], dtype=torch.float32, device=self.device)
                        muz_q_es[valid_index] = muz_q_
                        muz_q_es = muz_q_es.view(original_shape + (-1,))

                        sigz_q_ = sigz_q.view((n_population, -1, self.z_size,))[idx]
                        sigz_q_es = torch.ones([total_size, self.z_size], dtype=torch.float32, device=self.device)
                        sigz_q_es[valid_index] = sigz_q_
                        sigz_q_es = sigz_q_es.view(original_shape + (-1,))

                        muz_s = (muz_q_es * (1 / sigz_q_es) ** 2 + muz_p * (1 / sigz_p) ** 2) / ((1 / sigz_q_es) ** 2 + (1 / sigz_p) ** 2)
                        sigz_s = torch.sqrt(1 / ((1 / sigz_q_es) ** 2 + (1 / sigz_p) ** 2))

                        z_s = (z_q * (1 / sigz_q) ** 2 + z_p * (1 / sigz_p) ** 2) / ((1 / sigz_q) ** 2 + (1 / sigz_p) ** 2)

                        self.aif_iterations = generation
                    
                    if not self.record_final_z_q:
                        break
                    
            # convert back to original shape with zero padding using mask

            z_q_ = z_q.view((n_population, -1, self.z_size,))[idx]
            z_q = torch.zeros([total_size, self.z_size], dtype=torch.float32, device=self.device)
            z_q[valid_index] = z_q_
            z_q = z_q.view(original_shape + (-1,))

            muz_q_ = muz_q.view((n_population, -1, self.z_size,))[idx]
            muz_q = torch.zeros([total_size, self.z_size], dtype=torch.float32, device=self.device)
            muz_q[valid_index] = muz_q_
            muz_q = muz_q.view(original_shape + (-1,))

            sigz_q_ = sigz_q.view((n_population, -1, self.z_size,))[idx]
            sigz_q = torch.ones([total_size, self.z_size], dtype=torch.float32, device=self.device)
            sigz_q[valid_index] = sigz_q_
            sigz_q = sigz_q.view(original_shape + (-1,))

            mux_pred_ = mux_pred.view((n_population, -1, 6) + self.input_size[1:])[idx]
            mux_pred = torch.zeros([total_size, *mux_pred_.shape[-3:]], dtype=torch.float32, device=self.device)
            mux_pred[valid_index] = mux_pred_
            mux_pred = mux_pred.reshape(original_shape + (-1,) + self.input_size[1:])

            if self.debug_mode:
                print(" ******************** time use {:.5f} *************************".format(time.time() - t0))
                print(" ** sigz_p = {:.2f}, sigz_q = {:.2f}, mu_p = {:.2f}, mu_q = {:.2f}, z_q = {:.2f} **".format(sigz_p.mean().item(), sigz_q.mean().item(), muz_p.mean().item(), muz_q.mean().item(), z_q.mean().item()))
                print(" ** loss curve: {}".format(np.array2string(self.loss_curve, precision=2, separator=', ', suppress_small=True)))
                print(" ***************************************************************")
            
            self.mux_pred = mux_pred.detach().cpu().numpy()
            self.mux_pred_prior = mux_pred_prior.detach().cpu().numpy()

        return z_q, muz_q, sigz_q, z_s, muz_s, sigz_s

    def step_with_env(self, env, x_t, x_g, behavior, action_return='normal', goal_function=None):
        
        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t).to(self.device).unsqueeze(0).to(torch.float32)
        if isinstance(x_g, np.ndarray):
            self.goal_obs = x_g
            x_g = torch.from_numpy(x_g).to(self.device).unsqueeze(0).to(torch.float32)
        
        phi_t = self.f_x2phi_h(x_t)
        
        self.h_t, _ = self.rnn(phi_t, self.h_t)

        self.muz_p_t = self.h2muzp(self.h_t)
        self.sigz_p_t = F.softplus(self.h2aspsigzp(self.h_t)) + 1e-3
        self.z_p_t = self.reparameterize(self.muz_p_t, self.sigz_p_t)
        
        if behavior == "habitual":
            mua, logsiga = self.f_s2pi0(self.z_p_t) 
        elif behavior == "goal-directed":
            self.z_q_t, self.muz_q_t, self.sigz_q_t, self.z_s_t, self.muz_s_t, self.sigz_s_t = self.infer_zq_es(self.z_p_t, self.muz_p_t, self.sigz_p_t, self.h_t, x_t, x_g, early_stop=False, goal_function=goal_function)
            mua, logsiga = self.f_s2pi0(self.z_q_t) 
        elif behavior == "synergized":
            self.z_q_t, self.muz_q_t, self.sigz_q_t, self.z_s_t, self.muz_s_t, self.sigz_s_t = self.infer_zq_es(self.z_p_t, self.muz_p_t, self.sigz_p_t, self.h_t, x_t, x_g, early_stop=True, goal_function=goal_function)
            mua, logsiga = self.f_s2pi0(self.z_s_t) 
        else:
            raise ValueError("behavior must be one of 'habitual', 'goal-directed', 'synergized'")

        siga = torch.exp(logsiga)

        if action_return == 'normal':
            self.noise = self.colored_noise_episode[self.env_step]
            u = mua + torch.from_numpy(self.noise).to(device=self.device) * siga
        elif action_return == 'mean':
            u = mua
        else:
            raise ValueError("action_return must be one of 'normal', 'mean'")
        
        self.env_step += 1

        if self.algorithm == "sac":
            self.a_t = torch.tanh(u)
        else:
            raise NotImplementedError
        
        # ---------- step in env ---------------
        a_t_numpy = self.a_t.detach().cpu().numpy()[0]
        results = env.step(a_t_numpy)

        if len(results) == 4:  # old Gym API
            x_curr, r_prev, done, info = results
        else:  # New Gym API
            x_curr, r_prev, terminated, truncated, info = results
            done = terminated or truncated
        
        # ---------------------------------------
        if self.record_internal_states:
            self.model_h_series.append(self.h_t.detach().cpu().numpy())
            self.model_z_p_series.append(self.z_p_t.detach().cpu().numpy())
            self.model_z_q_series.append(self.z_q_t.detach().cpu().numpy())
            self.model_mu_z_q_series.append(self.muz_q_t.detach().cpu().numpy())
            self.model_sig_z_q_series.append(self.sigz_q_t.detach().cpu().numpy())
            self.model_sig_z_p_series.append(self.sigz_p_t.detach().cpu().numpy())
            self.model_mu_z_p_series.append(self.muz_p_t.detach().cpu().numpy())
            self.aif_iterations_series.append(self.aif_iterations)
        
            if len(self.obs_series) == 0:
                self.obs_series.append(x_t.squeeze().detach().cpu().numpy())
            if self.mux_pred is not None:
                self.pred_visions.append(self.mux_pred)
            if self.mux_pred_prior is not None:
                self.pred_visions_prior.append(self.mux_pred_prior)

            self.obs_series.append(x_curr)
            self.a_series.append(a_t_numpy)
            self.r_series.append(r_prev)
            self.mua_series.append(mua.detach().cpu().numpy())
            self.siga_series.append(siga.detach().cpu().numpy())

        return x_curr, r_prev, done, info, a_t_numpy

    def learn(self, buffer):

        t0 = time.time()

        obs_batch, action_batch, reward_batch, done_batch, mask_batch, length_batch = buffer.sample_batch()

        minibatch_size = obs_batch.shape[0]
        max_stps = int(torch.max(length_batch))

        x_batch = obs_batch[:, :max_stps + 1] # observation data
        xg_batch = torch.zeros_like(x_batch[:, :-1])  # goal observation data
        for b in range(minibatch_size):
            dt = np.random.randint(1, length_batch[b].item() + 1)
            index = torch.arange(dt, dt + max_stps).to(self.device).clamp(1, length_batch[b])
            xg_batch[b] = x_batch[b, index]

        a_batch = action_batch[:, :max_stps] # action data
        r_batch = reward_batch[:, :max_stps] # reward data
        d_batch = done_batch[:, :max_stps].to(torch.float32)  # done (termination signal) data
        mask_batch = mask_batch[:, :max_stps].to(torch.float32) # mask (valid steps for variant lengths of sequences in RNN training) data
        maskp_batch = torch.cat([torch.ones_like(mask_batch[:, 0:1]), mask_batch], dim=1)  # mask data with one step shift forward

        h_beg = torch.zeros([minibatch_size, self.h_size], dtype=torch.float32, device=self.device) #  initial RNN states

        # ==================================== Predictive Coding ====================================        
        phi_batch = self.f_x2phi_h(x_batch.reshape([minibatch_size * (max_stps + 1), *self.input_size]))
        phi_batch = phi_batch.reshape([minibatch_size, max_stps + 1, -1])
        h_batch, _ = self.rnn(phi_batch, torch.unsqueeze(h_beg, 0))   # h_batch shape: [minibatch_size, max_stps + 1, h_size]

        muz_p_batch = self.h2muzp(h_batch[:, :-1]) # shape: [minibatch_size, max_stps, z_size]
        aspsigz_p_batch = self.h2aspsigzp(h_batch[:, :-1])
        sigz_p_batch = F.softplus(aspsigz_p_batch) + 1e-3

        phi_zq = self.f_x2phi_z(torch.cat([x_batch[:, :-1], xg_batch], dim=-3).reshape(
            [minibatch_size * max_stps, 2 * self.input_size[0], *self.input_size[1:]])).reshape([minibatch_size, max_stps, -1])
        muz_q_batch = self.hg2muz_q(phi_zq)
        aspsigz_q_batch = self.hg2aspsigz_q(phi_zq)
        sigz_q_batch = F.softplus(aspsigz_q_batch) + 1e-3
        z_q_batch = self.reparameterize(muz_q_batch, sigz_q_batch)

        # --------- free energy loss function  ------------
        kld_batch = self.compute_kl_divergence_gaussian(muz_q_batch, sigz_q_batch, muz_p_batch, sigz_p_batch).reshape([-1])
        mux_pred = self.pred_mux(z_q_batch.reshape(-1, self.z_size))
        logp_x_batch = self.compute_logp(mux_pred[:, :3], x_batch[:, :-1].reshape([-1, *self.input_size]))
        logp_x_batch += self.compute_logp(mux_pred[:, -3:], xg_batch.reshape([-1, *self.input_size]))
        loss_batch = (self.beta_z * kld_batch - logp_x_batch) * mask_batch.reshape([-1]) # free energy loss
        loss = loss_batch.mean()

        kld = (kld_batch * mask_batch.reshape([-1])).mean()
        logp_x = (logp_x_batch * mask_batch.reshape([-1])).mean()

        # ==================================== RL (SAC) =======================================
        if isinstance(self.beta_h, str):
            beta_h = torch.exp(self.log_beta_h).detach()
        else:
            beta_h = self.beta_h
        
        mask_batch = torch.unsqueeze(mask_batch, -1)
        maskp_batch = torch.unsqueeze(maskp_batch, -1)
        d_batch = torch.unsqueeze(d_batch, -1)
        r_batch = torch.unsqueeze(r_batch, -1)

        ha_tensor = z_q_batch

        with torch.no_grad():
            mua_tensor, logsia_tensor = self.f_s2pi0(ha_tensor)
            siga_tensor = torch.exp(logsia_tensor.clamp(LOG_STD_MIN, LOG_STD_MAX))

            sampled_u = self.reparameterize(mua_tensor.detach(), siga_tensor.detach()).detach()
            sampled_a = torch.tanh(sampled_u)

            log_pi_exp = torch.sum(- (mua_tensor.detach() - sampled_u.detach()).pow(2)
                                    / (siga_tensor.detach().pow(2)) / 2
                                    - torch.log(siga_tensor.detach() * torch.tensor(2.5066)),
                                    dim=-1, keepdim=True)
            log_pi_exp = log_pi_exp - torch.sum(torch.log(1.0 - sampled_a.pow(2) + EPS), dim=-1,
                                                keepdim=True)
            log_pi_exp = (log_pi_exp * mask_batch).detach().mean() / mask_batch.mean()

        # ------ loss_v ---------------
        hv_tensor = self.f_x2phi_v(x_batch.reshape([minibatch_size * (max_stps + 1), *self.input_size])).reshape(
            [minibatch_size, max_stps + 1, -1])
        
        v_tensor = self.f_s2v(hv_tensor[:, :-1])
        vp_tensor = self.f_s2v_tar(hv_tensor[:, 1:]).detach()

        q_tensor_1 = self.f_sa2q1(hv_tensor[:, :-1], a_batch) 
        q_tensor_2 = self.f_sa2q2(hv_tensor[:, :-1], a_batch)

        sampled_q = torch.min(self.f_sa2q1(hv_tensor[:, :-1], sampled_a).detach(),
                                self.f_sa2q2(hv_tensor[:, :-1], sampled_a).detach())

        q_exp = sampled_q

        v_tar = (q_exp - beta_h * log_pi_exp).detach() 
            
        loss_v = 0.5 * self.mse_loss(v_tensor * mask_batch, v_tar * mask_batch)

        loss_v = torch.mean(loss_v)

        q_tar_1 = (r_batch + (1 - d_batch) * self.gamma * vp_tensor.detach())
        q_tar_2 = (r_batch + (1 - d_batch) * self.gamma * vp_tensor.detach())
        
        loss_q = 0.5 * self.mse_loss(q_tensor_1 * mask_batch, q_tar_1 * mask_batch) + \
                    0.5 * self.mse_loss(q_tensor_2 * mask_batch, q_tar_2 * mask_batch)

        loss_q = torch.mean(loss_q)

        loss_critic = loss_q + loss_v

        loss = loss + loss_critic

        # ----- loss_a --------

        # Reparameterize a
        mua_tensor, logsia_tensor = self.f_s2pi0(ha_tensor)
        siga_tensor = torch.exp(logsia_tensor.clamp(LOG_STD_MIN, LOG_STD_MAX))

        mu_prob = dis.Normal(mua_tensor, siga_tensor)
        sampled_u = mu_prob.rsample()
        sampled_a = torch.tanh(sampled_u)

        log_pi = torch.sum(mu_prob.log_prob(sampled_u).clamp(LOG_STD_MIN, LOG_STD_MAX), dim=-1,
                                keepdim=True) - torch.sum(
            torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)
        
        loss_a = torch.mean(torch.mean(
            beta_h * log_pi * mask_batch - torch.min(
                self.f_sa2q1(hv_tensor.detach()[:, :-1], sampled_a),
                self.f_sa2q2(hv_tensor.detach()[:, :-1], sampled_a)
            ) * mask_batch + torch.min(
                self.f_sa2q1(hv_tensor.detach()[:, :-1], sampled_a.detach()),
                self.f_sa2q2(hv_tensor.detach()[:, :-1], sampled_a.detach())
            ) * mask_batch, dim=[1, 2]))

        loss_a = loss_a + REG / 2 * (
            torch.mean(torch.mean((siga_tensor * mask_batch.repeat_interleave(
                siga_tensor.size()[-1], dim=-1)).pow(2), dim=[1, 2]))
            + torch.mean(torch.mean((mua_tensor * mask_batch.repeat_interleave(
                mua_tensor.size()[-1], dim=-1)).pow(2), dim=[1, 2])))

        loss = loss + self.a_coef * loss_a

        self.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update entropy coefficient if required
        if isinstance(beta_h, torch.Tensor):
            self.optimizer_e.zero_grad()
            loss_e = torch.mean(- self.log_beta_h * (log_pi_exp + self.target_entropy).detach())
            loss_e.backward()
            self.optimizer_e.step()
        
        # update target network   
        self.soft_update(self.f_s2v_tar, self.f_s2v, 0.005)

        self.update_times += 1

        t_end = time.time()

        if self.update_times < 10:
            logging.info('update once time: {} s'.format(t_end - t0))

        return loss.item(), loss_v.item(), loss_q.item(), loss_a.item(), kld.item(), logp_x.item()

    def record_loss(self, buffer):

        with torch.no_grad():

            obs_batch, action_batch, reward_batch, done_batch, mask_batch, length_batch = buffer.sample_latest_experience()

            minibatch_size = obs_batch.shape[0]
            max_stps = int(torch.max(length_batch))

            x_batch = obs_batch[:, :max_stps + 1] # observation data
            xg_batch = torch.zeros_like(x_batch[:, :-1])  # goal observation data
            for b in range(minibatch_size):
                dt = np.random.randint(1, length_batch[b].item() + 1)
                index = torch.arange(dt, dt + max_stps).to(self.device).clamp(1, length_batch[b])
                xg_batch[b] = x_batch[b, index]

            a_batch = action_batch[:, :max_stps] # action data
            r_batch = reward_batch[:, :max_stps] # reward data
            d_batch = done_batch[:, :max_stps].to(torch.float32)  # done (termination signal) data
            mask_batch = mask_batch[:, :max_stps].to(torch.float32) # mask (valid steps for variant lengths of sequences in RNN training) data
            maskp_batch = torch.cat([torch.ones_like(mask_batch[:, 0:1]), mask_batch], dim=1)  # mask data with one step shift forward

            mask_batch = mask_batch / mask_batch.mean()
            maskp_batch = maskp_batch / maskp_batch.mean()

            h_beg = torch.zeros([minibatch_size, self.h_size], dtype=torch.float32, device=self.device) #  initial RNN states

            # ==================================== Predictive Coding ====================================        
            phi_batch = self.f_x2phi_h(x_batch.reshape([minibatch_size * (max_stps + 1), *self.input_size]))
            phi_batch = phi_batch.reshape([minibatch_size, max_stps + 1, -1])
            h_batch, _ = self.rnn(phi_batch, torch.unsqueeze(h_beg, 0))   # h_batch shape: [minibatch_size, max_stps + 1, h_size]

            muz_p_batch = self.h2muzp(h_batch[:, :-1]) # shape: [minibatch_size, max_stps, z_size]
            aspsigz_p_batch = self.h2aspsigzp(h_batch[:, :-1])
            sigz_p_batch = F.softplus(aspsigz_p_batch) + 1e-3

            phi_zq = self.f_x2phi_z(torch.cat([x_batch[:, :-1], xg_batch], dim=-3).reshape(
                [minibatch_size * max_stps, 2 * self.input_size[0], *self.input_size[1:]])).reshape([minibatch_size, max_stps, -1])
            muz_q_batch = self.hg2muz_q(phi_zq)
            aspsigz_q_batch = self.hg2aspsigz_q(phi_zq)
            sigz_q_batch = F.softplus(aspsigz_q_batch) + 1e-3
            z_q_batch = self.reparameterize(muz_q_batch, sigz_q_batch)

            # --------- free energy loss function  ------------
            kld_batch = self.compute_kl_divergence_gaussian(muz_q_batch, sigz_q_batch, muz_p_batch, sigz_p_batch).reshape([-1])
            mux_pred = self.pred_mux(z_q_batch.reshape(-1, self.z_size))
            logp_x_batch = self.compute_logp(mux_pred[:, :3], x_batch[:, :-1].reshape([-1, *self.input_size]))
            logp_x_batch += self.compute_logp(mux_pred[:, -3:], xg_batch.reshape([-1, *self.input_size]))
            loss_batch = (self.beta_z * kld_batch - logp_x_batch) * mask_batch.reshape([-1]) # free energy loss
            loss = loss_batch.mean()

            kld = (kld_batch * mask_batch.reshape([-1])).mean()
            logp_x = (logp_x_batch * mask_batch.reshape([-1])).mean()

            # ==================================== RL (SAC) =======================================
            if isinstance(self.beta_h, str):
                beta_h = torch.exp(self.log_beta_h).detach()
            else:
                beta_h = self.beta_h
            
            mask_batch = torch.unsqueeze(mask_batch, -1)
            maskp_batch = torch.unsqueeze(maskp_batch, -1)
            d_batch = torch.unsqueeze(d_batch, -1)
            r_batch = torch.unsqueeze(r_batch, -1)

            ha_tensor = z_q_batch

            with torch.no_grad():
                mua_tensor, logsia_tensor = self.f_s2pi0(ha_tensor)
                siga_tensor = torch.exp(logsia_tensor.clamp(LOG_STD_MIN, LOG_STD_MAX))

                sampled_u = self.reparameterize(mua_tensor.detach(), siga_tensor.detach()).detach()
                sampled_a = torch.tanh(sampled_u)

                log_pi_exp = torch.sum(- (mua_tensor.detach() - sampled_u.detach()).pow(2)
                                        / (siga_tensor.detach().pow(2)) / 2
                                        - torch.log(siga_tensor.detach() * torch.tensor(2.5066)),
                                        dim=-1, keepdim=True)
                log_pi_exp = log_pi_exp - torch.sum(torch.log(1.0 - sampled_a.pow(2) + EPS), dim=-1,
                                                    keepdim=True)
                log_pi_exp = (log_pi_exp * mask_batch).detach().mean() / mask_batch.mean()

            # ------ loss_v ---------------
            hv_tensor = self.f_x2phi_v(x_batch.reshape([minibatch_size * (max_stps + 1), *self.input_size])).reshape(
                [minibatch_size, max_stps + 1, -1])
            
            v_tensor = self.f_s2v(hv_tensor[:, :-1])
            vp_tensor = self.f_s2v_tar(hv_tensor[:, 1:]).detach()

            q_tensor_1 = self.f_sa2q1(hv_tensor[:, :-1], a_batch) 
            q_tensor_2 = self.f_sa2q2(hv_tensor[:, :-1], a_batch)

            sampled_q = torch.min(self.f_sa2q1(hv_tensor[:, :-1], sampled_a).detach(),
                                    self.f_sa2q2(hv_tensor[:, :-1], sampled_a).detach())

            q_exp = sampled_q

            v_tar = (q_exp - beta_h * log_pi_exp).detach() 
                
            loss_v = 0.5 * self.mse_loss(v_tensor * mask_batch, v_tar * mask_batch)

            loss_v = torch.mean(loss_v)

            q_tar_1 = (r_batch + (1 - d_batch) * self.gamma * vp_tensor.detach())
            q_tar_2 = (r_batch + (1 - d_batch) * self.gamma * vp_tensor.detach())
            
            loss_q = 0.5 * self.mse_loss(q_tensor_1 * mask_batch, q_tar_1 * mask_batch) + \
                        0.5 * self.mse_loss(q_tensor_2 * mask_batch, q_tar_2 * mask_batch)

            loss_q = torch.mean(loss_q)

            # ----- loss_a --------

            # Reparameterize a
            mua_tensor, logsia_tensor = self.f_s2pi0(ha_tensor)
            siga_tensor = torch.exp(logsia_tensor.clamp(LOG_STD_MIN, LOG_STD_MAX))

            mu_prob = dis.Normal(mua_tensor, siga_tensor)
            sampled_u = mu_prob.rsample()
            sampled_a = torch.tanh(sampled_u)

            log_pi = torch.sum(mu_prob.log_prob(sampled_u).clamp(LOG_STD_MIN, LOG_STD_MAX), dim=-1,
                                    keepdim=True) - torch.sum(
                torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)
            
            loss_a = torch.mean(torch.mean(
                beta_h * log_pi * mask_batch - torch.min(
                    self.f_sa2q1(hv_tensor.detach()[:, :-1], sampled_a),
                    self.f_sa2q2(hv_tensor.detach()[:, :-1], sampled_a)
                ) * mask_batch + torch.min(
                    self.f_sa2q1(hv_tensor.detach()[:, :-1], sampled_a.detach()),
                    self.f_sa2q2(hv_tensor.detach()[:, :-1], sampled_a.detach())
                ) * mask_batch, dim=[1, 2]))

            loss_a = loss_a + REG / 2 * (
                torch.mean(torch.mean((siga_tensor * mask_batch.repeat_interleave(
                    siga_tensor.size()[-1], dim=-1)).pow(2), dim=[1, 2]))
                + torch.mean(torch.mean((mua_tensor * mask_batch.repeat_interleave(
                    mua_tensor.size()[-1], dim=-1)).pow(2), dim=[1, 2])))

        return loss.item(), loss_v.item(), loss_q.item(), loss_a.item(), kld.item(), logp_x.item()

    def init_recording_variables(self):
        self.model_h_series = []
        self.model_mu_z_q_series = []
        self.model_sig_z_q_series = []
        self.model_mu_z_p_series = []
        self.model_sig_z_p_series = []
        self.model_z_q_series = []
        self.model_z_p_series = []
        self.model_mu_s_q_series = []
        self.model_sig_s_q_series = []
        self.model_s_q_series = []
        self.aif_iterations_series = []

        self.obs_series = []
        self.r_series = []
        self.a_series = []
        self.mua_series = []
        self.siga_series = []
        self.pred_visions = []
        self.pred_visions_prior = []

        # --------------- For Active Inference --------------
        self.step_weighting_series = []
        self.pred_trajectories = []
        self.z_aif_batch_series = []

    def save_episode_data(self, filename=None, info=None, goal_pos=None):
        data = {}

        data['model_h'] = np.array(self.model_h_series).squeeze()
        data['model_z_q'] = np.array(self.model_z_q_series).squeeze()
        data['model_z_p'] = np.array(self.model_z_p_series).squeeze()
        data['model_sig_z_q'] = np.array(self.model_sig_z_q_series).squeeze()
        data['model_mu_z_q'] = np.array(self.model_mu_z_q_series).squeeze()
        data['model_sig_z_p'] = np.array(self.model_sig_z_p_series).squeeze()
        data['model_mu_z_p'] = np.array(self.model_mu_z_p_series).squeeze()

        data['obs'] = (np.array(self.obs_series).squeeze() * 255).astype(np.uint8)
        data['reward'] = np.array(self.r_series).squeeze()
        data['action'] = np.array(self.a_series).squeeze()
        data['mua'] = np.array(self.mua_series).squeeze()
        data['siga'] = np.array(self.siga_series).squeeze()

        data['aif_iterations'] = np.array(self.aif_iterations_series).squeeze()
        data['pred_visions'] = (1 / (1 + 1 / np.exp(np.array(self.pred_visions).squeeze())) * 255).astype(np.uint8)  # sigmoid
        data['pred_visions_prior'] = (1 / (1 + 1 / np.exp(np.array(self.pred_visions_prior).squeeze())) * 255).astype(np.uint8)

        try:
            data['goal_obs'] = np.array(self.goal_obs)
        except:
            pass

        if goal_pos is not None:
            data['goal_pos'] = goal_pos

        if info is not None:
            data['info'] = info

        if filename is None:
            return data
        else:
            return sio.savemat(filename, data)
