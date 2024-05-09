import math
import torch
import abc
from tqdm import tqdm
import torchvision.utils as tvutils
import os
from scipy import integrate


class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def ode_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################

    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)

    def reverse_sde_step_mean(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t)

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)

    def reverse_ode_step(self, x, score, t):
        return x - self.ode_reverse_drift(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

        return x


#############################################################################


class IRSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self, max_sigma, T=100, sample_T=-1, schedule='cosine', eps=0.01,  device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self.sample_T = self.T if sample_T < 0 else sample_T
        self.sample_scale = self.T / self.sample_T
        self._initialize(self.max_sigma, self.sample_T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            # print('constant schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            # print('linear schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            """
            # print('cosine schedule')
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
            
        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None

    #####################################

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    #####################################

    def mu_bar(self, x0, t):
        return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, t):
        return self.thetas[t] * (self.mu - x) * self.dt

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - self.sigmas[t]**2 * score) * self.dt

    def ode_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - 0.5 * self.sigmas[t]**2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def score_fn_(self, x, t, scale=1.0):
        # need to pre-set mu and score_model
        x0 = self.model(x, self.mu, t * scale)
        score = -(x - self.mu_bar(x0, t)) / self.sigma_bar(t)**2
        return score

    def score_fn(self, x, t, scale=1.0, **kwargs):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu, t * scale, **kwargs)
        return self.get_score_from_noise(noise, t)

    def noise_fn(self, x, t, scale=1.0, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t * scale, **kwargs)

    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t)**2

    def get_init_state_from_noise(self, xt, noise, t):
        A = torch.exp(self.thetas_cumsum[t] * self.dt)
        return (xt - self.mu - self.sigma_bar(t) * noise) * A + self.mu

    # forward process to get x(T) from x(0)
    # 定义前向模拟过程，从初始状态 x0 演化到最终时间 T 的状态
    def forward(self, x0, T=-1, save_dir='forward_state'):
        # 如果传入的 T 为负数，则使用类实例的 T 属性作为模拟的总时间
        T = self.T if T < 0 else T
        # 从输入的初始状态 x0 创建一个副本，用于在模拟过程中更新状态
        x = x0.clone()

        # 使用 tqdm 库创建一个进度条，显示模拟进度
        for t in tqdm(range(1, T + 1)):
            # 在每个时间步执行前向模拟步骤，更新状态 x
            x = self.forward_step(x, t)

            # 如果保存目录不存在，则创建它，如果已存在则不抛出错误
            os.makedirs(save_dir, exist_ok=True)

            # 将当前状态 x 沿第一个维度分成两部分，可能是为了分别保存不同的状态分量
            x_L, x_R = x.chunk(2, dim=1)

            # 将两部分状态沿维度0拼接，然后调用 tvutils 库的 save_image 函数保存为图像文件
            # 文件名包含时间步信息，用于后续的可视化和分析
            tvutils.save_image(torch.cat([x_L, x_R], dim=0).data, f'{save_dir}/state_{t}.png', normalize=False)

        # 模拟结束后，返回最终的状态 x
        return x

    # 定义逆向SDE过程，从最终状态 xt 逆向模拟回到初始状态
    def reverse_sde(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        # 如果传入的 T 为负数，则使用类实例的 sample_T 属性作为逆向模拟的总时间
        T = self.sample_T if T < 0 else T

        # 从输入的最终状态 xt 创建一个副本，用于在逆向模拟过程中更新状态
        x = xt.clone()

        # 使用 tqdm 库创建一个进度条，显示逆向模拟进度
        for t in tqdm(reversed(range(1, T + 1))):
            # 调用 score_fn 方法计算给定状态和时间的评分函数（也称为概率密度函数的梯度）
            score = self.score_fn(x, t, self.sample_scale, **kwargs)
            # 执行逆向SDE步骤，使用评分函数更新状态 x
            x = self.reverse_sde_step(x, score, t)
            # x = self.reverse_sde_step_mean(x, score, t)  # 这行代码被注释掉了，可能表示一个备用的逆向模拟步骤

            # 如果 save_states 为 True，则保存逆向模拟过程中的状态
            if save_states:
                # 计算保存状态的间隔，这里假设只保存100个图像
                interval = self.T // 100
                # 如果当前时间步是保存间隔的整数倍，则保存状态
                if t % interval == 0:
                    # 计算当前状态的索引
                    idx = t // interval
                    # 如果保存目录不存在，则创建它
                    os.makedirs(save_dir, exist_ok=True)
                    # 将当前状态 x 沿第一个维度分成两部分
                    x_L, x_R = x.chunk(2, dim=1)
                    # 将两部分状态沿维度3拼接，并保存为图像文件
                    tvutils.save_image(torch.cat([x_L, x_R], dim=3).data, f'{save_dir}/state_{idx}.png',
                                       normalize=False)

        # 逆向模拟结束后，返回最终的状态 x
        return x

    def reverse_ode(self, xt, T=-1, save_states=False, save_dir='ode_state'):
        T = self.sample_T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
        # for t in tqdm(reversed(range(1, 81))):
            score = self.score_fn(x, t, self.sample_scale)
            x = self.reverse_ode_step(x, score, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    x_L, x_R = x.chunk(2, dim=1)
                    tvutils.save_image(torch.cat([x_L, x_R], dim=3).data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    # sample ode using Black-box ODE solver (not used)
    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3,):
        shape = xt.shape

        def to_flattened_numpy(x):
          """Flatten a torch tensor `x` and convert it to numpy."""
          return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
          """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
          return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                     rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    ################################################################

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)

    # sample states for training
    def generate_random_states(self, x0, mu, timesteps=None, T_start=1, T_end=-1):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        if timesteps is None:
            batch = x0.shape[0]
            T_end = self.T + 1 if T_end <= 1 else T_end + 1
            timesteps = torch.randint(T_start, T_end, (batch, 1, 1, 1)).long()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32)

    def noise_state(self, tensor):
        return tensor + torch.randn_like(tensor) * self.max_sigma
        # return tensor + torch.randn_like(tensor) * self.sigma_bar(91).cpu()


