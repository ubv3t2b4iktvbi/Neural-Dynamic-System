# Neural Dynamic System

这个 README 只描述当前仓库里实际实现的模型与训练流程，不再沿用旧版 `q,m` 叙事。

当前代码里的真实 latent state 是

$$
z_t = (q_t, h_t),
$$

其中：

- `q_t` 是慢变量 / 慢坐标。
- `h_t` 是隐藏快变量，也是当前代码里承担 closure / memory embedding 作用的状态。
- CLI 和部分导出字段里仍保留 `--m_dim`、`memory_rate_mean`、`m_supervised_weight` 这类名字作为兼容别名，但它们在实现上都等价于当前的 `h`。

一句话概括当前实现：

> 这是一个“窗口编码 + 慢变量 `q` + 条件化隐藏状态空间 `h` + 结构化解码器 + 多步一致性训练”的模型；它已经具备把隐藏子系统改写成 SSD / selective scan 风格内核的接口条件，但整个系统还不是完整的 Mamba kernel。

## 当前实现包含什么

当前仓库把下面几件事放在了同一个可训练模型里：

1. 从窗口序列里提取慢坐标 `q`，并用 VAMP 风格目标约束其时间相关结构。
2. 用一个 `q` 条件化的仿射隐藏状态空间来表示快变量 / closure 状态 `h`。
3. 用结构化解码器

   $$
   \hat x_t = g(q_t) + B(q_t) h_t
   $$

   把“慢流形主项”和“离流形修正项”分开。
4. 用多步 rollout、一致性、半群、RG-like 约束把 latent dynamics 训练成真正可推进的动力系统，而不只是一个自编码器。

## 仓库结构

```text
neural_dynamic_system/
  __init__.py
  cli.py
  config.py
  data.py
  model.py
  synthetic.py
  training.py
scripts/
  run_neural_dynamic_system.py
README.md
```

核心文件：

- `neural_dynamic_system/model.py`：编码器、`q/h` latent 表示、动力学、解码器、coarse graining。
- `neural_dynamic_system/training.py`：损失函数、课程训练、checkpoint 选择。
- `neural_dynamic_system/data.py`：窗口采样、标准化、train/val 拆分。
- `neural_dynamic_system/synthetic.py`：`toy`、`no_gap_toy`、`alanine_like` 合成数据。
- `neural_dynamic_system/cli.py`：命令行入口、产物导出、线性 probe。

## 安装

仓库目前没有 `pyproject.toml`，运行依赖需要手动安装：

```bash
pip install torch numpy pandas
```

当前实现没有使用 `torchdiffeq`；旧 README 里那段可选 ODE solver 说明已经不适用了。

## 快速开始

运行一个小的 `toy` 合成实验：

```bash
python -m neural_dynamic_system.cli \
  --synthetic_kind toy \
  --num_episodes 4 \
  --steps 4096 \
  --obs_dim 8 \
  --window 32 \
  --q_dim 2 \
  --h_dim 2 \
  --latent_scheme soft_spectrum \
  --modal_dim 8 \
  --out_dir runs/neural_dynamic_system/toy_demo
```

等价脚本入口：

```bash
python scripts/run_neural_dynamic_system.py \
  --synthetic_kind toy \
  --out_dir runs/neural_dynamic_system/toy_demo
```

运行 `alanine_like` 基准：

```bash
python -m neural_dynamic_system.cli \
  --synthetic_kind alanine_like \
  --num_episodes 4 \
  --steps 8192 \
  --obs_dim 16 \
  --window 64 \
  --q_dim 2 \
  --h_dim 4 \
  --latent_scheme soft_spectrum \
  --modal_dim 12 \
  --curriculum_preset alanine_bootstrap \
  --out_dir runs/neural_dynamic_system/alanine_like
```

训练自定义轨迹：

```bash
python -m neural_dynamic_system.cli \
  --data_path path/to/trajectory.npy \
  --window 64 \
  --q_dim 3 \
  --h_dim 3 \
  --horizons 1 2 4 8 \
  --out_dir runs/neural_dynamic_system/custom_run
```

## 数据格式与采样行为

`data.py` 里支持的输入轨迹格式是：

- `1D` 数组：自动视为单变量时间序列，reshape 成 `[T, 1]`
- `2D` 数组：`[T, d]`
- `3D` 数组：`[N, T, d]`
- `Sequence[np.ndarray]`：每个 episode 一条轨迹，允许不同长度
- 文件格式：`.npy`、`.npz`、`.csv`、`.txt`

标签 `labels` 需要和轨迹保持相同的 episode 结构与步数。

给定窗口长度 `L` 和 horizon 集合 `\mathcal H`，`ArrayTrajectoryDataset` 的一个训练样本是：

$$
W_t = [x_{t-L+1}, \dots, x_t] \in \mathbb{R}^{L \times d},
$$

$$
x_t = W_t[-1],
$$

$$
\{x_{t+h}\}_{h \in \mathcal H},
$$

以及对应的未来窗口

$$
W_{t+h} = [x_{t-L+1+h}, \dots, x_{t+h}].
$$

代码里的拆分和标准化行为也很具体：

1. 先按 episode 做 train/val 切分。
2. 观测标准化只用 train 段统计量：

   $$
   x \mapsto \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + 10^{-6}}.
   $$

3. validation 的起点不是简单的 `split_index`，而是

   $$
   \text{val\_start} = \max(\text{split\_index} - L - \max \mathcal H,\ 0),
   $$

   这样验证集也能构造完整窗口和未来目标。
4. 如果提供标签，标签也按同样的 episode 切分；默认还会做基于 train split 的标准化。
5. 当 `--q_supervision_mode angular` 且 `q` 监督权重大于零时，对应 `q` 标签维度不会做标准化。

## 输出文件

每次运行会在 `--out_dir` 下写出：

- `model.pt`：模型参数、`model_config`、`train_config`、`loss_config`、轨迹标准化统计量、可选 `label_stats`
- `history.csv`：每个 epoch 的 train/val 指标
- `config.json`：数据源、配置、标准化信息、标签名、synthetic metadata
- `summary.json`：最佳验证指标、phase 选择信息、soft spectrum 摘要
- `trajectory_preview.csv`：每个 episode 前 512 步观测
- `synthetic_hidden_state.csv`：仅合成数据时输出
- `synthetic_labels.csv`：仅合成数据且存在监督标签时输出
- `synthetic_probe_labels.csv`：仅合成数据且存在 probe 标签时输出
- `label_probe.json`：存在显式标签训练/评估时输出
- `synthetic_hidden_probe.json`：未提供标签但使用合成 probe 标签时输出
- `label_component_corrs.csv` 或 `synthetic_component_corrs.csv`：最佳 latent-component 相关性

有两个实现细节值得知道：

1. 训练结束后，代码会加载“最高 phase 中验证集最好的 checkpoint”，不一定是全程全局最小 loss 的那个；`summary.json` 里会同时记录 `overall_best_val_loss`。
2. `eval_batch_size` 只用于导出后的线性 probe，不参与主训练。

## 数学模型

### 1. 窗口编码

输入是标准化后的窗口

$$
W_t = [x_{t-L+1}, \dots, x_t] \in \mathbb{R}^{L \times d}.
$$

编码器输出两路 hidden summary：

$$
(u_t^{(q)}, u_t^{(h)}) = E_\theta(W_t).
$$

当前有两种编码器：

#### `temporal_conv`

`TemporalMultiscaleEncoder` 先把输入转成 `[B, d, L]`，再通过：

- 一个 stem `Conv1d`
- 多层残差时序卷积块
- 层间 stride-2 下采样
- 一个 bottleneck 残差块

得到多尺度 summary。实现上的分工是：

- `q` 分支只看 bottleneck 的时间平均；
- `h` 分支看 bottleneck 加上所有尺度 summary 的拼接。

所以可以抽象地写成

$$
u_t^{(q)} = P_q(\bar h_t^{\text{bottleneck}}),
$$

$$
u_t^{(h)} = P_h([\bar h_t^{(1)}, \dots, \bar h_t^{(J)}, \bar h_t^{\text{bottleneck}}]).
$$

#### `mlp`

窗口直接 flatten 成 `Ld` 维向量，再经过 MLP：

$$
u_t^{(q)} = u_t^{(h)} = \operatorname{MLP}(\operatorname{vec}(W_t)).
$$

### 2. Latent 分配：`hard_split` 与 `soft_spectrum`

#### `hard_split`

直接把编码器输出送进两条头：

$$
q_t^{\text{src}} = u_t^{(q)}, \qquad h_t^{\text{src}} = u_t^{(h)}.
$$

#### `soft_spectrum`

先学习共享 modal coordinates：

$$
a_t = \tanh\!\Big(f_{\text{modal}}([u_t^{(q)}, u_t^{(h)}])\Big) \in \mathbb{R}^{K}.
$$

每个 mode 还带一个全局可训练正 decay rate：

$$
\lambda_k = \lambda_{\min}^{\text{slow}} + \operatorname{softplus}(\theta_k).
$$

对 log-rate 做居中和温度缩放：

$$
c_k = \frac{\log \lambda_k - \frac{1}{K}\sum_{j=1}^{K}\log \lambda_j}{\tau},
$$

其中 `\tau = modal_temperature`。

然后构造两组 soft partition 权重：

$$
w_k^{(q)} = \frac{e^{-c_k}}{\sum_j e^{-c_j}},
\qquad
w_k^{(h)} = \frac{e^{c_k}}{\sum_j e^{c_j}}.
$$

最终送入两条头的是

$$
q_t^{\text{src}} = a_t \odot w^{(q)},
\qquad
h_t^{\text{src}} = a_t \odot w^{(h)}.
$$

这里有一个非常重要的实现事实：

- 这组 `modal_rates = \lambda_k` 只用于 encoder 侧的 modal split 和 Koopman-style modal decay loss。
- 真正用于 latent rollout 的慢/快时间尺度，是后面动力学里单独学出来的 `slow_rates(q)` 和 `fast_rates(q)`。

也就是说，当前代码里有“两套速率”：

1. `modal_rates`：给 soft spectrum 和 Koopman modal loss 用。
2. `slow_rates / fast_rates`：给真实动力学推进用。

### 3. 慢变量头、VAMP whitening 与隐藏头

慢变量头先产生未白化特征：

$$
\tilde q_t = f_{\text{vamp}}(q_t^{\text{src}}).
$$

`RunningWhitenedVAMP` 维护运行均值和协方差：

$$
\mu_{\text{run}}, \qquad C_{\text{run}}.
$$

训练时如果 batch size 大于 1，会用当前 batch 的统计量做指数滑动更新；输出的 whitened slow coordinate 是

$$
q_t = (\tilde q_t - \mu) C^{-1/2}.
$$

其中 `C^{-1/2}` 通过特征分解计算，且 whitening 矩阵在代码里是 `detach` 的，所以梯度不会穿过协方差估计本身。

隐藏变量头是一个线性层：

$$
h_t = W_h h_t^{\text{src}} + b_h.
$$

最终 latent state 为

$$
z_t = [q_t, h_t] \in \mathbb{R}^{d_q + d_h}.
$$

### 4. 最终连续时间动力学

当前代码里的真实动力学不是旧 README 那种“通用非线性 `m` 方程”，而是下面这个更结构化的 `q/h` 系统。

#### 4.1 慢变量 `q`

定义：

$$
r_q(q) = r_{q,\min} + \operatorname{softplus}(f_{q,\text{rate}}(q)) \in \mathbb{R}^{d_q},
$$

$$
s_q(q) = \alpha_q \tanh(f_{q,\text{res}}(q)),
\qquad \alpha_q = \texttt{slow\_residual\_scale},
$$

$$
C_q(q) = \frac{1}{\sqrt{d_h}} \operatorname{reshape}(f_{q,\text{cpl}}(q)) \in \mathbb{R}^{d_q \times d_h}.
$$

于是

$$
\dot q = -\,r_q(q) \odot q + s_q(q) + C_q(q) h.
$$

这可以理解为：

- 一个显式稳定的线性回拉项 `-r_q(q) \odot q`
- 一个有界残差项 `s_q(q)`
- 一个由隐藏状态驱动的线性耦合项 `C_q(q)h`

#### 4.2 隐藏变量 `h`

先定义快变量速率：

$$
r_h(q) = r_{h,\min} + \operatorname{softplus}(f_{h,\text{rate}}(q)) \in \mathbb{R}^{d_h}.
$$

再定义两个 `q` 条件化矩阵：

$$
S(q) = \frac{1}{2}\left(M(q) - M(q)^\top\right),
$$

$$
D(q) = \frac{N(q)^\top N(q)}{d_h}.
$$

其中：

- `S(q)` 是反对称矩阵，只负责旋转 / 混合，不直接制造能量增长；
- `D(q)` 是半正定耗散矩阵。

于是隐藏生成元是

$$
A_h(q) = \beta S(q) - D(q) - \operatorname{diag}(r_h(q)),
\qquad \beta = \texttt{hidden\_operator\_scale}.
$$

驱动项是

$$
b_h(q) = \alpha_h \tanh(f_{h,\text{drive}}(q)),
\qquad \alpha_h = \texttt{hidden\_drive\_scale}.
$$

最终隐藏动力学为

$$
\dot h = A_h(q) h + b_h(q).
$$

这意味着当前隐藏子系统在 `h` 上是“仿射线性”的，只对 `q` 非线性。

这是当前实现和旧文案最关键的区别之一：

- `h` 仍然可以解释成 closure / memory embedding。
- 但数学上它已经不是“任意非线性 memory kernel ODE”，而是一个 `q` 条件化的仿射状态空间系统。

### 5. 离散步进：`midpoint_q_plus_exact_affine_h`

当前 `step(z, dt)` 的离散化非常具体，`summary.json` 里也会把它记成：

```text
integrator = midpoint_q_plus_exact_affine_h
```

#### 5.1 隐藏子系统的精确仿射步进

如果把 `q` 在一步内冻结，`h` 的方程

$$
\dot h = A h + b
$$

的精确离散解可以写成

$$
h_{t+\Delta t} = \bar A(\Delta t)\, h_t + \bar b(\Delta t),
$$

其中 `\bar A, \bar b` 来自增广矩阵指数：

$$
\exp\!\left(
\begin{bmatrix}
\Delta t\, A & \Delta t\, b \\
0 & 0
\end{bmatrix}
\right)
=
\begin{bmatrix}
\bar A(\Delta t) & \bar b(\Delta t) \\
0 & 1
\end{bmatrix}.
$$

代码里的 `_affine_hidden_transition(...)` 和 `hidden_ssm_matrices(...)` 就是在做这件事。

#### 5.2 整体一步的真实更新公式

设

$$
f_q(q,h) = -\,r_q(q) \odot q + s_q(q) + C_q(q)h.
$$

代码中的一步更新是：

1. 先做 `q` 的 midpoint predictor

   $$
   q_{t+\frac{1}{2}} = q_t + \frac{\Delta t}{2} f_q(q_t, h_t).
   $$

2. 用冻结在 `q_t` 的隐藏算子做半步精确更新

   $$
   h_{t+\frac{1}{2}} = \bar A(q_t, \tfrac{\Delta t}{2})\, h_t + \bar b(q_t, \tfrac{\Delta t}{2}).
   $$

3. 用中点状态评估 `q` 的 drift，并完成整步

   $$
   q_{t+\Delta t} = q_t + \Delta t\, f_q(q_{t+\frac{1}{2}}, h_{t+\frac{1}{2}}).
   $$

4. 再用冻结在 `q_{t+\frac{1}{2}}` 的隐藏算子，从原始 `h_t` 直接做一整步精确更新

   $$
   h_{t+\Delta t} = \bar A(q_{t+\frac{1}{2}}, \Delta t)\, h_t + \bar b(q_{t+\frac{1}{2}}, \Delta t).
   $$

注意最后一步不是从 `h_{t+\frac{1}{2}}` 再推进，而是重新从 `h_t` 出发、用中点冻结算子做整步。这正是代码现在的实现。

### 6. 结构化解码器与慢流形投影

解码器不是黑箱 MLP，而是显式写成

$$
\hat x_t = g(q_t) + B(q_t) h_t.
$$

其中：

$$
g(q_t) = \texttt{manifold\_decoder}(q_t),
$$

$$
B(q_t) = \operatorname{reshape}(\texttt{hidden\_readout\_net}(q_t)) \in \mathbb{R}^{d \times d_h}.
$$

所以观测重构被拆成：

- `g(q)`：慢流形主项
- `B(q)h`：离流形修正 / 快变量修正

模型还暴露了一个显式慢流形投影：

$$
\Pi_{\mathcal M}(q,h) = (q, 0),
$$

对应 `project_to_manifold(...)`。

### 7. RG-like coarse graining

当前 coarse graining 作用在 latent 上：

$$
\mathcal C(q,h)
=
\left(
q + \delta q(q),\ \frac{h}{s}
\right),
$$

其中

$$
\delta q(q) = \tanh(f_{\text{coarse}}(q)) \cdot \frac{\alpha_{\text{cg}}}{s},
$$

$$
s = \texttt{rg\_scale},
\qquad
\alpha_{\text{cg}} = \texttt{coarse\_strength}.
$$

这不是 Wilsonian RG，只是一个在 latent 上检查 coarse/fine 动力学是否大致交换的先验。

## 每个性质是怎么被约束的

下面这部分只写当前 `training.py` 里实际算出来的损失。

记 horizon 集合为 `\mathcal H`，其中主 horizon 是排序后的第一个：

$$
h_* = \min \mathcal H.
$$

一些谱相关损失只用 `h_*`，而多步一致性损失用所有 horizon。

### 1. 重构：让 `(q,h)` 真正解释当前观测

$$
\mathcal L_{\text{rec}} = \operatorname{MSE}(D(z_t), x_t).
$$

这里 `D` 就是上面的结构化解码器。

### 2. VAMP-2：让 `q` 成为慢特征

取当前窗口和主 horizon 未来窗口的 slow code：

$$
Q_0 = q_t,
\qquad
Q_1 = q_{t+h_*}^{\text{enc}}.
$$

先做 batch 内中心化，得到

$$
\bar Q_0,\ \bar Q_1.
$$

再定义

$$
C_{00} = \frac{1}{B-1}\bar Q_0^\top \bar Q_0 + \varepsilon I,
$$

$$
C_{11} = \frac{1}{B-1}\bar Q_1^\top \bar Q_1 + \varepsilon I,
$$

$$
C_{01} = \frac{1}{B-1}\bar Q_0^\top \bar Q_1.
$$

VAMP-2 score 是

$$
\mathcal S_{\text{VAMP2}}
=
\left\|
C_{00}^{-1/2} C_{01} C_{11}^{-1/2}
\right\|_F^2.
$$

训练里最小化的是

$$
\mathcal L_{\text{vamp}}
=
-\frac{1}{d_q}\mathcal S_{\text{VAMP2}}.
$$

### 3. 时间滞后对角化：让 `q` 尽量解耦

仍然使用主 horizon 的 time-lag covariance：

$$
C_{01} = \frac{1}{B-1}\bar Q_0^\top \bar Q_1 + \varepsilon I.
$$

代码里真正优化的是 off-diagonal Frobenius 范数：

$$
\mathcal L_{\text{diag}}
=
\left\|
C_{01} - \operatorname{diag}(C_{01})
\right\|_F^2.
$$

### 4. Koopman modal decay：只约束 encoder 侧 modal basis

如果 `latent_scheme = soft_spectrum`，当前和主 horizon 未来的 modal feature 分别是

$$
a_t,\qquad a_{t+h_*}.
$$

给定 `modal_rates = \lambda`，代码构造指数衰减预测：

$$
\hat a_{t+h_*} = e^{-\lambda\, h_* \Delta t} \odot a_t.
$$

损失是

$$
\mathcal L_{\text{koop}}
=
\operatorname{MSE}(a_{t+h_*}, \hat a_{t+h_*}).
$$

如果 `hard_split`，这一项自动为零。

### 5. 多步 rollout 预测：让 latent dynamics 能解码成未来观测

从当前 latent `z_t` 出发做 rollout：

$$
\hat z_{t+h} = \Phi_h(z_t), \qquad h \in \mathcal H.
$$

未来观测预测损失是

$$
\mathcal L_{\text{pred}}
=
\frac{1}{|\mathcal H|}
\sum_{h \in \mathcal H}
\operatorname{MSE}(D(\hat z_{t+h}), x_{t+h}).
$$

### 6. `q` 对齐：让 rollout 的慢变量和重新编码的慢变量一致

这是代码里名叫 `vamp_align_loss` 的项，但实际含义是 `q`-alignment：

$$
\mathcal L_{q\text{-align}}
=
\frac{1}{|\mathcal H|}
\sum_{h \in \mathcal H}
\operatorname{MSE}(\hat q_{t+h}, q_{t+h}^{\text{enc}}).
$$

这里

$$
\hat q_{t+h} = \text{split}_q(\hat z_{t+h}).
$$

### 7. latent 对齐：让 rollout 的整个 `z` 和重新编码的 `z` 一致

$$
\mathcal L_{\text{latent-align}}
=
\frac{1}{|\mathcal H|}
\sum_{h \in \mathcal H}
\operatorname{MSE}(\hat z_{t+h}, z_{t+h}^{\text{enc}}).
$$

### 8. 半群一致性：让 latent dynamics 更像真正的流

代码对所有满足 `h_1 + h_2 \in \mathcal H` 的组合做：

$$
\Phi_{h_2}(z_{t+h_1}^{\text{enc}})
\approx
z_{t+h_1+h_2}^{\text{enc}}.
$$

对应损失：

$$
\mathcal L_{\text{semigroup}}
=
\operatorname{mean}_{h_1,h_2}
\operatorname{MSE}\!\left(
\Phi_{h_2}(z_{t+h_1}^{\text{enc}}),
z_{t+h_1+h_2}^{\text{enc}}
\right).
$$

注意它不是从 `z_t` 的 rollout 再拆分，而是直接拿未来窗口重新编码出来的 latent 做半群检查。

### 9. 时间尺度分离：让 `h` 比 `q` 快

从当前 batch 的动力学统计量里取：

$$
\bar r_q = \operatorname{mean}(r_q(q_t)),
\qquad
\bar r_h = \operatorname{mean}(r_h(q_t)).
$$

定义 gap：

$$
\Delta_{\text{sep}} = \bar r_h - \bar r_q.
$$

代码里的损失是

$$
\mathcal L_{\text{sep}}
=
\operatorname{ReLU}(\gamma_{\text{sep}} - \Delta_{\text{sep}}),
$$

其中 `\gamma_{\text{sep}} = separation_margin`。

### 10. 隐藏子系统收缩裕量：检查对称部谱界

对隐藏生成元

$$
A_h(q),
$$

取其对称部分

$$
A_h^{\text{sym}}(q)
=
\frac{1}{2}\left(A_h(q) + A_h(q)^\top\right).
$$

定义最大特征值上界

$$
\lambda_{\max}^{\text{sym}}(q)
=
\lambda_{\max}(A_h^{\text{sym}}(q)).
$$

代码中直接最小化

$$
\mathcal L_{\text{contract}}
=
\operatorname{ReLU}\!\left(
\lambda_{\max}^{\text{sym}}(q) + \gamma_{\text{contract}}
\right),
$$

其中 `\gamma_{\text{contract}} = contraction_margin`。

这里有个实现层面的重点：

- 由于 `A_h(q)` 的参数化本身就是“反对称 + 负半定耗散 + 负对角速率”，稳定性已经被硬编码进结构里。
- `contract_loss` 更像是一个额外的谱裕量检查，而不是唯一的稳定性来源。

### 11. RG-like 一致性：让 coarse graining 和推进近似交换

只在 `rg_weight > 0` 且课程训练进入 phase 3 时启用。

代码比较的是：

$$
\mathcal C(\Phi_r(z_t))
\quad\text{vs}\quad
\Phi_r^{(s\Delta t)}(\mathcal C(z_t)),
$$

其中：

- `r = rg_horizon`
- 右边用的是更大的步长 `s \Delta t`
- 两边 step 数都还是 `r`

所以损失写成

$$
\mathcal L_{\text{rg}}
=
\operatorname{MSE}\!\left(
\mathcal C(\Phi_r^{(\Delta t)}(z_t)),
\Phi_r^{(s\Delta t)}(\mathcal C(z_t))
\right).
$$

### 12. 几何保持：让 `q` 不是纯谱上好看、几何上塌缩

代码对 batch 内 pairwise distance 做均值归一化：

$$
\widetilde D(X) = \frac{D(X)}{\operatorname{mean}(D(X)) + 10^{-6}}.
$$

其中输入端用的是展平窗口 `\operatorname{vec}(W_t)`，latent 端用的是 `q_t`。

于是

$$
\mathcal L_{\text{metric}}
=
\operatorname{MSE}\!\left(
\widetilde D(\operatorname{vec}(W_i), \operatorname{vec}(W_j)),
\widetilde D(q_i, q_j)
\right).
$$

为了节约计算，batch 太大时会随机 subsample 到 `metric_subsample` 个样本。

### 13. 隐藏幅值稀疏化：不要让 `h` 无限制膨胀

代码使用的是简单的 `L1` 惩罚：

$$
\mathcal L_{|h|} = \|h_t\|_1^{\text{mean}}.
$$

在配置和日志里它有时叫 `hidden_l1_loss`，有时叫 `memory_l1_loss`，本质上都是同一项。

### 14. 可选监督：`q` / `h` 对标签分量做对齐

如果传入标签并给正权重，会计算：

$$
\mathcal L_{\text{sup}}
=
w_q \mathcal L_{\text{sup},q}
+
w_h \mathcal L_{\text{sup},h}.
$$

其中：

- `q` 监督既比较当前步，也比较所有 future horizon。
- `h` 监督也比较当前步和 future horizon。
- `q_mode = direct` 时用普通 MSE。
- `q_mode = angular` 时用 wrap 后的角距离平方：

  $$
  \delta(\hat y, y) = \operatorname{atan2}(\sin(\hat y-y), \cos(\hat y-y)),
  \qquad
  \ell = \delta^2.
  $$

还有一个实现细节很关键：

- 当 `q_mode = angular` 时，监督的不是 whitened `q`，而是 `q_raw`。
- 也就是角度监督发生在 VAMP whitening 之前，避免 whitening 把角变量几何关系扭曲掉。

### 15. 总目标

当前代码的总损失是

$$
\mathcal L
=
\lambda_{\text{rec}}\mathcal L_{\text{rec}}
+
\lambda_{\text{vamp}}\mathcal L_{\text{vamp}}
+
\lambda_{q\text{-align}}\mathcal L_{q\text{-align}}
+
s_{\text{koop}}(p)\lambda_{\text{koop}}\mathcal L_{\text{koop}}
+
s_{\text{diag}}(p)\lambda_{\text{diag}}\mathcal L_{\text{diag}}
+
s_{\text{pred}}(p)\lambda_{\text{pred}}\mathcal L_{\text{pred}}
+
s_{\text{latent}}(p)\lambda_{\text{latent}}\mathcal L_{\text{latent-align}}
+
s_{\text{sg}}(p)\lambda_{\text{sg}}\mathcal L_{\text{semigroup}}
+
s_{\text{contract}}(p)\lambda_{\text{contract}}\mathcal L_{\text{contract}}
+
s_{\text{sep}}(p)\lambda_{\text{sep}}\mathcal L_{\text{sep}}
+
s_{\text{rg}}(p)\lambda_{\text{rg}}\mathcal L_{\text{rg}}
+
\lambda_{\text{metric}}\mathcal L_{\text{metric}}
+
\lambda_{|h|}\mathcal L_{|h|}
+
\mathcal L_{\text{sup}}.
$$

这里 `p` 表示 curriculum phase。

默认权重来自 `LossConfig`：

```python
LossConfig(
    reconstruction_weight=1.0,
    vamp_weight=0.2,
    vamp_align_weight=0.25,
    koopman_weight=0.25,
    diag_weight=0.05,
    prediction_weight=1.0,
    latent_align_weight=0.75,
    semigroup_weight=0.5,
    separation_weight=0.2,
    contract_weight=0.2,
    rg_weight=0.05,
    metric_weight=0.1,
    hidden_l1_weight=1e-4,
    separation_margin=0.5,
    contraction_margin=0.10,
)
```

## 课程训练与 phase 行为

当前训练分三阶段：

### Phase 1

- dynamics 模块冻结
- 开启 `koopman`、`diag`
- 关闭 `prediction`、`latent_align`、`semigroup`、`contract`、`separation`、`rg`
- `reconstruction`、`vamp`、`q-align(vamp_align)`、`metric`、`hidden_l1`、监督项仍然存在

### Phase 2

- dynamics 模块解冻
- 开启 `prediction`、`latent_align`、`contract`、`separation`
- `semigroup` 和 `rg` 仍关闭

### Phase 3

- 开启全部项
- 学习率按 `phase3_lr_scale` 缩放

CLI 里可以用：

- `--curriculum_preset legacy`
- `--curriculum_preset conservative`
- `--curriculum_preset alanine_bootstrap`

也可以手动覆盖 `epochs`、`phase1_fraction`、`phase2_fraction`、`phase3_lr_scale`。

## 为了稳定性和效率做的合理简化

这一节专门解释“代码现在为什么这样写”。

### 1. 用 `q,h` 而不是旧的 `q,m` 理论叙事

当前实现里，`h` 更准确的说法是：

- hidden fast state
- closure state
- memory embedding

它当然仍然能承担 Mori-Zwanzig 风格的“有限维记忆嵌入”角色，但代码已经不再直接实现“通用 nonlinear memory ODE + kernel source term”那套更宽泛的文案。

### 2. 把隐藏动力学限制成 `q` 条件化的仿射状态空间

这是当前实现最重要的结构简化：

$$
\dot h = A_h(q)h + b_h(q),
$$

而不是更一般的

$$
\dot h = F(q,h).
$$

这么做的好处是：

- 可以直接用矩阵指数得到稳定的一步离散化
- 能把耗散、旋转、正速率分开参数化
- 为后续 selective scan / SSD 风格内核预留统一接口

代价是：

- `h` 的非线性只来自 `q` 条件化，不直接来自 `h` 自身的更高阶非线性
- 表达能力比“任意非线性快变量方程”更受限

### 3. 稳定性大部分来自结构参数化，不是靠罚项硬拉

当前 `A_h(q)` 的结构

$$
\beta S(q) - D(q) - \operatorname{diag}(r_h(q))
$$

本身就把“反对称混合 + 耗散 + 正衰减率”编码进去了。

这比只写一个任意矩阵然后靠 spectral penalty 去补救更稳定，也更容易解释。

### 4. `q` 用 midpoint，`h` 用 exact affine step

这不是为了好看，而是因为：

- `q` 是非线性、低维、需要中点信息
- `h` 是条件仿射线性，完全可以利用精确离散化

所以现在的 integrator 是一个混合方案，而不是统一用 Euler 或黑盒 ODE solver。

### 5. VAMP whitening 统计量不参与反向传播

`RunningWhitenedVAMP` 里的均值、协方差和 whitening 矩阵在使用时都被 `detach` 了。

这样做会牺牲一部分“端到端严格性”，但训练会明显更稳，也避免协方差逆平方根把梯度图搞得很重。

### 6. 谱损失只盯主 horizon

`VAMP`、`diag`、`koopman modal decay` 都只用排序后最小的 horizon `h_*`。

这是一个明确的效率/稳定性折中：

- 优点：目标更稳定，梯度更集中，训练开销更低
- 缺点：长时谱性质主要靠 rollout / semigroup / prediction 间接约束，而不是每个 horizon 都直接加谱损失

### 7. RG-like coarse graining 只发生在 latent 上

当前 coarse graining 只改 `q` 和 `h`：

- `q` 加一个小的 learned correction
- `h` 除以 `rg_scale`

它没有去 coarse-grain encoder、decoder 或原始观测空间，所以它是一个实用的 latent consistency prior，不是完整的多层级 RG 框架。

### 8. 兼容别名还在，但语义已经迁移

代码里还保留了：

- `--m_dim` 作为 `--h_dim` 的别名
- `min_memory_rate` 作为 `min_fast_rate` 的别名
- `m_indices` / `m_weight` 作为 `h_indices` / `h_weight` 的别名
- 导出的 `memory_*` 指标

这是为了兼容旧脚本，不代表模型内部仍然以旧 `m` 叙事为核心。

## 为什么说它是 SSD-ready，但还不是完整 Mamba kernel

这里的 “SSD-ready” 不是营销词，而是指当前隐藏子系统已经具备被改写成 selective state-space / scan 风格实现的数学接口。

### 1. 为什么说它已经 SSD-ready

对给定的 `q` 轨迹，隐藏状态一步更新已经可以写成

$$
h_{k+1} = \bar A_k h_k + \bar b_k,
$$

其中

$$
\bar A_k = \bar A(q_{k+\frac12}, \Delta t),
\qquad
\bar b_k = \bar b(q_{k+\frac12}, \Delta t).
$$

代码里 `hidden_ssm_matrices(q, dt)` 已经直接返回：

- `generator`
- `transition`
- `bias`
- `fast_rates`

也就是说，隐藏子系统已经被整理成了“每步给出状态转移矩阵和偏置”的形式。

从这个角度看，它已经满足：

1. 有显式状态矩阵 `A`
2. 有显式离散转移 `\bar A`
3. 有显式输入/驱动偏置 `\bar b`
4. hidden recurrence 和 decoder 是解耦接口

这就是后续接 scan kernel、chunked recurrence、SSD-style dual view 的基础。

### 2. 为什么它还不是完整 Mamba kernel

当前实现距离完整 Mamba / fused selective scan 还有明确差距：

1. 整个 latent 更新不是纯线性 scan  
   `q` 的更新依赖非线性 midpoint drift，而且 `q` 和 `h` 双向耦合，所以整套 `z=(q,h)` 还不能直接视为一个单一线性递推核。

2. 当前每步都在做 dense matrix exponential  
   `hidden_ssm_matrices` 通过 `torch.matrix_exp` 计算增广矩阵指数。这在数学上很干净，但不是 Mamba 那种专门为长序列做的结构化、融合化 kernel。

3. 没有 fused CUDA / Triton selective scan  
   现在的 rollout 是 Python/PyTorch 循环里的逐步 `model.step(...)`，没有并行 prefix scan、没有 chunked backward kernel、也没有专用 GPU fused 实现。

4. 没有 Mamba 风格的完整输入输出投影接口  
   当前模型有窗口 encoder 和结构化 decoder，但没有 Mamba 那种围绕 token stream、causal conv、input-dependent parameter generation 搭起来的整套 block。

5. `A_t` 的选择性来自 `q_t`，不是直接来自每个 token 的投影参数流  
   这更像“非线性慢变量驱动的条件 SSM”，而不是标准 Mamba block 里的 selective parameterization。

6. 训练范式还是“窗口编码 + rollout 对齐”  
   不是直接对整段序列做高吞吐流式 selective scan。

所以更准确的说法应该是：

> 当前仓库已经把 `h` 子系统整理成了可离散、可导出、可替换成 scan 内核的仿射 SSM 形式，因此是 SSD-ready；但由于 `q` 的非线性 midpoint 更新、`q/h` 耦合、dense matrix exponential 和缺失 fused scan kernel，它还不是完整的 Mamba kernel。

## 重要 CLI 选项

最常用的参数：

- `--window`：窗口长度 `L`
- `--q_dim`：慢变量维度
- `--h_dim`：隐藏快变量维度
- `--m_dim`：`--h_dim` 的兼容别名
- `--latent_scheme {hard_split,soft_spectrum}`
- `--modal_dim`
- `--modal_temperature`
- `--encoder_type {temporal_conv,mlp}`
- `--encoder_levels`
- `--encoder_kernel_size`
- `--horizons`
- `--dt`
- `--curriculum_preset`
- `--q_label_indices`
- `--h_label_indices`
- `--m_label_indices`：`--h_label_indices` 的兼容别名
- `--q_supervised_weight`
- `--h_supervised_weight`
- `--m_supervised_weight`：`--h_supervised_weight` 的兼容别名
- `--q_supervision_mode {direct,angular}`
- `--eval_batch_size`

当前 CLI 还有两个边界需要写清楚：

1. `--contract_batch` 目前会被解析并记录进 `TrainConfig`，但训练代码没有实际使用它。
2. `ModelConfig` 里的 `min_slow_rate`、`min_fast_rate`、`hidden_operator_scale`、`hidden_drive_scale`、`slow_residual_scale`、以及 `LossConfig` 里的 `separation_margin` / `contraction_margin` 目前没有 CLI 开关，仍使用代码默认值。

## 合成基准的真实内容

### `toy`

隐藏真值是四维：

$$
(q_1, q_2, a, m),
$$

其中有明显慢快分离；导出的 `synthetic_hidden_state.csv` 和 `synthetic_labels.csv` 都是这四个量。

### `no_gap_toy`

也是四维隐藏系统，但慢快 gap 被显著削弱，用来检查模型是否会在没有清晰谱分离时过拟合出虚假的慢变量结构。

### `alanine_like`

隐藏真值是：

- 两个角变量 `(\phi,\psi)`
- `alanine_fast_dim` 个快变量
- 一个 closure-like hidden scalar

实现里：

- 监督标签 `labels` 是 `[\phi,\psi,\text{fast}_1,\dots]`
- probe 标签 `probe_labels` 是 `[sin\phi, cos\phi, sin\psi, cos\psi, \text{fast}_1,\dots]`
- closure scalar 只在 `hidden_state` 里，不在默认监督标签里

## 当前局限

- 没有打包元数据和正式安装入口。
- hidden recurrence 虽然已经是 affine SSM，但仍用 dense `matrix_exp`，不适合长序列高吞吐训练。
- `q` 分支仍是非线性 midpoint 更新，无法直接并入一个单核 selective scan。
- RG-like 项只是 latent consistency prior，不是完整 RG 理论实现。
- phase 训练里有些项一直开启、有些项按 phase gate；这是一套工程上好用的课程策略，不是严格从理论推导出来的唯一方案。
- `contract_batch` 目前未被使用。

## 代码级对照表

| 概念 | 真实实现位置 |
| --- | --- |
| 窗口编码 | `TemporalMultiscaleEncoder` / `encoder_backbone` |
| soft spectral split | `modal_rates`、`modal_weight_vectors`、`encode_components` |
| VAMP whitening | `RunningWhitenedVAMP` |
| `q/h` latent state | `split_latent`、`join_latent` |
| 慢变量动力学 | `latent_statistics` 中的 `dq` |
| 隐藏仿射 SSM | `_hidden_operator`、`hidden_ssm_matrices`、`_affine_hidden_transition` |
| 混合积分器 | `step` |
| 结构化解码器 | `decode_parts` |
| 慢流形投影 | `project_to_manifold` |
| latent coarse graining | `coarse_grain` |
| VAMP-2 | `_vamp2_score` |
| modal Koopman loss | `_koopman_consistency_loss` |
| 多步 rollout | `_rollout_cache` |
| `q` 对齐 | `_q_align_loss` |
| latent 对齐 | `_latent_align_loss` |
| 半群一致性 | `_semigroup_loss` |
| 收缩裕量 | `_memory_contract_loss` |
| 总损失拼装 | `_loss_bundle` |

## 结论

如果只用一句话描述当前代码，请以这句为准：

> 当前仓库实现的是一个 `q/h` 慢快状态空间模型：`q` 通过 VAMP 风格目标学习慢坐标，`h` 通过 `q` 条件化仿射 SSM 承担快变量 / closure 状态，观测由 `g(q)+B(q)h` 解码，训练由重构、多步预测、latent 对齐、半群、时间尺度分离、谱对角化和 RG-like 一致性共同约束。它已经具备 hidden SSM 层面的 SSD-ready 结构，但还没有实现完整的 Mamba fused kernel。
