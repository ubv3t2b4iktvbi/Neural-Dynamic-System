# Neural Dynamic System

目标：从时间序列中学出一个可解释的慢快动力学模型，把观测分解成 `Koopman slow state q + hidden fast memory h`，用于重构、预测和 RG/semigroup 一致性约束。

这个 README 只描述当前代码里真实实现的模型，不沿用旧的 `q,m` 文案。

## 核心直觉

我们假设观测时间序列里同时有两种时间尺度：

- 慢变量 `q`：跨更长 horizon 仍保留、对 coarse-graining 更稳定的结构
- 快变量 `h`：衰减更快、承担短时记忆和局部修正的隐藏状态

这个模型的重点不是把所有信息都塞进一个黑箱 latent，而是显式写成：

- `q`：Koopman-aligned slow state
- `h`：`q` 条件化的 affine hidden SSM / memory state
- `\hat x`：由 `g(q) + D(q) h` 生成的重构或预测

## 和常见模型路线相比，这个设计想突出什么

下面这个对比是“结构差异”而不是 benchmark 排名。它想说明的不是别的方法都不行，而是当前实现特意把哪些性质做成了模型本身的一部分。

| 路线 | 典型做法 | 常见问题 | 当前实现强调的点 |
| --- | --- | --- | --- |
| 黑箱 autoencoder / latent ODE | 一个统一 latent，加一个通用向量场或 RNN | 慢快分工通常要靠后验分析，latent 语义容易混在一起 | `q` 和 `h` 从一开始就分工，`h` 也不是任意黑箱 ODE，而是 affine hidden SSM |
| 纯 Koopman autoencoder | 显式谱坐标，强调线性或近线性推进 | 容易把闭合误差、短时记忆、局部修正都挤进同一组谱坐标 | `q` 只承接慢模态，`h` 专门承担 closure / memory / fast correction |
| 纯 SSM / Mamba 类模型 | 强状态递推、强扫描友好性 | 状态往往缺少清晰的几何和谱语义 | 我们保留显式 Koopman slow coordinates、结构化 decoder、RG/semigroup 分支 |
| 仓库里的 legacy 变体 | `joint` Koopman 输入 + `direct` hidden 坐标 | `phi/q` 和 fast correction 更容易纠缠 | 推荐几何版本把 Koopman 输入限制在 slow summary，并把 `h` 放到 normal residual 坐标里 |

如果用一句话概括，这个模型想做的不是“再造一个更大的黑箱”，而是把可解释慢模态、闭合误差对应的快记忆、以及后续 SSD 化兼容性，尽量同时放进一个统一结构里。

## 什么是慢，什么是快

这里的“快 / 慢”指的是 latent dynamics 的时间尺度，不是 encoder 通道深浅，也不是 `u^(q)` / `u^(h)` 这种实现里的中间 summary。

- 慢：对应更小的衰减率，经过更长时间推进后仍保留，也更容易在 RG coarse-graining 下保留。
- 快：对应更大的阻尼 / 收缩率，主要负责短时记忆、局部修正，以及对 slow manifold 的反馈。

在当前模型里：

- `q` 是慢变量，和显式 Koopman 速率 `\lambda_q` 对齐。
- `h` 是快变量，按 `q` 条件化的 affine hidden SSM 推进。

## 理论主图

```mermaid
flowchart TB
    classDef io fill:#f6efe4,stroke:#946c3b,stroke-width:1.5px,color:#2b2117;
    classDef stage fill:#eef5ee,stroke:#537d60,stroke-width:1.5px,color:#142117;
    classDef core fill:#edf3fb,stroke:#4d709e,stroke-width:1.6px,color:#162338;
    classDef detail fill:#fff1e3,stroke:#c2762e,stroke-width:1.2px,color:#3f2411;
    classDef aux fill:#f3edf8,stroke:#7b5ca6,stroke-width:1.2px,color:#251739;

    subgraph META["核心细节"]
        direction LR
        D1["显式 Koopman 特征 phi"]
        D2["slow state q + fast state h"]
        D3["q: midpoint; h: affine hidden SSM"]
    end

    subgraph MAIN[" "]
        direction LR
        X["输入窗口 / 观测 x"]
        ENC["Encoder E(W_t)"]
        CORE["q/h 状态空间核心<br/>构造 z_t = (q_t, h_t)，并推进到 z_(t+1)"]
        DEC["结构化 decoder<br/>x_hat = g(q) + D(q) h"]
        Y["重构 / 预测"]

        X --> ENC --> CORE --> DEC --> Y
    end

    subgraph RGB["RG branch (loss only)"]
        direction TB
        RG1["RG 谱坐标"]
        RG2["q_tilde = sqrt(lambda_q) * q<br/>h_tilde = H_damp(q)^(-1/2) h"]
        CG["coarse-grain C_s"]
        RG1 --> RG2 --> CG
    end

    D1 --> CORE
    D2 --> CORE
    D3 --> CORE
    CORE -. "仅用于 RG loss" .-> RG1

    class X,Y io;
    class ENC,DEC stage;
    class CORE core;
    class D1,D2,D3 detail;
    class RG1,RG2,CG aux;

    style META fill:#fffaf3,stroke:#e0c49e,stroke-width:1.0px
    style MAIN fill:#ffffff,stroke:#ffffff,stroke-width:0px
    style RGB fill:#faf6fc,stroke:#cbb5df,stroke-width:1.5px
```

读图方式：

- 顶层主图只保留理论对象和理论模块：观测、`q/h` 状态空间核心、decoder、RG branch。
- `u^(q)` / `u^(h)` 这样的编码器内部 summary 不放在这里，而放到后面的“实现映射”一节。
- 中间主块对应的是整个 `q/h` 状态空间核心，而不是代码里每一个小模块调用。

## 理论定义

最核心的状态和观测关系是：

$$
z_t = (q_t, h_t)
$$

$$
\hat x_t = g(q_t) + D(q_t) h_t
$$

其中 `q` 负责慢时间尺度结构，`h` 负责快时间尺度记忆和修正。

## 动力学定义

### 1. Koopman 速率

模型维护一组全局可训练正速率

$$
0 < \lambda_1 \le \lambda_2 \le \dots \le \lambda_K
$$

其中 `K = koopman_dim`。

前 `q_dim` 个速率就是慢变量的主对角衰减率：

$$
\Lambda_q = diag(\lambda_1, \dots, \lambda_{d_q})
$$

### 2. 慢变量方程

当前代码里的 `q` 动力学是

$$
\dot q = - \Lambda_q q + r_q(q) + B h
$$

其中：

- `- Lambda_q q` 是 Koopman-aligned slow decay
- `r_q(q)` 是小的非线性残差项
- `B h` 是 hidden memory 对 slow manifold 的耦合

实现上：

- `B` 是一个常数线性耦合矩阵
- `r_q(q)` 由 `q_residual_net` 给出并经过 `tanh`

### 3. 隐藏快变量方程

当前代码把 `h` 写成 `q` 条件化的 affine hidden SSM：

$$
\dot h = A(q) h + b(q)
$$

其中

$$
A(q) = - diag(r_h(q)) + alpha U diag(gamma(q)) V^T
$$

这里：

- `r_h(q)` 是正的 fast rates
- `U, V` 是全局低秩因子
- `gamma(q)` 是 `q` 条件化低秩系数
- `alpha = hidden_operator_scale`

驱动项是

$$
b(q) = beta tanh(f_h(q))
$$

这里 `beta = hidden_drive_scale`。

这意味着当前 `h` 子系统不是一个任意黑箱 ODE，而是一个稳定对角基座加低秩修正的仿射状态空间模型。

## 哪些性质是模型设计自动实现的，哪些是 loss 在推动

这个区分很重要，因为如果只写目标，很容易让人误以为所有性质都只是“希望学出来”。当前实现里，确实有一部分性质是由参数化和积分方式直接决定的。

### 结构上直接成立的性质

- `q` 直接取自显式 Koopman 特征 `phi` 的前 `q_dim` 维，不是另外再学一套没有谱含义的 slow latent。
- Koopman 速率通过“正基值 + 正增量”的方式参数化，所以速率天然为正，而且按索引有序。
- `h` 的连续时间方程始终是 `A(q) h + b(q)` 的 affine hidden SSM；其中 `A(q)` 始终是“负对角阻尼 + 条件低秩修正”。
- `h` 的离散一步始终来自增广矩阵指数的 exact affine step，而不是普通显式积分器近似。
- RG 谱归一化坐标只出现在 RG loss 分支，不会直接改写主干的 encoder、rollout、decoder 坐标。
- 当 `hidden_coordinate_mode=normal_residual` 时，`h` 会先由 decoder 几何里的 normal residual 定义，再做一个小 refinement；也就是说，切向变化优先归给 `q`，法向补偿优先归给 `h`。

### 仍然需要 loss 去推动的性质

- `phi` 是否真的接近 time-lag Koopman 模态，要靠 VAMP、time-lag diagonalization、Koopman decay consistency 一起约束。
- `h` 比 `q` 更快这件事不是完全自动的，要靠 separation loss 继续拉开平均速率间隔。
- hidden operator 的整体收缩性不是完全由参数化硬保证的，要靠 contract loss 压住对称部谱上界。
- `F_dt(F_dt(z)) \approx F_{2dt}(z)` 这类 semigroup 一致性，以及 coarse-grain 前后可交换性，要靠 semigroup loss 和 RG loss。
- memory branch 不要无界膨胀，要靠 hidden L1 和预测/重构共同约束。

## 为什么 `q` 用 midpoint，而 `h` 用 exact affine step

### `q` 用 midpoint / RK2

代码里的 `q` 用的是固定步长 midpoint：

$$
q_{n+1/2} = q_n + 0.5 dt f_q(q_n, h_n)
$$

$$
q_{n+1} = q_n + dt f_q(q_{n+1/2}, h_{n+1/2})
$$

这里不是直接用 neural ODE solver，原因是：

1. 当前训练目标本来就是固定 `dt`、固定 horizon 的离散 rollout。
2. Koopman、semigroup、RG 都围绕一个明确的离散映射 `F_dt` 来约束。
3. neural ODE 学到的是“向量场 + 求解器”的组合，不够干净，也不利于和 SSD 快分支对齐。
4. midpoint 比 Euler 稳，代价又远低于通用 ODE solver。

所以这里选 midpoint，不是因为“看起来像 ODE”，而是因为它给了我们一个明确、固定、二阶的离散流映射。

### `h` 不用普通 RK2，而用 exact affine step

当一步内把 `q` 冻结或用中点冻结时，`h` 的方程

$$
\dot h = A h + b
$$

在一步内是仿射线性的，所以一步解可以直接写成

$$
h_{n+1} = \bar A h_n + \bar b
$$

其中 `bar A` 和 `bar b` 来自增广矩阵指数。

这比直接对 `h` 用 RK2 更合理，原因是：

1. `h` 子系统本来就有结构化闭式一步映射，没必要退回近似法。
2. `h` 是快变量，通常更 stiff，普通显式 RK2 更容易破坏收缩性。
3. `q` 的方程里有 `B h`，所以 `h` 的数值误差会直接污染 slow dynamics。
4. SSD / scan kernel 需要的正是

   $$
   h_{n+1} = \bar A_n h_n + \bar b_n
   $$

   这种递推形式。

所以当前代码保留了：`q` 用 midpoint，`h` 用 exact affine / exponential step。

## 为什么 RG 特殊变换只在 RG 分支里用

RG 关心的是“尺度比较”，不是主模型的本体坐标。

主模型的 latent 还是

$$
z = (q, h)
$$

但在 RG 分支里，代码先做一个专用坐标变换：

$$
\tilde q = \sqrt{\lambda_q} \odot q
$$

$$
H_{damp}(q) = - \frac{A(q) + A(q)^T}{2}
$$

$$
\tilde h = H_{damp}(q)^{-1/2} h
$$

这一步的目的只是让不同时间尺度的量在同一个尺度下比较。
也就是说，当前 `h` 的 RG 归一化不再是逐维 `1 / \sqrt{r_h(q)}`，而是按 hidden operator 的对称阻尼度量做矩阵归一化。

它不应该被强行塞进主 encoder、主 rollout、主 decoder 里，否则整个模型都会被 RG 坐标绑架，重构与预测会更难训，解释也会变差。

所以当前实现是：

- 主动力学在原始 `z = (q,h)` 上做
- 只有 RG loss 里，才进入 `(\tilde q, \tilde h)` 坐标

## 当前 RG 是怎么做的

在 RG 坐标里，coarse-graining 写成

$$
\tilde q' = m(s) \odot \tilde q + delta_q(\tilde q)
$$

$$
\tilde h' = \tilde h / s
$$

其中：

- `s = rg_scale`
- `m(s)` 是按 Koopman 速率构造的 soft mask
- `delta_q` 是一个 near-identity 小修正

soft mask 的思想是：

- 慢模态保留更多
- 相对更快的 slow modes 在更 coarse 的尺度下被压小

代码里 cutoff 用的是

$$
\lambda_c = max(\lambda_q) / s
$$

再通过 `sigmoid` 做 soft mask。
对 `h` 的 coarse-graining 仍然是在 RG 坐标里做 `\tilde h' = \tilde h / s`，然后再通过 `H_{damp}(q)^{1/2}` 映回原始 hidden 坐标。

RG loss 比较的是：

$$
C_s(F_{dt}^r(z))
$$

和

$$
F_{s dt}^r(C_s(z))
$$

也就是：

- 先推进再 coarse-grain
- 先 coarse-grain 再用大步长推进

它们是否近似一致。

注意：这里只有带 `C_s` 的项才是 RG。单纯的

$$
F_dt(F_dt(z)) \approx F_{2dt}(z)
$$

只是 semigroup consistency，不是 RG。

## 实现映射：代码里怎样从窗口得到 `z`

前面的理论部分只讲 `q` 和 `h`。真正落到代码时，模型会先从输入窗口里提取两个内部 summary，再构造 Koopman 特征和 hidden init。

输入窗口是

$$
W_t = [x_{t-L+1}, \dots, x_t]
$$

先经过 encoder：

$$
(u_t^{(q)}, u_t^{(h)}) = E(W_t)
$$

然后显式产生 Koopman 特征：

$$
\phi_t^{raw} = K_\theta(u_t)
$$

$$
\phi_t = Whiten(\phi_t^{raw})
$$

这里 `phi_t` 的维度是 `koopman_dim`。在当前代码里，这一步更准确地说是 running mean 加按通道方差归一化，用来尽量保留 Koopman 模态顺序。

实现上，慢变量不是单独再学一套完全独立的向量，而是直接取 Koopman 特征的 slow 子空间：

$$
q_t = \phi_t[:d_q]
$$

隐藏快变量由 encoder 的 fast summary 和 Koopman 特征共同初始化：

$$
h_t = H_\theta([u_t^{(h)}, \phi_t^{fast}])
$$

最终得到

$$
z_t = (q_t, h_t)
$$

这里要特别注意：

- `u_t^(q)` / `u_t^(h)` 只是 encoder 内部 summary，不是理论上的最终 latent。
- 理论上的主角始终是 `q` 和 `h`，也就是慢变量和快变量。
- RG 坐标只用于 RG loss 分支，不进入主干 encoder、主干 rollout 和主干 decoder。

### `temporal_conv`

`TemporalMultiscaleEncoder` 用多尺度时序卷积提取：

- `u^(q)`：偏慢、偏全局的 summary
- `u^(h)`：带更多多尺度 fast 信息的 summary

### `mlp`

窗口直接 flatten 后经过 MLP。

### `latent_scheme`

当前仍支持：

- `hard_split`
- `soft_spectrum`

差别是：

- `hard_split`：Koopman 头只看 `u^(q)`
- `soft_spectrum`：Koopman 头看 `[u^(q), u^(h)]`，并用谱权重构造 fast-side initialization

也就是说，`soft_spectrum` 现在不再是旧版“`q,m` 两块 latent 的最终定义”，而是 Koopman feature construction 的方式。

### `koopman_input_mode`

当前支持：

- `slow_only`
- `joint`

差别是：

- `slow_only`：Koopman 头只吃 slow summary，等于先尽量把 `phi/q` 固定成“慢结构坐标”，再让 `h` 承担 short-memory 修正。
- `joint`：Koopman 头同时看 slow/fast summary，表达力更强，但也更容易把快修正直接混进谱坐标。

这是算法设计里的一个关键点：我们不是只做“是否显式建 Koopman 特征”的选择，而是进一步决定“快记忆能不能直接污染 Koopman 坐标”。

### `hidden_coordinate_mode`

当前支持：

- `direct`
- `normal_residual`

差别是：

- `direct`：直接从 fast summary 和 Koopman fast-side features 生成 `h`。这是最直接的 baseline，灵活，但 `h` 的几何语义更弱。
- `normal_residual`：先用 `g(q)` 生成当前点在 slow manifold 上的 base，再把观测残差拆成 tangent 部分和 normal 部分，只把 normal residual 投到 hidden basis 上，最后做一个小 refine。

`normal_residual` 的意义很大，因为很多模型想靠额外的正交损失去“希望”慢变量和快修正分工；这里是先通过坐标构造把这件事落实，再让训练去微调。

## 仓库里现成的对比入口

如果想把“我们的算法和现有变体的差异”落到可复现实验，而不是停留在概念层面，可以直接看 `scripts/vdp_suite.py` 里的四组配置：

- `full_geometry_baseline`：`koopman_input_mode=joint`，`hidden_coordinate_mode=normal_residual`，`metric_mode=mahalanobis_dynamics`
- `reduced_geometry`：`koopman_input_mode=slow_only`，`hidden_coordinate_mode=normal_residual`，`metric_mode=mahalanobis_dynamics`
- `reduced_direct`：`koopman_input_mode=slow_only`，`hidden_coordinate_mode=direct`，`metric_mode=euclidean`
- `reduced_geometry_semigroup_weak`：在 `reduced_geometry` 基础上削弱 semigroup 权重

这组对照的价值在于，它没有把 backbone 整个换掉，而是尽量控制住其他因素，专门测试四件事：

1. Koopman 头是否同时看 slow/fast summary，还是只看 slow summary。
2. `h` 是否定义在 normal residual 坐标里。
3. 度量约束是否从普通欧氏距离升级为更贴近局部动力学的 Mahalanobis 动态距离。
4. semigroup 约束减弱后，结构化 rollout 的稳定性会掉多少。

## 当前训练目标

### 1. Reconstruction

结构化 decoder 仍然是

$$
\hat x = g(q) + D(q) h
$$

损失：

$$
L_rec = MSE(\hat x_t, x_t)
$$

### 2. VAMP-2

VAMP 现在作用在显式 Koopman 特征 `phi` 上，而不是只作用在 `q` 上。

$$
L_vamp = - VAMP2(\phi_t, \phi_{t+\tau})
$$

### 3. Time-lag diagonalization

仍然作用在 `phi` 上，推动 time-lag covariance 更接近对角。

### 4. Koopman modal decay

对每个 Koopman feature，代码用学习到的速率做一阶指数衰减约束：

$$
\phi_{t+\tau} \approx exp(- \lambda \tau) \odot \phi_t
$$

### 5. Multi-step prediction

从 `z_t` rollout 到多个 horizon，再解码到观测空间。

### 6. Q alignment

rollout 后的 `q` 要和未来窗口重新编码得到的 `q` 对齐。

### 7. Latent alignment

rollout 后的整个 `z` 要和未来窗口重新编码得到的 `z` 对齐。

### 8. Semigroup consistency

要求不同 horizon 的 flow 复合近似一致。

### 9. Separation

要求 `h` 的平均快速率大于 `q` 的平均慢速率。

### 10. Contract

要求 hidden generator 的对称部谱上界不要跑到不稳定区域。

### 11. RG loss

只在 phase 3 打开，并且只在 RG 分支里用谱归一化坐标。

### 12. Hidden L1

控制 `h` 的幅值，不让 memory branch 无限制膨胀。

## 当前代码为什么说是 SSD-ready

因为 `h` 的一步更新已经被整理成了

$$
h_{n+1} = \bar A_n h_n + \bar b_n
$$

并且 `A(q)` 本身是“稳定对角基座 + 低秩条件修正”的形式。这和后续的 SSD / DPLR / scan kernel 很接近。

当前还不是完整 Mamba kernel，原因也很明确：

1. `q` 仍然是非线性 midpoint 更新。
2. `q` 和 `h` 之间仍有双向耦合。
3. 当前 `h` 的离散化还是用 `torch.matrix_exp`，不是 fused selective scan。
4. 没有 Triton / CUDA fused kernel。

所以更准确的说法是：

> 现在的 hidden branch 已经是 SSD-ready 的 affine SSM，但整个模型还不是完整的 Mamba block。

## 主要代码位置

- `neural_dynamic_system/model.py`
  - `koopman_head`
  - `koopman_whitener`
  - `latent_statistics`
  - `hidden_ssm_matrices`
  - `step`
  - `rg_transform`
  - `coarse_grain`

- `neural_dynamic_system/training.py`
  - `_vamp2_score`
  - `_koopman_consistency_loss`
  - `_q_align_loss`
  - `_semigroup_loss`
  - `_loss_bundle`

- `neural_dynamic_system/cli.py`
  - `train` 子命令的训练实现
  - summary 导出
  - koopman / q / h / z probe

- `neural_dynamic_system/app.py`
  - 根入口
  - `train / plot / suite` 子命令分发

- `neural_dynamic_system/run_config.py`
  - YAML / JSON 启动配置读写
  - 分组参数导出
  - ablation 运行复用

- `neural_dynamic_system/suite_cli.py`
  - van der Pol ablation suite
  - 每个 run 自动导出自己的 `run_config.yaml`

- `neural_dynamic_system/plot_cli.py`
  - 已完成 run 的单独出图入口

- `neural_dynamic_system/plots.py`
  - 单次 run 出图
  - 多 run 对比图
  - 把画图逻辑和指标汇总逻辑拆开

- `neural_dynamic_system/run_artifacts.py`
  - 已保存 run 的模型 / 配置 / synthetic 标签回载

- `neural_dynamic_system/synthetic.py`
  - 当前只保留 van der Pol synthetic 实验面
  - 统一生成 trajectory / label / probe label

## 运行方式

当前默认的 synthetic 实验面只保留 van der Pol。

一个小的 train smoke test：

```bash
python -m neural_dynamic_system train \
  --num_episodes 2 \
  --steps 256 \
  --obs_dim 6 \
  --window 16 \
  --q_dim 1 \
  --h_dim 1 \
  --koopman_dim 4 \
  --batch_size 32 \
  --epochs 2 \
  --horizons 1 2 \
  --out_dir runs/neural_dynamic_system/demo
```

也可以直接跑脚本包装：

```bash
python scripts/run_neural_dynamic_system.py --num_episodes 2 --steps 256
```

也可以把参数保存成 YAML 后一键运行：

```bash
python scripts/run_neural_dynamic_system.py --config configs/examples/van_demo.yaml
```

现在每次训练都会把最终生效的启动参数保存为 `run_config.yaml`，方便你做消融时直接复制一份、改少量字段再重跑。

默认 help 现在只显示常用参数；如果要看完整参数面，可以用：

```bash
python scripts/run_neural_dynamic_system.py --help-expert
```

phase / curriculum 的细粒度阈值已经从主 CLI 下沉到 YAML 里，推荐直接在 `configs/examples/van_demo.yaml` 或自动导出的 `run_config.yaml` 里改。

跑 van der Pol ablation suite：

```bash
python -m neural_dynamic_system suite --mode all
```

或者继续用脚本包装：

```bash
python scripts/vdp_suite.py --mode all
```

对已完成 run 单独补图：

```bash
python -m neural_dynamic_system plot --run_dir runs/neural_dynamic_system/demo
```

## 新增和重要参数

- `--koopman_dim`
  - 显式 Koopman 特征维度
  - 如果不传，代码会默认取 `max(q_dim, modal_dim)`

- `--hidden_rank`
  - hidden low-rank operator 的秩

- `--rg_temperature`
  - RG soft mask 的温度

- `--q_dim`
  - 取 Koopman 特征前多少维作为 slow subspace

- `--h_dim`
  - hidden fast state 维度

- `--latent_scheme`
  - `hard_split` 或 `soft_spectrum`

## 输出文件

当前会导出：

- `model.pt`
- `history.csv`
- `run_config.yaml`
- `config.json`
- `summary.json`
- `trajectory_preview.csv`
- 合成数据时的 `synthetic_hidden_state.csv`
- probe 结果

如果跑的是 `scripts/vdp_suite.py`，还会额外保存：

- `study_config.json`
- `study_config.yaml`
- 每个子 run 目录下各自的 `run_config.yaml`
- `plot` 子命令重新导出的图目录

现在 probe 会同时评估：

- `koopman`
- `q`
- `h`
- `z`

## 当前实现的取舍

1. 保留显式 Koopman 特征头  
   因为如果没有 `phi`，Koopman 那部分就只剩文案，没有明确对象。

2. `q` 不走 black-box neural ODE  
   因为我们更需要一个固定步长、明确的 `F_dt`。

3. `h` 不走普通 RK2  
   因为它本来就是 affine hidden SSM，更适合 exact step。

4. RG 变换只在 RG 分支里用  
   因为它是尺度比较坐标，不是主模型本体坐标。

5. 当前 hidden operator 先做成 DPLR-like 结构  
   这样比完全 dense 的 `A(q)` 更适合后续 SSD 化。

## 局限

- 还没有 fused Mamba kernel。
- `h` 虽然已经 SSD-ready，但当前离散化还是 dense `matrix_exp`。
- RG 仍然是一个 latent-level coarse-grain consistency prior，不是完整的 RG 理论程序。
- 还没有涨落-耗散或随机项。

## TODO / 后续工作

- 加入更直接的 Koopman spectrum regularizer。
  当前已经有 VAMP、time-lag diagonalization 和 modal decay consistency，但还缺少更显式的谱级约束，例如 spectral gap、慢模态簇稳定性，或对 Koopman 线性算子谱结构的直接正则。
- 面向跨系统泛化的谱建模。
  当前 slow-side Koopman rates 仍是全局共享参数；如果后续要覆盖参数变化较大或机理不完全一致的系统族，更合理的方向是共享谱模板加条件化偏移，或只共享相对谱排序 / 谱簇结构，而不是硬共享一套绝对谱值。
- 探索 `q/h` 有效维度自适应。
  当前 `q_dim` 和 `h_dim` 仍由人工指定；后续可以考虑基于门控、稀疏化、ARD、谱簇截断或其他容量控制机制，让 slow / fast 子空间更自动地贴近真实有效维度，而不是完全依赖手工调参。
- 借鉴 MHC 风格的维度自动对齐设计。
  如果后续实验继续表明 `q_dim / h_dim` 对结果较敏感，可以考虑把 slow / fast 子空间先设为容量上限，再通过可学习门控、活跃维度选择或簇级对齐，让模型更自动地贴近系统的真实有效维度，而不是只靠手动搜索超参数。

## 当前版本的最终描述

> 当前仓库实现的是一个显式 Koopman + Mori-Zwanzig + RG 风格的 `q/h` 慢快状态空间模型。`phi` 是显式 Koopman 特征，`q` 是其 slow 子空间，`h` 是 affine hidden SSM memory state；`q` 用 midpoint 推进，`h` 用 exact affine / exponential step 推进；RG 通过专门的谱归一化坐标和 coarse-grain 映射只在 RG 分支里约束。这个结构在理论上比旧版 `q,m` 文案更贴近当前代码，在工程上也更接近后续 SSD/Mamba 化的方向。
