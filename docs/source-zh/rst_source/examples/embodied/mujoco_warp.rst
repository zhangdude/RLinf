基于 MuJoCo Warp 的强化学习
============================

.. figure:: https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/benchmarks/humanoid/rollout_three_humanoids.webp
   :align: center
   :width: 90%

   MuJoCo Warp 批量仿真的三个 humanoid（图片来源：`MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`__）。

本页介绍如何用 PPO 在五个经典 MuJoCo 运动控制任务上训练 MLP policy，并给出收敛的训练结果。`MuJoCo Warp (MJWarp) <https://github.com/google-deepmind/mujoco_warp>`__ 在 NVIDIA GPU 上把一个 RLinf env worker 里的全部 world 一起推进。每个物理子步是一次批量调用，一个控制步会按该任务的 frame skip 重复这次调用：half_cheetah、ant、humanoid 为 5，hopper、walker2d 为 4。

概述
----

选择任务后，即可通过 RLinf 的 actor、rollout 和 env worker 训练低维 policy。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      MLP

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      half_cheetah · hopper · walker2d · ant · humanoid

   .. grid-item-card:: 硬件
      :text-align: center

      单节点 · 单张 NVIDIA GPU

| **你将完成：** 安装 → 选择任务 → 启动训练 → 观察 ``eval/return``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 训练需要 NVIDIA GPU；环境开发可使用 CPU。

任务
~~~~

通过 ``env.train.task_name`` 和 ``env.eval.task_name`` 选择任务，并为 policy 设置对应维度。

.. list-table::
   :header-rows: 1
   :widths: 24 16 16 44

   * - 任务
     - 观测维度
     - 动作维度
     - 终止条件
   * - ``half_cheetah``
     - 17
     - 6
     - 仅由回合步数上限截断。
   * - ``hopper``
     - 11
     - 3
     - 躯干高度或角度越界，或状态出现非有限值。
   * - ``walker2d``
     - 17
     - 6
     - 躯干高度或角度越界，或状态出现非有限值。
   * - ``ant``
     - 27
     - 8
     - 躯干高度越界，或状态出现非有限值。
   * - ``humanoid``
     - 45
     - 17
     - 躯干高度越界，或状态出现非有限值。

env worker 里所有 world 共享一份批量物理状态，每个物理子步把所有 world 一起推进。奖励、终止条件和初始状态由 ``task_name`` 选中的任务定义给出。想加一个相同奖励形式的任务，只需在任务表里加一条定义和对应的 MJCF 文件；观测或奖励不同，则要为这个 ``task_name`` 写新的任务定义。

观测与动作
~~~~~~~~~~

每个回合在 ``max_episode_steps`` 截断。下表是 policy 读到的观测和环境返回的奖励。

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 字段
     - 定义
   * - 观测
     - ``qpos`` 去掉全局 x 坐标（3D 任务再去掉 y 坐标），与 ``qvel`` 拼接。
   * - 动作
     - 连续执行器控制量，裁剪到各模型的控制范围内。
   * - 奖励
     - 根节点 x 方向速度加上任务特定的存活奖励，减去 ``ctrl_cost_weight * ||ctrl||^2``，再乘以 ``reward_scale``\ （默认 ``1.0``）。
   * - Prompt
     - 不使用，这是低维运动控制配方。

half_cheetah、hopper、walker2d 的奖励项与 Gymnasium MuJoCo v4 相同。本配方在躯干高度、躯干角度或状态非有限时终止 hopper 和 walker2d；Gymnasium v4 还会把关节速度裁剪到 [-10, 10]，并在其余状态超出 (-100, 100) 时终止。

ant 的控制代价权重是 5e-4，前进速度取根节点 x 方向速度；Gymnasium Ant-v4 的权重是 0.5，速度取躯干质心速度。

humanoid 的观测是任务表里的 45 维 ``qpos`` 与 ``qvel`` 向量。未乘 ``reward_scale`` 时奖励为 ``1.25 + root_x_velocity - 0.1 * ||ctrl||^2``；Gymnasium Humanoid-v4 的观测有 376 维，奖励为 ``5.0 + 1.25 * center-of-mass velocity - 0.1 * ||ctrl||^2``。

.. note::

   MJWarp 没有 PGS 求解器。加载模型时，PGS 会被替换为 CG。自带的模型资产里只有 humanoid 声明了 ``solver="PGS"``，所以它的接触动力学与 Gymnasium 的 MuJoCo CPU 后端不同。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: MLP + PPO

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env mujoco_warp && source .venv/bin/activate

      **2. 启动训练**

      启动 half-cheetah 配方：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp

      这条命令加载 ``examples/embodiment/config/mujoco_warp_ppo_mlp.yaml``，在一张 GPU 上启动 actor、rollout 和 env worker（512 个训练环境、16 个评估环境），开始用 PPO 更新 MLP policy 和价值头。

      换任务时，同时覆盖两个环境的任务名和 actor 的维度：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp \
             env.train.task_name=humanoid env.eval.task_name=humanoid \
             actor.model.obs_dim=45 actor.model.action_dim=17

安装脚本固定 ``mujoco-warp==3.17.0`` 和 ``warp-lang==1.16.0``。这个组合在 CUDA 12 驱动上验证过，也能避免无驱动的 Docker 构建解析到仅支持 CUDA 13 的 Warp 安装包。

其他任务的维度见上表。每次 rollout 产生 512000 个样本，默认 micro batch 和 global batch 均为 256，下文参考实验按任务使用 32000 或 2048。修改环境数量或 rollout 长度后，batch 必须整除新的样本数，同时不超过 GPU 显存容量。

可视化与结果
------------

在仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir logs --port 6006

启动脚本把 TensorBoard 事件写到 ``logs/<日期>-<配置名>/tensorboard``；如果你覆盖了 ``runner.logger.log_path``，把 ``--logdir`` 指向那个目录。打开 http://localhost:6006，选择 env、eval、time 指标组。

``eval/return`` 看评估表现，``env/return`` 看训练表现，对应的 ``episode_len`` 看存活时间。:doc:`训练指标 <../../reference/metrics>` 给出了这些量的定义和轨迹计数口径：它们是已完成回合上的均值，不是每步的平均奖励。评估时 policy 按确定性动作执行。humanoid、hopper、walker2d、ant 的奖励含存活奖励，回报更高或活得更久不代表移动更快、步态更自然，要看评估视频。

训练入口默认每 10 个 iteration 评估一次并保存 checkpoint。

.. dropdown:: 不训练，直接评估一个 checkpoint

   用评估入口，并在 ``rollout.model`` 下给出 policy 的维度。下面的例子评估一个 humanoid checkpoint，并把 16 个评估世界录到 ``<log_path>/video/eval/seed_42/0.mp4``：

   .. code:: bash

      python evaluations/eval_embodied_agent.py \
          --config-path examples/embodiment/config \
          --config-name mujoco_warp_ppo_mlp \
          env.eval.task_name=humanoid \
          runner.ckpt_path=<checkpoint>/actor/model_state_dict/full_weights.pt \
          runner.logger.log_path=../results/mujoco_warp_humanoid_eval \
          +rollout.model.model_type=mlp_policy \
          +rollout.model.policy_setup=locomotion \
          +rollout.model.obs_dim=45 +rollout.model.action_dim=17 \
          +rollout.model.num_action_chunks=1 +rollout.model.hidden_dim=256 \
          +rollout.model.add_value_head=true +rollout.model.is_lora=false \
          env.eval.video_cfg.save_video=true

   其中 ``<checkpoint>`` 是训练时保存的 ``global_step_<N>`` 目录，同一目录下的 TensorBoard 事件包含评估指标。评估按顺序取 reset 状态 ID，跨 reset 递进，所以两次评估的起点不必相同。对比 checkpoint 时，可以设 ``env.eval.use_fixed_reset_state_ids=true``，让每个世界每次都用同一个初始状态。

参考实验
~~~~~~~~

以下实验用同一套 PPO + MLP 配方把五个任务训到收敛，每组实验用一张 A800-SXM4-80GB GPU、512 个训练环境、16 个评估环境。一个训练 iteration 为每个训练环境采集 1000 个控制步（共 512000 条样本），然后执行 PPO 更新。每 10 个 iteration 评估一次并录制视频，\ ``max_episode_steps=1000``\ 。

软件版本为 Python 3.11.18、PyTorch 2.11.0+cu128、MuJoCo 3.17.0、MuJoCo Warp 3.17.0、Warp 1.16.0。env 种子控制 reset 噪声；actor 种子固定为 1234，其余 PPO 设置保持默认。

各任务只有两处设置不同：优化器学习率和 actor batch size。默认学习率 3e-4 下，humanoid 在训练初期就会发散，hopper 在约 400 个 iteration 后崩溃，所以这两个任务和 walker2d 都用 3e-5 的学习率、2048 的 batch size。

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 14

   * - 任务
     - 学习率
     - Batch size
     - Iteration 数
   * - ``half_cheetah``
     - 3e-4
     - 32000
     - 400
   * - ``hopper``
     - 3e-5
     - 2048
     - 700
   * - ``walker2d``
     - 3e-5，后改 1e-5
     - 2048
     - 570
   * - ``ant``
     - 3e-4
     - 32000
     - 500
   * - ``humanoid``
     - 3e-5
     - 2048
     - 750

下表给出第一次评估（第 10 个 iteration）和收敛结果。最后三列分别是报告点之前 5 次评估的均值、单次评估峰值和最终平均回合长度。每次评估在 16 个评估环境的已完成回合上取平均。

.. list-table::
   :header-rows: 1
   :widths: 17 12 15 15 12 28

   * - 任务（env 种子）
     - 评估回报（第 10 次）
     - 评估回报（末 5 次均值）
     - 评估回报（峰值）
     - 评估长度（最终）
     - 材料
   * - ``half_cheetah`` (42)
     - 196.25
     - 8951.51
     - 9194.95 (400)
     - 1000.00
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/half_cheetah_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/half_cheetah_seed42_iterations_10_400.mp4>`__
   * - ``hopper`` (42)
     - 284.47
     - 2336.49
     - 3286.43 (440)
     - 490.96
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/hopper_seed42_stable_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/hopper_seed42_stable_iterations_10_700.mp4>`__
   * - ``walker2d`` (42)
     - 304.98
     - 4904.81
     - 7029.61 (570)
     - 1000.00
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/walker2d_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/walker2d_seed42_iterations_10_570.mp4>`__
   * - ``ant`` (42)
     - 1003.75
     - 5769.54
     - 7062.29 (460)
     - 1000.00
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/ant_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/ant_seed42_iterations_10_500.mp4>`__
   * - ``humanoid`` (42)
     - 116.35
     - 4752.41
     - 5893.03 (710)
     - 1000.00
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_750.mp4>`__
   * - ``humanoid`` (43)
     - 110.70
     - 4755.76
     - 5048.28 (440)
     - 1000.00
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed43_curves.png>`__

half_cheetah、ant、humanoid 在训练早期就能存活满 1000 步，回报增长主要来自前进速度；hopper 和 walker2d 先学会不摔倒，再逐渐走快。动作是否真的在行走，要看链接里的视频。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_curves.png
   :width: 100%
   :alt: humanoid 在 750 个 PPO iteration 上的训练回报与回合长度

   env 种子 42 的 humanoid：训练回报、评估回报与回合长度。回合长度上限为 1000；policy 大约在 150 个 iteration 后就能存活整个回合，之后速度持续提高。

下面的视频左边是第 10 个 iteration，右边是第 750 个 iteration。`下载 humanoid 对比视频（MP4） <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_750.mp4>`__。

.. raw:: html

   <video controls muted playsinline preload="metadata" style="width: 100%; max-width: 960px;">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_750.mp4" type="video/mp4">
   </video>

视频两侧各是 16 个评估世界的拼格画面。第 750 个 iteration 时，humanoid 能持续向前行走，整个回合不摔倒。

另一组 humanoid 实验只把 env 种子从 42 换成 43，训练 440 个 iteration。两个种子都能存活整个回合，回报接近，说明配方不依赖某一条 reset 序列。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_two_seeds.png
   :width: 100%
   :alt: env 种子 42 与 43 的 humanoid 评估回报与长度

   两个 env reset 种子下的 humanoid 评估曲线。actor 种子与训练超参数不变。

训练后期的价值函数失稳
^^^^^^^^^^^^^^^^^^^^^^

回合变长、价值目标涨到几千之后，PPO 在这些任务上会变得不稳定：value loss 在一个 iteration 内飙高两个数量级，policy 随之崩溃。hopper 在学习率 3e-4、410 个 iteration 后出现这种情况；walker2d 在 3e-5、560 个 iteration 后出现，从健康的 checkpoint 用 1e-5 重启也压不住；half_cheetah 在 3e-4、400 个 iteration 后出现。完整过程见 `hopper 崩溃曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/hopper_seed42_default_lr3e-4_curves.png>`__。

humanoid 遇到这种下跌能自己恢复。walker2d 和 half_cheetah 报告的是失稳前的收敛 checkpoint（分别为第 570 和第 400 个 iteration）。

超过上表继续训练时，把 ``runner.save_interval`` 调小，盯住 ``train/critic/value_loss``；一旦 value loss 飙高，就从最近一个健康的 checkpoint 用更低的学习率重启。

与已有工作对比
^^^^^^^^^^^^^^

上面的回报用的是“观测与动作”一节里的奖励定义，ant 和 humanoid 与 Gymnasium v4 不同；而且本实验的训练量是 205-384 亿个控制步，远大于常见参考实现。在这些差异之下，收敛回报仍落在已有 MuJoCo 结果的常见区间内。下表对照 `RL Baselines3 Zoo 基准 <https://github.com/DLR-RM/rl-baselines3-zoo>`__\ （单次训练 1 到 2 百万步后做确定性评估）：

.. list-table::
   :header-rows: 1
   :widths: 18 20 16 16 16

   * - 任务
     - RLinf（末 5 次均值）
     - PPO（Zoo）
     - SAC（Zoo）
     - TD3（Zoo）
   * - half_cheetah
     - 8951.51
     - 5819
     - 9535
     - 9656
   * - hopper
     - 2336.49
     - 2410
     - 2326
     - 3606
   * - walker2d
     - 4904.81
     - 3479
     - 3863
     - 4718
   * - ant
     - 5769.54
     - 1327
     - 4616
     - 5813
   * - humanoid
     - 4752.41
     - —
     - 6232
     - 5567

读 humanoid 和 ant 两行时注意两点。RLinf 的 humanoid 每步存活奖励是 1.25，Gymnasium 是 5.0，一个完整回合换算到 v4 公式大约要高 3750；4752.41 约合 v4 口径的 8500，与 off-policy 参考值相当。RLinf 的 ant 控制代价是 ``5e-4 * ||ctrl||^2``，v4 是 ``0.5 * ||ctrl||^2``，同样的步态在这里得分更高。

复现实验
^^^^^^^^

从结果表选一个任务，在命令行给出它的维度，并在配方 YAML（``mujoco_warp_ppo_mlp.yaml``）里设置它的学习率、batch size 和 iteration 数。env 种子默认就是 42。下面的例子先把两个优化器学习率改成 3e-5，batch size 改成 2048，runner 的步数上限改成 750，再训练 humanoid：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp \
       env.train.task_name=humanoid env.eval.task_name=humanoid \
       actor.model.obs_dim=45 actor.model.action_dim=17 \
       env.eval.video_cfg.save_video=true

每组实验的 TensorBoard 事件、checkpoint 和评估视频都写在 ``runner.logger.log_path`` 下。录视频会增加渲染耗时，测吞吐时关掉。`合并的指标 CSV <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/mujoco_warp_metrics.csv>`__ 和 `运行元数据 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/mujoco_warp_results.json>`__ 包含图中的全部数值和软件版本。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 验证项
     - 结果
   * - 单 epoch CI 配方
     - 两个并行环境在一张 NVIDIA GPU 上完成 rollout、advantage 计算和一次 PPO 更新。

评估录视频默认关闭，用 ``env.eval.video_cfg.save_video=true`` 打开。视频写在 ``<log_path>/video/eval/seed_<seed>`` 下，文件名从 0 开始按评估顺序编号：``runner.val_check_interval=10`` 时，``0.mp4`` 是第 10 个 iteration，``1.mp4`` 是第 20 个，依此类推。每个视频是 16 个评估世界的拼格画面；提前结束的世界会在同一个视频里重置，所以一个视频可能包含多个回合。

录视频时物理仿真仍由 MJWarp 执行。环境把每个世界的状态拷贝到原生 MuJoCo 渲染 RGB 帧，再由视频封装写成 MP4。渲染会引入设备到主机的同步和逐世界渲染开销，测吞吐时保持关闭。

第一次仿真步要 JIT 编译 MJWarp 内核，比后续步慢。
