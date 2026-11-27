RL with MuJoCo Warp
===================

.. figure:: https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/benchmarks/humanoid/rollout_three_humanoids.webp
   :align: center
   :width: 90%

   Three humanoids simulated by MuJoCo Warp (image: `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`__).

Use this recipe to train an MLP policy with PPO on five classic MuJoCo locomotion tasks. `MuJoCo Warp (MJWarp) <https://github.com/google-deepmind/mujoco_warp>`__ runs the worlds in one RLinf env worker together on an NVIDIA GPU. Each physics substep is one batched call, and a control step repeats that call for the task's frame skip: 5 for half_cheetah, ant, and humanoid, and 4 for hopper and walker2d. This page covers the tasks, installation, launch commands, and converged training results.

Overview
--------

Choose a task and train a low-dimensional policy through RLinf's actor, rollout, and env workers.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      MLP

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      half_cheetah · hopper · walker2d · ant · humanoid

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 1 NVIDIA GPU

| **You'll do:** install → pick a task → launch training → watch ``eval/return``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · an NVIDIA GPU for training; CPU is supported for environment development.

Tasks
~~~~~

Select a task with ``env.train.task_name`` and ``env.eval.task_name``, and match the policy dimensions to the task.

.. list-table::
   :header-rows: 1
   :widths: 24 16 16 44

   * - Task
     - Obs dim
     - Act dim
     - Termination
   * - ``half_cheetah``
     - 17
     - 6
     - Only the episode step limit.
   * - ``hopper``
     - 11
     - 3
     - Unhealthy torso height or angle, or non-finite state.
   * - ``walker2d``
     - 17
     - 6
     - Unhealthy torso height or angle, or non-finite state.
   * - ``ant``
     - 27
     - 8
     - Unhealthy torso height or non-finite state.
   * - ``humanoid``
     - 45
     - 17
     - Unhealthy torso height or non-finite state.

Every world in an env worker shares one batched physics state, so each physics substep advances every world together. Reward, termination, and the initial state come from the task definition that ``task_name`` selects. A new task with the same reward form is another entry in the task table plus its MJCF file. A different observation or reward is a new task definition that the env constructs for that ``task_name``.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Every episode truncates at ``max_episode_steps``. The table is the observation the policy reads and the reward the env returns.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Specification
   * - Observation
     - ``qpos`` without the global x position (and y position for 3D tasks), concatenated with ``qvel``.
   * - Action
     - Continuous actuator controls, clamped to each model's control range.
   * - Reward
     - Root-x velocity plus a task-specific alive bonus, minus ``ctrl_cost_weight * ||ctrl||^2``, multiplied by ``reward_scale`` (default ``1.0``).
   * - Prompt
     - Not used; this is a low-dimensional policy-control recipe.

half_cheetah, hopper, and walker2d use the same reward terms as Gymnasium MuJoCo v4. This recipe terminates hopper and walker2d on torso height, torso angle, or a non-finite state. Gymnasium v4 also clips joint velocities to [-10, 10] and terminates when the remaining state leaves (-100, 100).

Ant's control-cost weight is 5e-4, and its forward velocity is the root x speed. Gymnasium Ant-v4 uses weight 0.5 and the torso center-of-mass speed.

Humanoid's observation is the 45-dimensional ``qpos`` and ``qvel`` vector in the task table. Before ``reward_scale``, its reward is ``1.25 + root_x_velocity - 0.1 * ||ctrl||^2``. Gymnasium Humanoid-v4 observes 376 values, and its reward is ``5.0 + 1.25 * center-of-mass velocity - 0.1 * ||ctrl||^2``.

.. note::

   MJWarp has no PGS solver. On load, a model whose solver is already PGS is switched to CG. Among the shipped assets, only humanoid sets ``solver="PGS"``, so its contact dynamics differ from Gymnasium's MuJoCo CPU backend.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: MLP + PPO

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env mujoco_warp && source .venv/bin/activate

      **2. Launch**

      Start the half-cheetah recipe:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp

      The command loads ``examples/embodiment/config/mujoco_warp_ppo_mlp.yaml``, starts actor, rollout, and env workers on one GPU with 512 training worlds and 16 evaluation worlds, and starts PPO updates of the MLP policy and value head.

      To select another task, override both env task names and the actor dimensions:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp \
             env.train.task_name=humanoid env.eval.task_name=humanoid \
             actor.model.obs_dim=45 actor.model.action_dim=17

The installer pins ``mujoco-warp==3.17.0`` and ``warp-lang==1.16.0``. This verified combination works with CUDA 12 drivers and avoids resolving CUDA 13-only Warp wheels during driverless Docker builds.

Use the task table above for the other dimension pairs. Each rollout produces 512000 transitions. The default micro and global batch size is 256, and the reference runs below use 32000 or 2048 per task. If you change the world count or the rollout length, choose a batch size that divides the resulting transition count and fits your GPU memory.

Visualization and Results
-------------------------

Launch TensorBoard from the repository root:

.. code:: bash

   tensorboard --logdir logs --port 6006

The launch script writes TensorBoard events under ``logs/<date>-<config>/tensorboard``; with a custom ``runner.logger.log_path``, point ``--logdir`` at that directory. Open http://localhost:6006 and select the env, eval, and time metric groups.

Watch ``eval/return`` for evaluation performance and ``episode_len`` for survival time. :doc:`Training metrics <../../reference/metrics>` defines these quantities and their trajectory counts. They are means over completed episodes, not per-step rewards. Evaluation runs the policy deterministically. Humanoid, hopper, walker2d, and ant also receive alive bonuses, so higher returns or longer survival alone do not show faster or more natural locomotion; check the evaluation video.

The training entrypoint runs evaluation and saves checkpoints every 10 iterations by default.

.. dropdown:: Evaluate a saved checkpoint without further training

   Use the evaluation entrypoint and provide the policy dimensions under ``rollout.model``. This example evaluates a humanoid checkpoint and records all 16 evaluation worlds to ``<log_path>/video/eval/seed_42/0.mp4``:

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

   Here, ``<checkpoint>`` is the ``global_step_<N>`` directory saved during training. TensorBoard events in the same directory contain the evaluation metrics. Evaluation samples reset-state IDs in order and advances the sequence across resets, so separate evaluations need not start from identical states. Set ``env.eval.use_fixed_reset_state_ids=true`` to reuse the same initial state for each world when comparing checkpoints.

Reference Experiments
~~~~~~~~~~~~~~~~~~~~~

The following runs train all five tasks with the same PPO + MLP recipe on one A800-SXM4-80GB GPU per run, with 512 training worlds and 16 evaluation worlds. One training iteration collects 1000 control steps per training world (512000 transitions), then applies the PPO updates. Evaluation and video recording run every 10 iterations, and ``max_episode_steps=1000``. The software is Python 3.11.18, PyTorch 2.11.0+cu128, MuJoCo 3.17.0, MuJoCo Warp 3.17.0, and Warp 1.16.0. The environment seed controls reset noise; actor seed 1234 and the other PPO settings stay at their defaults.

Two settings differ per task: the optimizer learning rate and the actor batch size. With the default learning rate 3e-4, humanoid diverges early in training and hopper collapses after about 400 iterations, so these tasks and walker2d train at 3e-5 with a batch size of 2048.

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 14

   * - Task
     - Learning rate
     - Batch size
     - Iterations
   * - ``half_cheetah``
     - 3e-4
     - 32000
     - 400
   * - ``hopper``
     - 3e-5
     - 2048
     - 700
   * - ``walker2d``
     - 3e-5, then 1e-5
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

The next table reports the first evaluation (iteration 10) and the converged result. The final columns give the mean over the last five evaluations before the reported iteration, the best single evaluation, and the final mean episode length. Each evaluation averages over the completed episodes of the 16 evaluation worlds.

.. list-table::
   :header-rows: 1
   :widths: 17 12 15 15 12 28

   * - Task (env seed)
     - Eval return (10)
     - Eval return (final-5)
     - Eval return (peak)
     - Eval length (final)
     - Media
   * - ``half_cheetah`` (42)
     - 196.25
     - 8951.51
     - 9194.95 (400)
     - 1000.00
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/half_cheetah_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/half_cheetah_seed42_iterations_10_400.mp4>`__
   * - ``hopper`` (42)
     - 284.47
     - 2336.49
     - 3286.43 (440)
     - 490.96
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/hopper_seed42_stable_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/hopper_seed42_stable_iterations_10_700.mp4>`__
   * - ``walker2d`` (42)
     - 304.98
     - 4904.81
     - 7029.61 (570)
     - 1000.00
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/walker2d_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/walker2d_seed42_iterations_10_570.mp4>`__
   * - ``ant`` (42)
     - 1003.75
     - 5769.54
     - 7062.29 (460)
     - 1000.00
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/ant_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/ant_seed42_iterations_10_500.mp4>`__
   * - ``humanoid`` (42)
     - 116.35
     - 4752.41
     - 5893.03 (710)
     - 1000.00
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_750.mp4>`__
   * - ``humanoid`` (43)
     - 110.70
     - 4755.76
     - 5048.28 (440)
     - 1000.00
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed43_curves.png>`__

half_cheetah, ant, and humanoid survive the full 1000 steps from early in training; their return growth tracks forward speed. Hopper and walker2d learn to survive first, then to move faster. Use the linked videos to check that the motion is actual locomotion.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_curves.png
   :width: 100%
   :alt: Humanoid training return and episode length over 750 PPO iterations

   Humanoid with environment seed 42: training return, evaluation return, and episode length. Episode length is capped at 1000; the policy survives full episodes from about iteration 150 and then keeps increasing its speed.

The following video compares iteration 10 on the left with iteration 750 on the right. `Download the humanoid comparison (MP4) <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_750.mp4>`__.

.. raw:: html

   <video controls muted playsinline preload="metadata" style="width: 100%; max-width: 960px;">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_750.mp4" type="video/mp4">
   </video>

The video shows the 16 evaluation worlds on each side. At iteration 750 the humanoid walks forward and stays upright for full episodes.

A second humanoid run changes only the environment seed from 42 to 43 and trains for 440 iterations. Both seeds reach full-length episodes with similar returns, so the recipe does not depend on one reset sequence.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_two_seeds.png
   :width: 100%
   :alt: Humanoid evaluation return and length for environment seeds 42 and 43

   Humanoid evaluation with two environment reset seeds. The actor seed and training hyperparameters are unchanged.

Late-Training Value Instability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PPO becomes unstable on these tasks once long episodes push the value targets into the thousands: the value loss spikes by two orders of magnitude within one iteration and the policy collapses with it. Hopper shows this at learning rate 3e-4 after iteration 410, walker2d at 3e-5 after iteration 560 (restarting from a healthy checkpoint at 1e-5 does not prevent it), and half_cheetah at 3e-4 after iteration 400. The `hopper collapse curve <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/hopper_seed42_default_lr3e-4_curves.png>`__ shows a full run through the collapse. Humanoid recovers from these dips on its own. Walker2d and half_cheetah are reported at their converged checkpoints (iterations 570 and 400) before the instability.

When you train past the tables above, keep ``runner.save_interval`` small, watch ``train/critic/value_loss``, and restart from the latest healthy checkpoint with a lower learning rate if the value loss spikes.

Compare with Published Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The returns above use the reward definitions in `Observation and Action`_, which differ from Gymnasium v4 for ant and humanoid, and the runs train for 205-384 million control steps, far more than typical reference implementations. Within these differences, the converged returns land in the same range as established MuJoCo results, such as the `RL Baselines3 Zoo benchmark <https://github.com/DLR-RM/rl-baselines3-zoo>`__ (deterministic evaluation of single runs trained for 1-2 million steps):

.. list-table::
   :header-rows: 1
   :widths: 18 20 16 16 16

   * - Task
     - RLinf (final-5)
     - PPO (Zoo)
     - SAC (Zoo)
     - TD3 (Zoo)
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

Two adjustments matter when reading the humanoid and ant rows. RLinf's humanoid pays an alive bonus of 1.25 per step against Gymnasium's 5.0, so a full-length RLinf episode would score about 3750 higher under the v4 formula; 4752.41 corresponds to roughly 8500 in v4 terms, in line with the off-policy references. RLinf's ant charges ``5e-4 * ||ctrl||^2`` against v4's ``0.5 * ||ctrl||^2``, so the same gait scores higher here.

Reproduce the Experiments
^^^^^^^^^^^^^^^^^^^^^^^^^

Pick a task from the results table and set its dimensions on the command line, and its learning rate, batch size, and iteration count in the recipe YAML (``mujoco_warp_ppo_mlp.yaml``). The environment seed defaults to 42. This example trains humanoid after setting both optimizer learning rates to 3e-5, the batch sizes to 2048, and the runner limits to 750 iterations:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp \
       env.train.task_name=humanoid env.eval.task_name=humanoid \
       actor.model.obs_dim=45 actor.model.action_dim=17 \
       env.eval.video_cfg.save_video=true

Each run creates its TensorBoard events, checkpoints, and evaluation videos under ``runner.logger.log_path``. Video recording adds rendering time; disable it for throughput measurements. The `combined metrics CSV <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/mujoco_warp_metrics.csv>`__ and `run metadata <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/mujoco_warp_results.json>`__ contain the plotted values and software versions.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Validation
     - Result
   * - One-epoch CI recipe
     - Two parallel worlds complete rollout, advantage computation, and one PPO update on one NVIDIA GPU.

Video recording is disabled by default. Enable it for evaluation with ``env.eval.video_cfg.save_video=true``. Evaluation videos are written under the log path's ``video/eval`` directory. The zero-based file counter follows the evaluation order: ``0.mp4`` records iteration 10, ``1.mp4`` records iteration 20, and so on. Each video contains a grid of the 16 evaluation worlds. A world that terminates early resets within the same video, so one video can contain several episodes.

MJWarp still runs the physics when recording. The environment copies each world's state into native MuJoCo and renders RGB frames; the video wrapper writes those frames to MP4. Recording adds device-to-host synchronization and per-world rendering, so leave it off for throughput runs.

The first simulation step JIT-compiles MJWarp kernels and can take longer than later steps.
