# Prefix Sampling: targeting 50% rollout pass rate for more efficient agentic RL

Tianshu Zhu, Wenyu Zhang, Lun Tian, Haotian Zhao, Ruijie Xu, Yuxin Zhang, Jingnan Gu*, Daxiang Dong*, Jianmin Wu*

*Corresponding authors

> By Baidu | Work In Progress | Published on TBD

## TL;DR

![SWE-bench Verified Score](figures/qwen3_14b_ps_vs_baseline_1to1_repro_v2.png)

*Figure 1. SWE-bench Verified pass@1 score over training steps. PS reaches the baseline's peak score 1.8x faster and continues improving, ultimately reaching 0.298 vs the baseline's 0.273.*

Reinforcement learning (RL) for coding agents wastes a large fraction of compute on tasks with skewed rollout pass rates.
When the model almost always fails or almost always succeeds, the gradient signal is weak and biased. **50% rollout pass rate maximizes gradient signal** — that's the target.

**Prefix Sampling (PS) is the most direct way to hit that 50% target**: it replays trajectory prefixes to shift each task's effective rollout pass rate toward 50% — the information-theoretic optimum for binary-reward RL. For mostly-failing tasks, PS gives the model a head start from a rare successful trajectory. For mostly-passing tasks, it imposes a handicap from a rare failing trajectory. In both cases, biased signal is converted into balanced, high-information signal.

On Qwen3-14B (SWE-bench, R2E_Gym), PS is **~2.07x faster end-to-end** to reach the same SWE-bench Verified pass@1 score (1.8x fewer steps × 1.15x faster per step), and ultimately achieves a higher final pass@1 score: **0.298** vs **0.273** for baseline — a quality improvement with no trade-off.

## Most RL Tasks Have the Wrong Pass Rate

In our SWE-bench RL training setup (Qwen3-14B, N=8 rollouts per task, batch_size=64), we observe that tasks fall into several categories with very different learning efficiency:

- **All fail** (0/8 rollout pass rate): The model can't solve the task at all. Zero gradient signal. Discarded by rejection sampling.
- **All pass** (8/8 rollout pass rate): The model solves it every time. Zero gradient signal. Also discarded.
- **Heavily skewed** (1/8, 2/8, 6/8, 7/8): Technically nonzero signal, but heavily biased — the few outlier trajectories carry all the information, and the advantage estimates are noisy and inefficient. These tasks *do* contribute to training, but far below their potential.
- **Balanced** (3/8, 4/8, 5/8): The sweet spot. Roughly balanced positive and negative rollouts produce strong, contrastive gradient signal.

The standard mitigation is **rejection sampling**: discard tasks with 0% or 100% rollout pass rate.  
But this only addresses the extreme cases. In our baseline training, only ~26 out of 64 tasks per batch have partial rollout pass rates (the `solve_partial` metric), and many of those are still heavily skewed.  
Result: a large fraction of compute produces weak learning signal. The root cause: most tasks are far from the 50% rollout pass rate that maximizes gradient signal.

**The key insight behind Prefix Sampling:** we cannot rescue all-fail or all-pass tasks, because there is no successful or failing trajectory to reuse.  
But we *can* recycle heavily skewed tasks (for example 1/8 or 7/8 rollout pass rate) by replaying a trajectory prefix to shift effective difficulty.  
For mostly failing tasks, replay a rare successful prefix as a head start, pushing rollout pass rate toward 50%. For mostly passing tasks, replay a rare failing prefix as a handicap, making success non-trivial again.  
In both cases, low-quality biased signal is turned into balanced, information-rich signal.

## Why 50% Pass Rate Is the Optimal Target

We show from first principles that for binary-reward RL, the most informative regime is a balanced rollout pass rate (`p ≈ 0.5`).
We support this from three complementary views: entropy, GRPO advantage variance, and contrastive pair count.

**Reader shortcut (conclusion first):**
- Information (`H(p)`), GRPO signal strength (`p(1-p)`), and contrastive structure (`k(N-k)`) all peak at `p=0.5`.
- Skewed tasks (for example `1/8` or `7/8`) still train, but with substantially lower sample efficiency.
- Prefix Sampling improves efficiency by shifting skewed tasks toward this high-signal regime.

#### Perspective 1: Information Theory

With binary pass/fail feedback, per-rollout information is bounded by the entropy of a Bernoulli random variable:

```
H(p) = -p·log₂(p) - (1-p)·log₂(1-p)
```

where `p` is the pass probability for a task.

Take derivatives:

```
dH/dp = -log₂(p) - 1/ln(2) + log₂(1-p) + 1/ln(2) = log₂((1-p)/p)
```

Set `dH/dp = 0`:

```
log₂((1-p)/p) = 0  →  (1-p)/p = 1  →  p = 0.5
```

Second derivative:

```
d²H/dp² = -1/(p·ln2) - 1/((1-p)·ln2) < 0  for all p ∈ (0, 1)
```

So `H(p)` is uniquely maximized at `p = 0.5`, with `H(0.5)=1` bit (the binary maximum), while `H(0)=H(1)=0`.  
Concrete scale: `H(0.1)≈0.47`, `H(0.01)≈0.08`.  
Implication: skewed rollout pass rates waste information per rollout.

#### Perspective 2: GRPO Gradient Signal Strength

Entropy quantifies available information; GRPO variance quantifies update strength.  
For a task with `N` rollouts, rewards `rᵢ ∈ {0,1}`, `k` passes, and `p = k/N`, mean-centered advantage is:

```
Aᵢ = rᵢ - r̄,  where r̄ = (1/N) Σⱼ rⱼ = k/N
```

Hence:

```
Passing rollouts (rᵢ = 1):  Aᵢ = 1 - k/N = (N-k)/N
Failing rollouts (rᵢ = 0):  Aᵢ = 0 - k/N = -k/N
```

Policy gradient:

```
∇J ∝ Σᵢ Aᵢ · ∇log π(τᵢ)
```

Signal strength is tied to advantage variance:

```
Var(A) = E[A²] - E[A]²
```

Since `E[A]=0`, compute `E[A²]`:

```
E[A²] = p·(1-p)² + (1-p)·p² = p(1-p)·[(1-p) + p] = p(1-p)
```

Therefore:

```
Var(A) = p(1-p)
```

Maximization:

```
d[p(1-p)]/dp = 1 - 2p = 0  →  p = 0.5
d²[p(1-p)]/dp² = -2 < 0  (confirmed maximum)
```

So `Var(A)` is maximized at `p=0.5` with value `0.25`.  
Reference points: `Var(0.1)=0.09`, `Var(0.01)=0.0099`, and for `p=1/8`, `Var=0.109`.  
Implication: as rollout pass rate becomes skewed, GRPO contrast weakens sharply.

#### Why This Also Maximizes Credit Assignment

Balanced rollout pass rates also maximize credit-assignment opportunities.  
With `N` rollouts and `k` successes, the number of success-failure contrastive pairs is:

```
C(k) = k × (N - k)
```

Complete the square:

```
C(k) = k(N-k) = -(k - N/2)² + N²/4
```

This is a downward parabola with vertex at `k = N/2` (equivalently `p=0.5`). For `N=8`:

| k (passes) | Pass rate | C(k) = k(8-k) | Contrastive pairs |
|---|---|---|---|
| 0 | 0% | 0 | No positive examples |
| 1 | 12.5% | 7 | Limited contrast |
| 2 | 25% | 12 | Better but skewed |
| 4 | 50% | 16 | **Maximum contrast** |
| 7 | 87.5% | 7 | Symmetric to k=1 |
| 8 | 100% | 0 | No negative examples |

At `k=4`, there are 16 pairs, more than 2x the 7 pairs at `k=1`.  
Implication: balanced rollout pass rates provide the richest structure for step-level credit assignment.

#### Summary

All three objectives peak at balance: entropy `H(p)`, GRPO variance `p(1-p)`, and contrastive pairs `k(N-k)` are all maximized at `p=0.5`.  
So the training target is not just “nonzero rollout pass rate,” but “maximally informative rollout pass rate.” Prefix Sampling uses replayed prefixes to move skewed tasks toward that regime.

## How Prefix Sampling Steers Tasks to 50%

### Overview

![Prefix Sampling Workflow](figures/workflow.png)

*Figure 2. The Prefix Sampling workflow. Each batch mixes fresh tasks and prefix tasks. Tasks are classified by rollout pass rate into five categories; too-hard and too-easy tasks are recycled via prefix replay into the next batch.*

The figure above shows the complete Prefix Sampling workflow.  
At each training step, a mixed batch is assembled from two sources: fresh dataloader tasks and prefix tasks collected from previous steps.  
Each task runs N rollouts with **prefix replay**. Fresh tasks start from scratch, while prefix tasks start from a restored environment state.  
Based on rollout pass rate, each task follows one of five paths:

**All-Fail (0%)**: All N rollouts fail. No successful trajectory exists to construct a prefix from. These tasks are discarded via rejection sampling.

**All-Pass (100%)**: All N rollouts pass. No failing trajectory exists. Also discarded.

**Normal (30%-70%)**: Pass rate is balanced — some rollouts pass, some fail. These tasks proceed directly to RL training. The gradient signal is already high-quality; no prefix intervention needed.

**Too-Easy (70%-100%)**: Most rollouts pass, but at least one fails. Original rollouts are trained on, but additionally, one failing trajectory is saved and sent to **Prefix & Mask**. Using **prefix selection**, a portion of the failing trajectory is selected as a prefix. Using **adaptive prefix** control, the prefix length is calibrated to target 50% rollout pass rate. The selected prefix is masked during training, while the discarded portion is dropped. This prefix task is injected back into the next batch, where it will be replayed with the failing prefix as a "handicap."

**Too-Hard (0%-30%)**: Most rollouts fail, but at least one passes. Original rollouts are trained on, and one successful trajectory is saved and sent to **Prefix & Mask**. A portion of the successful trajectory becomes the prefix, giving the model a "head start" when replayed in the next batch.

In short:
- **Drop:** all-fail and all-pass tasks (standard rejection sampling).
- **Train directly:** normal tasks (already balanced).
- **Recycle:** too-hard and too-easy tasks via prefix replay to rebalance rollout pass rate.

The key mechanism: **prefix replay** restores environment state from saved trajectories, **prefix selection** determines how much to replay, **prefix masking** excludes prefix tokens from gradients, and **adaptive prefix** control maintains ~50% rollout pass rate. Tasks with skewed rollout pass rates are converted into more informative training samples through this prefix-guided replay loop.

### Prefix Replay

What does it mean to "replay a prefix" in a multi-turn coding agent environment?

Unlike single-turn reasoning tasks where a prefix is simply prepended text, SWE-bench trajectories involve stateful interactions: the agent reads files, edits code, runs tests, and maintains conversation history. Replaying a prefix means **restoring the full environment state** at a specific step in a saved trajectory:

1. **Code state**: All file edits made up to step K are applied to the repository
2. **Conversation history**: The full dialogue (user messages, agent responses, tool outputs) up to step K is loaded
3. **Execution context**: The agent is positioned exactly where the prefix trajectory left off

From this restored state, the model generates new continuations — making its own decisions about what to do next. The prefix is not part of the model's input for training; it's purely the *starting condition* for a new rollout. This is fundamentally different from SFT, where the model would be trained to imitate the prefix actions themselves.

### Prefix Selection

Which trajectory and how much of it to replay?

- **Too hard tasks → Successful prefix:** Replay the first K steps of a rare successful trajectory, giving the model a "head start." This increases the rollout pass rate toward 50%.
- **Too easy tasks → Failing prefix:** Replay the first K steps of a rare failing trajectory, giving the model a "handicap." This decreases the rollout pass rate toward 50%.

The prefix length K is parameterized by a **ratio** and a **cap**:

**Too hard (remaining steps mode):**
```
remaining = min(int(total_steps × remaining_ratio), remaining_cap)
target_step = total_steps - remaining
```
With `remaining_ratio=0.25`, a 20-step trajectory replays 15 steps and the model completes the last 5 on its own.

**Too easy (prefix steps mode):**
```
target_step = min(int(total_steps × prefix_ratio), prefix_cap)
```
With `prefix_ratio=0.25`, a 20-step trajectory replays the first 5 steps of a failing trajectory.

The cap prevents extremely long prefixes in very long trajectories.

In practice, fixed ratios of `prefix_ratio=0.25` and `remaining_ratio=0.25` work reasonably well across different tasks and model capabilities.

### Prefix Masking

A critical design choice: **prefix tokens are excluded from gradient updates.** During training, the response mask is set to zero for all tokens in the prefix region. Only the model's own continuation — the decisions it made after the prefix — receives gradient signal.

Why is this essential? Without masking, the prefix would participate in advantage computation. If the continuation fails, the prefix steps would receive negative advantage — penalizing actions the model didn't choose. If the continuation succeeds, the prefix would receive positive advantage — reinforcing actions from a different trajectory. Either way, the model would be learning to imitate (or avoid) someone else's decisions, which is SFT, not RL.

With masking, the training objective remains pure RL: the model is only rewarded or penalized for its own choices, given a particular starting state.

### Adaptive Prefix (Optional)

*Note: the Qwen3-14B experiments above use fixed ratios — adaptive prefix is not required to achieve the reported speedups.*

Fixed ratios work reasonably well, but the optimal prefix length changes as the model improves. A ratio that produces 50% prefix task rollout pass rate at step 10 may produce 80% at step 100 (the model got better at completing from prefixes).

PS includes an adaptive feedback loop:

1. **Track per-category prefix task rollout pass rates** using exponential moving averages (EMA, α=0.05, ~13.5-step half-life)
2. **Adjust ratios** when the EMA leaves a deadzone around the 0.5 target:
   - If prefix task rollout pass rate > 0.53: increase the ratio (less prefix → harder)
   - If prefix task rollout pass rate < 0.47: decrease the ratio (more prefix → easier)
   - Step size: ±0.05 per adjustment
3. **5-step cooldown** after each adjustment to prevent overshoot from EMA latency

The cooldown is important: because the EMA is smoothed over multiple steps, a ratio change takes several steps to fully reflect in the metric. Without cooldown, the controller would keep pushing in the same direction, overshooting the target.

## Results: Faster Training by Targeting 50%

We compare Prefix Sampling against a baseline on identical infrastructure and hyperparameters. The baseline follows the [DeepSWE](https://www.together.ai/blog/deepswe) training setup — a state-of-the-art RL approach for training coding agents on SWE-bench that uses GRPO with rejection sampling to filter out all-fail and all-pass tasks.

| | Baseline | Prefix Sampling |
|---|---|---|
| Model | Qwen3-14B | Qwen3-14B |
| Task | R2E_Gym_Subset | R2E_Gym_Subset |
| Rollouts per task | 8 | 8 |
| Batch size | 64 | 64 |
| Rejection sampling | Yes (0%, 100%) | Yes (0%, 100%) |
| PS thresholds | — | low=0.3, high=0.7 |
| PS ratios | — | remaining=0.25, prefix=0.25 |

### Training Efficiency: Faster Convergence and Lower Cost

<div style="display: flex; gap: 16px;">
  <img src="figures/pass_rate_comparison.png" style="width: 50%;" alt="Pass Rate Comparison">
  <img src="figures/step_time_comparison.png" style="width: 50%;" alt="Step Time Comparison">
</div>

*Figure 3. Left: training rollout pass rate on no-prefix tasks over steps. The horizontal line marks the baseline's convergence score; PS matches it 1.55x faster and keeps improving. Right: average wall-clock time per training step (seconds).*

Both PS and baseline reach similar peak rollout pass rates (~0.39), confirming that speedup does not require sacrificing final performance.  
In the first figure, the horizontal line marks the baseline's convergence score. PS matches it in **201 steps** vs **312 steps** for baseline, a **1.55x step-efficiency improvement on no-prefix training samples**.  
PS also continues improving beyond that point.

Step count is only part of the story.  
The second figure shows that PS also runs **~1.15x faster per step** (1398s vs 1601s average).  
This may seem counterintuitive because PS adds prefix replay work. But replayed prefix steps are deterministic and need no LLM inference, which is much faster than generating from scratch.  
The inference savings outweigh the overhead of managing the prefix-task queue.

Combined, these two effects yield an overall **~1.78x end-to-end speedup** to reach the same rollout pass rate: 78h vs 139h wall-clock time.

### Higher Quality and Quantity Training Samples

<div style="display: flex; gap: 16px;">
  <img src="figures/rerollout_pass_rate.png" style="width: 50%;" alt="Prefix Task Pass Rate">
  <img src="figures/valid_samples_comparison.png" style="width: 50%;" alt="Valid Samples Comparison">
</div>

*Figure 4. Left: rollout pass rate on prefix tasks only (target: 0.5). Mean = 0.529, std = 0.078 — the prefix length calibration successfully targets the information-theoretic sweet spot. Right: number of valid training tasks per batch (tasks with mixed pass/fail rollouts).*

The efficiency gains come from converting low-quality training signal into balanced, information-rich signal.  
The first figure shows the core evidence: prefix-task rollout pass rate (measured only on tasks with prefix guidance) stays near the 0.5 target throughout training (mean = 0.529, std = 0.078).  
This confirms that prefix-length calibration shifts task difficulty toward the information-theoretic sweet spot.

The second figure shows the quantity payoff.  
The `solve_partial` metric counts tasks per batch with mixed pass/fail rollouts, i.e., tasks that produce gradient signal.  
PS averages **36.5 valid tasks per batch** vs **25.9** for baseline, a **41% increase** in useful training data per step.

But the gain is not only in quantity.  
Tasks with heavily skewed rollout pass rates (e.g., 1/8 or 7/8) that would have contributed weak, biased gradient signal are transformed through prefix replay into balanced partial-pass results near 50% rollout pass rate.  
These samples are *more informative*, because their rollout pass rates concentrate near the information-theoretic optimum instead of marginal extremes.  
That combination, 41% more samples at higher information value, drives the ~1.55x step-efficiency improvement.

### Higher Entropy, Better Exploration

<img src="figures/entropy_comparison.png" style="width: 50%;" alt="Entropy Comparison">

*Figure 5. Output entropy over training steps. PS maintains higher entropy than the baseline before convergence, indicating healthier exploration. Vertical lines mark the convergence points (steps 201 and 312).*

Entropy measures diversity in the model output distribution.  
Higher entropy implies more exploration; collapsing entropy suggests the policy is becoming too deterministic.  
PS maintains higher entropy than baseline before convergence, which is beneficial for RL training because it supports better exploration.  
Both curves later converge to similar entropy levels, indicating PS does not cause premature entropy collapse.  
The vertical lines (steps 201 and 312) show PS reaches the same final state while maintaining healthier exploration along the way.

## Takeaways

The key insight: **50% rollout pass rate maximizes gradient signal** for binary-reward RL. Prefix Sampling operationalizes this — replaying trajectory prefixes to steer skewed tasks toward that target, converting low-quality biased signal into balanced, high-information signal.  
The core mechanism is simple: for skewed tasks, replay trajectory prefixes to push effective rollout pass rate toward 50%, where binary feedback carries maximum information and contrastive credit assignment is strongest.  
PS does not try to rescue completely unsolvable or trivially solved tasks; those are still discarded.  
Instead, it targets the large middle ground of tasks that already produce gradient signal, but do so inefficiently.

**Main contributions:**

1. **Bidirectional prefix mechanism** handles both too-hard tasks (with successful prefixes) and too-easy tasks (with failing prefixes), converting biased signal from both extremes into balanced 50% rollout pass rate samples.
2. **Prefix replay for agentic RL** restores full environment state (code edits, conversation history, execution context) from saved trajectories, enabling true multi-turn agent continuation.
3. **Adaptive prefix control** automatically recalibrates prefix length as the model improves, maintaining the 50% target without manual tuning.
4. **Dynamic on-policy prefix** uses the latest model's self-generated trajectories from the current step, ensuring prefix tasks always reflect the model's current capabilities and avoiding off-policy staleness.

Our experiments on Qwen3-14B with R2E_Gym tasks show that PS matches the baseline's peak score with **1.55x better step-efficiency** and **~1.15x faster wall-clock time per step** — an overall **~1.78x end-to-end speedup**, while continuing to improve beyond the baseline's convergence point. The reported gains in this post come from fixed-ratio PS (`remaining=0.25`, `prefix=0.25`); adaptive control is included as an extensible mechanism.

**Future work:** The current prefix length selection uses fixed ratios or a simple EMA-based adaptive controller. A promising direction is better prefix length selection — for example, learning a model that predicts the optimal prefix length for a given task and current model capability, directly targeting 50% prefix task rollout pass rate with fewer adjustment steps and less overshoot.

## Citation

```bibtex
@misc{zhu2026prefixsampling,
  title = {Prefix Sampling: Agentic RL with Prefix Guidance},
  url = {TBD},
  author = {Tianshu Zhu and Wenyu Zhang and Lun Tian and Haotian Zhao and Ruijie Xu and Jingnan Gu and Daxiang Dong and Jianmin Wu},
  year = {2026},
  month = {Mar},
}
```

🫡 We appreciate you reading. We’re continuing to improve this work and would love to hear your feedback.
