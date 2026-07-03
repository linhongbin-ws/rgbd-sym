Research Proposal: Conditionally Equivariant POMDPs for Dynamic and Contact-Rich Environments

1. Abstract / Executive Summary

Equivariant Reinforcement Learning (RL) has significantly improved sample efficiency by embedding geometric inductive biases into neural policies. However, existing Equivariant POMDP frameworks strictly rely on a global and static symmetry assumption (e.g., universal $SE(3)$ equivariance). In real-world robotic tasks, such as contact-rich manipulation, system dynamics and valid symmetries dynamically change or degenerate depending on the physical context (e.g., transition from free space to contact).

To bridge this gap, we propose the Conditionally (or Contextually) Equivariant POMDP. By introducing a context variable $c$ mapped to a context-specific group action $\phi(c) \in G$, we allow the transition, observation, and reward functions to exhibit dynamic symmetries. We theoretically show that Contextual Equivariant POMDPs preserve Value Invariance and Policy Equivariance under local gauge symmetries. Furthermore, we demonstrate how this framework resolves the critical "physics violation" problem in data augmentation for multi-stage robotic tasks like pick-and-place.

2. Motivation and Related Work Limitations

Our work addresses a critical intersection of three existing research domains, filling a significant theoretical and algorithmic gap:

Equivariant RL & POMDPs: Recent advances (e.g., Nguyen et al., CoRL 2023) prove value invariance in POMDPs but assume a single global symmetry group $G$ across the entire state space. This assumption breaks down in multi-modal environments (e.g., when a robot makes contact with a table, $SE(3)$ symmetry degenerates).

Contextual MDPs (cMDPs): While cMDPs model non-stationary dynamics by conditioning on a context variable $c$, they are purely data-driven and lack geometric inductive biases, resulting in high sample complexity.

Gauge/Conditional Equivariance: Found in Geometric Deep Learning (e.g., Gauge CNNs), these methods allow symmetries to depend on local states. However, they have predominantly been applied to static datasets and have not been rigorously integrated into the sequential decision-making and latent dynamics of POMDPs.

Our Contribution: We unify these domains by introducing context-conditioned gauge symmetries into the generative models of POMDPs, enabling "symmetry on demand" for complex robotic tasks.

3. Physical Formulation: Objective Symmetries in POMDPs

We first define the objective physical laws governing the environment using a standard POMDP. Let the true state of the system be $s \in \mathcal{S}$ (e.g., exact 3D poses of the robot and objects). We introduce a context variable $c(s) \in \mathcal{C}$ that captures the current physical mode (e.g., "in free space" vs. "in contact").

We map this context to a specific group action via $\phi: \mathcal{C} \rightarrow G$. Crucially, $\phi(c)$ dictates not only the abstract symmetry group (e.g., $SE(3)$ vs. $SE(2)$) but also its specific mathematical representation and anchor in the state space (i.e., exactly which physical entities are being transformed and around which geometric origin). The underlying physical environment is constrained by the following contextual equivariance properties:

Transition Equivariance:

$$T(s, a, s'; c) = T(\phi(c)s, \phi(c)a, \phi(c)s'; c)$$

Observation Equivariance:

$$O(o | s, a; c) = O(\rho_O(\phi(c))o | \phi(c)s, \phi(c)a; c)$$

(where $\rho_O$ is the group representation in the observation space)

Reward Invariance:

$$R(s, a; c) = R(\phi(c)s, \phi(c)a; c)$$

4. Subjective Inference: Belief-Conditioned MDP and Theoretical Guarantees

The Challenge: While the physical transition $T$ obeys the symmetry $\phi(c(s))$, the agent in a POMDP cannot directly observe $s$, and thus cannot know the true physical context $c(s)$. Therefore, it is impossible for the agent to explicitly apply the correct geometric transformation to its actions or observations in real-time.

The Solution (Belief-Conditioned Gauge Symmetry): To resolve this, we reformulate the problem into a fully observable Belief MDP. The agent maintains a belief state $b_t = P(s_t \mid o_{\le t}, a_{<t})$ residing in a latent belief space $\mathcal{B}$.

We propose a context extractor network $c_\theta: \mathcal{B} \rightarrow \mathcal{C}$ that acts as a "gauge estimator". It infers the valid local gauge symmetry $g_b = \phi(c_\theta(b))$ directly from the subjective belief state, enabling the agent to dynamically anchor its reference frame.

Based on this Belief MDP formulation, we establish two core theorems for dynamic contexts:

Theorem 1 (Contextual Value Invariance): Under the inferred local gauge symmetry $g_b$, the optimal value function and Q-function remain locally invariant:

$$V^*(b) = V^*(g_b \cdot b)$$

$$Q^*(b, a) = Q^*(g_b \cdot b, g_b \cdot a)$$

Theorem 2 (Contextually Equivariant Policy): The corresponding optimal policy is conditionally equivariant to the inferred symmetry:

$$\pi^*(a | b) = \pi^*(g_b \cdot a | g_b \cdot b)$$

5. Practical Application: Pick-and-Place Manipulation

Traditional global $SE(3)$ equivariant RL fails in multi-stage tasks because it applies uniform transformations across all states. This not only leads to physically impossible augmented data during contact (e.g., objects clipping through solid tables) but also confuses the neural network when the same abstract symmetry applies to fundamentally different task stages.

Our framework organically solves this by dynamically shifting the symmetry group, representations, and reference frames based on the inferred context:

Phase 1: Approach (Context $c_1$)

Physical State: Arm moving in unloaded free space towards the target object.

Symmetry Group: Global $SE(3)$ centered around the target object.

Representation / Data Augmentation: The transformation mathematically acts strictly on the relative pose between the robot and the target object. It structurally masks out the irrelevant goal location to preserve the correct task state.

Phase 2: Contact & Grasp (Context $c_2$)

Physical State: Gripper makes physical contact with the object on the table.

Symmetry Group: Degenerated to $SE(2)$ (translation along the XY plane, rotation around the Z-axis).

Mechanism: The context network detects contact (e.g., via force/torque spikes) and strictly disables Pitch and Roll augmentations. This prevents the generation of physically invalid "table-penetration" states, preserving the integrity of the learned Q-values.

Phase 3: Placing to Goal (Context $c_3$)

Physical State: Robot carrying the object to a target receptacle.

Symmetry Representation Shift: Although the abstract symmetry group returns to $SE(3)$ (identical to Phase 1), the mathematical representation of the group acting on the state space fundamentally changes.

Mechanism (The Core Motivation): The geometric anchor shifts from the object to the goal location. The $SE(3)$ transformation operands are now synchronously applied to the "robot+object composite rigid body" and the target Goal, ignoring the object's original resting place.
Furthermore, modulating the context variable explicitly informs the agent of the shifted mass dynamics (unloaded in $c_1$ vs. loaded in $c_3$). This resolves a critical ambiguity in traditional global equivariance: without the context gauge $c$, the Critic network would be forced to map the exact same abstract $SE(3)$ transformations to contradictory underlying physical dynamics, leading to high variance and failure to converge. The context-based local gauge provides the necessary inductive bias to separate these representations.

6. Conclusion and Future Directions

The Conditionally Equivariant POMDP provides a rigorous mathematical foundation for applying geometric inductive biases to non-stationary and contact-rich environments. By allowing the symmetry group to dynamically adapt to the system's physical context, we prevent the collapse of Q-value estimation caused by symmetry breaking.

Future implementation will focus on:

Designing a Mixture-of-Equivariant-Experts (MoEE) actor network guided by the context classifier.

Formulating a Canonical Frame Alignment loss to self-supervise the training of the belief-to-context mapping $c_\theta(b)$.