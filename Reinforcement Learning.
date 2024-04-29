### RL in a nutshell
![RL process](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/rl-process.jpg)

Agent given some state $s_t$ takes an action $a_0$ according to policy ($\pi(\cdot|s_t)$ or $\mu(s_t)$ ). Action is passed to the environment that returns reward $r_{t+1}$ and next state $s_{t+1}$. 

> [!info]
$\pi$ — stochastic policy sampling from set of actions 
$\mu$ — deterministic policy

### Intro to policies
##### Policy based
Train the policy directly to create a direct $s_t \rightarrow a_t$ mapping. We aim to obtain the optimal policy, often denoted $\pi^*$.
##### Value based
Train the agent to learn how to assign values to states. Knowing the values of states the policy will be to get to the highest values state. This way we aim to obtain the optimal value function.

> [!info]
> **Policy-Value Link**
> $$\pi^*(s) = \rm{argmax}Q^*(s,a)$$

### Bellman Equation
Default value-based reward calculation is repetitive and greedy so very not optimal.

> [!info]
> Value $V$ — cumulative rewards until finish
> Reward $R$ — points that the agent will get for performing action
> 

$$V_\pi = E_\pi[R_{t+1} + \gamma \cdot V_\pi(S_t+1)\mid S_t = s]$$

The main pros here is that we compute the values for previous states and calculate building on these known values.

> [!info]
> Exptected values are linear
> 1. $E = [R_1 + R_2]$
> 2. $E[\alpha R] = \alpha E[R]$
> 3. $E[\alpha_1 R_1 + \alpha_2 R_2] = E[\alpha_1 R_1] + E[\alpha_2 R_2]$

### Learning Strategies 
Monte Carlo vs Temporal Difference Learning

#### Monte Carlo 
Uses entire episode to update the policy.

$$V(S_t) \leftarrow V(S_t) + \alpha[G_t-V(S_t)]$$

Where $\alpha$ is learning rate and $G_t$ is total reward for the episode.
![[Pasted image 20240427233955.png]]

As this is a value-based policy example then $G_0 = R_1 + R_2 + ... + R_{n}$, therefore $G_0=3$, $G_1=2$ etc. so the update for the initial state is $V_{S_0} \leftarrow V_{S_0} + 1 \cdot [3 - V_{S_0}]$. This requires capturing a history of episodes tuples $(S, A, R, S_{t+1})$.
**Problem**: 

#### Temporal Difference 
Uses only the $(S_t, A_t, R_{t+1}, S_{t+1})$ step for learning. This approach does not know future rewards, therefor we can only look at the information from the tuple. 
$$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1}+\gamma V(S_{t+1}) - V(s_t)]$$
$$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1}+\gamma\rm{max_aQ}(S_{t+1}, a) - V(s_t)]$$
This is called TD(0) or one-step TD. For the example from above, after the first action taken we can update the initial state $V(S_0) \leftarrow V(S_0) + \alpha[R_{1}+\gamma V(S_{1}) - V(s_0)]=0+1[1+1\cdot0-0]=1$. We do this one-step update in order to get the estimated return called TD target.
**Problem:** High bias can cause fail to converge. Combination of this bias, off-policy (example: Q-Learning) and using function approximators is called *The Deadly Triad*.

#### Comparison
When comparing on-policy techniques only, both TD and Monte Carlo realize the same:
$$𝑄(𝑆𝑡,𝐴𝑡)=𝑄(𝑆𝑡,𝐴𝑡)+𝛼(𝑋−𝑄(𝑆𝑡,𝐴𝑡))$$
The $X$ is estimated differently in Monte Carlo and TD. 
- Monte Carlo, due to the fact that we know all the rewards following picking this state:
$$X_{MC}=∑_{𝑘=0}^{𝜏−𝑡−1}=𝛾^𝑘𝑅_{𝑡+𝑘+1}$$
- Temporal Differences, doesn't know future rewards, only the current reward and the value of next state. We want to bootstrap the overall reward by looking at next state value (which after some iterations will proxy next state value etc.)
$$X_{TD}=𝑅_{𝑡+1}+𝛾\rm{max}_a𝑄(𝑆_{𝑡+1},𝐴_{𝑡+1})$$

###### Bias
In both cases we are looking for an estimate of value:
$$q(s,a) = \mathbb{E}_{\pi}[\sum_{k=0}^{\tau-t-1} \gamma^kR_{t+k+1}|S_t=s, A_t=a]$$
This is exactly the same as Monte Carlo, hence Monte Carlo is not biased. Temporal Differences is biased with the difference between initialization states (whatever initialization we choose to use, for example zeros), this bias will decay exponentially. ==(WHY EXPONENTIALLY?)==

###### Variance
TD => less components => less variance
Monte Carlo => more stacked states before update => more variance

###### Overall
Monte Carlo is preferred even though variance is higher, as it is more likely to converge.
### Q-Learning
It's 
- Off-policy: different policy is used for acting (inference) and updating (training), for example greedy or epsilon-greedy (on-policy uses same)
- Value-based method: looking for a value function assigning value for each state or state-action pair
- TD approach: temporal differences approach updating values after each step 

> [!info]
> **epsilon-greedy**
>$$
>\rm{\epsilon-greedy} = \begin{cases}
\rm{max }Q_t(A) & \text{with probability } 1-\epsilon,\\
a \sim A  & \text{with probability } \epsilon
\end{cases}
>$$



![[Pasted image 20240428001644.png]]

The Q comes from "quality" (value) of the action given that state. Internally the Q-function is a big (state, action) -> (value) mapping. 

![[Pasted image 20240428003213.png]]

But this is so naive that it can't work in more complicated cases. Mainly due to enormous size the Q-table would have to have.

### Deep Q-Learning (DQN)
Comparison between Q-Learning and Deep Q-Learning:

![Deep Q Learning](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/deep.jpg)

Basically, we want to train a function that, given a state, will estimate quality of choosing each of available actions.
#### Atari games as an example
![Preprocessing](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/preprocessing.jpg)

Preprocessing decisions for higher complexity games can be of great importance. Here we:
1. Move from RGB to Grayscale, $(160,210,3) \rightarrow (160,210,1)$
2. Crop the image to the information dense part, $(160,210,1) \rightarrow (84, 84, 1)$
3. Add temporal information (to overcome Temporal Limitation), by stacking consecutive frames, kind of similar to TD, $(84,84,1)\rightarrow(84,84,4)$

This is put into a model similar to:

![Deep Q Network](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/deep-q-network.jpg)

In DQN we introduce 2 new terms: Q-target and Q-loss.
$$\text{Q-Target}=y_j=r_j+\gamma\rm{max}_{a'}\hat Q(\phi_{j+1},a;\theta^-):=R_{t+1}+\gamma\rm{max}_{a} Q(S_{t+1},a)$$
$$\text{Q-Loss}=y_j-Q(\phi_j,a_j;\theta):=R_{t+1}+\gamma\rm{max}_{a} Q(S_{t+1},a)-Q(S_t, A_t)$$
Where the later are taken from Q-Learning algorithm.
DQN tends to be unstable due to non-linear Q-value function and bootstrapping. To avoid that we introduce 3 stabilizing solutions:
1. Experience Replay to make efficient use of experiences
2. Fixed Q-target to stabilize the training
3. Double DQN to handle the problem of overestimating Q-values

#### Experience Replay
![Experience Replay](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/experience-replay.jpg)

- Sampling from the Experience buffer allows relearning using already known experiences
- Helps in avoiding catastrophic forgetting
- Remove correlation in observation sequences (dunno how), this helps in avoiding oscillating or diverging weights
- Buffer capacity $N$ is a tunable hyperparameter
#### Fixed Q-Target
![Fixed Q-target Pseudocode](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/fixed-q-target-pseudocode.jpg)

If we used the same weights $\theta$ for both estimations — current q and next state q — then the training would be very unstable, as the loss gap we're trying to close would be moving all the time. This is why weights are copied every $T$ steps and we have both $\theta$ and $\theta^-$.
#### Double DQN
Handles the problem of overestimating Q-values by running DQN- and Target- networks.
- **DQN-Network**: select the best action to take for the next state (action with highest Q-value)
- **Target-Network**: calculate the Q-value of taking this action at next state
???? why target-network
==DOPISAC==

