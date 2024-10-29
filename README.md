# VALUE ITERATION ALGORITHM

## AIM
To find an optimal policy for an agent navigating a grid-world with slippery tiles, aiming to reach a goal state while maximizing expected rewards using value iteration algorithm.

## PROBLEM STATEMENT
The problem involves using the Value Iteration algorithm to find the best strategy for an agent in the Frozen Lake environment. The agent must navigate icy terrain, avoid hazards, and reach the goal while optimizing cumulative rewards in an uncertain environment.

## POLICY ITERATION ALGORITHM
 - Policy iteration is a method of computing an optimal MDP policy and its value.
 - It begins with an initial guess for the value function, and iteratively updates it towards the optimal value function, according to the Bellman optimality equation. 
  - The algorithm is guaranteed to converge to the optimal value function, and in the process of doing so, also converges to the optimal policy.

</br>

The algorithm is as follows:
1. Initialize the value function V(s) arbitrarily for all states s.
2. Repeat until convergence:
   - Initialize aaction-value function Q(s, a) arbitrarily for all states s and actions a.
   - For all the states s and all the action a of every state:
     - Update the action-value function Q(s, a) using the Bellman equation.
     - Take the value function V(s) to be the maximum of Q(s, a) over all actions a.
     - Check if the maximum difference between Old V and new V is less than theta.
     - Where theta is a small positive number that determines the accuracy of estimation.
3. If the maximum difference between Old V and new V is greater than theta, then 
    - Update the value function V with the maximum action-value from Q.
    - Go to step 2.
4. The optimal policy can be constructed by taking the argmax of the action-value function Q(s, a) over all actions a.
5. Return the optimal policy and the optimal value function.


### Name: SANJAY S
### Register Number:  212221243002
## Frozen Lake environment
```python
envdesc  = ['SFHF','FHFF','GFHF', 'FFFH']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 8
P = env.env.P
```
## VALUE ITERATION FUNCTION
```python
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q = np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a] += prob*(reward + gamma * V[next_state] * (not done))
      if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
        break
      V = np.max(Q, axis=1)
      pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return V, pi

```
## optimal policy 
![Screenshot 2024-10-29 160510](https://github.com/user-attachments/assets/3ab84076-f86e-409f-9254-d1ae7a8e2e25)


## optimal value function  

![Screenshot 2024-10-29 160523](https://github.com/user-attachments/assets/79c676ba-f511-40e2-afe0-52b482c6f9ab)


## success rate for the optimal policy.
![Screenshot 2024-10-29 160534](https://github.com/user-attachments/assets/1eba3375-3996-41b6-be5c-7f00b71e3782)




## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
