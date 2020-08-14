from env import ArmEnv
from rl import DDPG

MAX_EPISODES = 500
MAX_EP_STEPS = 200

env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

rl = DDPG(a_dim, s_dim, a_bound)


for i in range(MAX_EPISODES):
	
	s = env.reset()

	for j in range(MAX_EP_STEPS):

		a = rl.choose_actions(s)

		s_, r, done = env.step(a)

		rl.store_transition(s, a, r, s_)

		if rl.memory_full:
			rl.learn()

		s = s_
