import numpy as np 
import pandas as pd 

class QLearningTable:
	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greeedy=0.9):
		#初始化超参数
		self.actions = actions #动作空间
		self.lr = learning_rate	#学习率
		self.gamma = reward_decay #折扣
		self.epsilon = e_greeedy #
		self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) #初始化列索引为actions的数据结构

	def choose_action(self, observation):
		#判断当前observation在是否在Q表中
		self.check_state_exist(observation)

		#小于阈值则选择最大值， 利用
		if np.random.uniform()<self.epsilon:
			state_action = self.q_table.loc[observation, :]

			#如果存在多个相同值，则随机选择一个，而不是总是选择第一个
			action = np.random.choice(state_action[state_action==np.max(state_action)].index)
		else:
			#否则随机选择动作， 探索
			action = np.random.choice(self.actions)
		return action

	def learn(self, s, a, r, s_):
		#判断s_是否在Q表中
		self.check_state_exist(s_)
		#预测值为当前最优值
		q_predict = self.q_table.loc[s, a]
		#更新目标值
		if s_ != 'terminal':
			#q = r + γ*max(q`)
			q_target = r + self.gamma*self.q_table.loc[s_, :].max()
		else:
			q_target = r
		#q = q + α*(q' - q)
		self.q_table.loc[s,a] += self.lr*(q_target - q_predict)

	def check_state_exist(self, state):
		#若state不再Q表中，则加入
		if state not in self.q_table.index:
			self.q_table = self.q_table.append(
				pd.Series(
					[0]*len(self.actions),
					index=self.q_table.columns,
					name=state,
				)
			)