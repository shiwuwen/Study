import numpy as np
import pyglet 

class ArmEnv(object):

	viewer = None
	dt = 0.1
	action_bound = [-1, 1]
	goal = {'x':100, 'y':100, 'l':40}
	state_dim = 9
	action_dim = 2


	def __init__(self):
		#np.zeros(size, dtype) 其中2生成两个数据，dtype=[(a),(b)]使得每个数据包含两个元素(l, r)
		self.arm_info = np.zeros(2, dtype=[('l', np.float32), ('r', np.float32)])
		self.arm_info['l'] = 100
		self.arm_info['r'] = np.pi/6

		self.on_goal = 0

	def step(self, action):
		done = False
		r = 0.
		action = np.clip(action, *self.action_bound)
		self.arm_info['r'] += action * self.dt
		self.arm_info['r'] %= np.pi * 2    # normalize

		# state
		s = self.arm_info['r']

		(a1l, a2l) = self.arm_info['l']  # radius, arm length
		(a1r, a2r) = self.arm_info['r']  # radian, angle
		a1xy = np.array([200., 200.])    # a1 start (x0, y0)
		a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
		finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

		# normalize features
		dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
		dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
		r = -np.sqrt(dist2[0]**2+dist2[1]**2)

		# done and reward
		if (self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2
		) and (self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2):
				r += 1.
				self.on_goal += 1
				if self.on_goal > 50:
					done = True
		else:
			self.on_goal = 0
		
		# state
		s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))

		return s, r, done

	def reset(self):
		self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
		self.on_goal = 0
		(a1l, a2l) = self.arm_info['l']  # radius, arm length
		(a1r, a2r) = self.arm_info['r']  # radian, angle
		a1xy = np.array([200., 200.])  # a1 start (x0, y0)
		a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
		finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
		# normalize features
		dist1 = [(self.goal['x'] - a1xy_[0])/400, (self.goal['y'] - a1xy_[1])/400]
		dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
		# state
		s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
		return s


	def render(self):
		if self.viewer is None:
			self.viewer = Viewer(self.arm_info, self.goal)
		self.viewer.render()

	def sample_action(self):
		return np.random.rand(2)-0.5	# two radians


class Viewer(pyglet.window.Window):
	'''
	可视化
	'''

	bar_thc = 5

	def __init__(self, arm_info, goal):
		super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
		pyglet.gl.glClearColor(1,1,1,1)
		self.arm_info = arm_info
		self.center_coord = np.array([200, 200])

		self.batch = pyglet.graphics.Batch() #display whole batch at once
		self.point = self.batch.add(
			4, pyglet.gl.GL_QUADS, None,
			('v2f',[goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,	#location
				    goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
				    goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
				    goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
			('c3B', (86,109,249)*4))	#color

		self.arm1 = self.batch.add(
			4, pyglet.gl.GL_QUADS, None,
			('v2f',[250,250,	#location
				    250,300,
				    260,300,
				    260,250]),
			('c3B', (249,86,86)*4))		#color

		self.arm2 = self.batch.add(
			4, pyglet.gl.GL_QUADS,None,
			('v2f',[100,150,
				    100,160,
				    200,160,
				    200,150]),
			('c3B', (249,86,86)*4))

	def render(self):
		self._update_arm()
		self.switch_to()
		self.dispatch_events()
		self.dispatch_event('on_draw')
		self.flip()

	def on_draw(self):
		self.clear()
		self.batch.draw()

	def _update_arm(self):
		(a1l, a2l) = self.arm_info['l']     # radius, arm length
		(a1r, a2r) = self.arm_info['r']     # radian, angle
		a1xy = self.center_coord            # a1 start (x0, y0)
		a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
		a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

		a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
		xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
		xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
		xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
		xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

		xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
		xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
		xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
		xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

		self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
		self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))


if __name__ == '__main__':
	env =ArmEnv()
	i = 0
	while i<5000:
		env.render()
		env.step(env.sample_action())
		i += 1


	# a = np.zeros(2, dtype=[('l', np.float32), ('r', np.float32)])
	# a['r'][0] = 1
	# a['r'][1] = 2
	# print(a)