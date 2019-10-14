UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# GOAL_REWARD must have a unique value
# the way that training is currently
# implemented

# Default values are fast to teach
# the agent for this application, but
# almost seem a bit like cheating, 
# since strategies like "don't walk
# into the walls" and "walk in a 
# direction that decreases distance 
# to the goal", are explicitly encouraged,
# rather than emerge naturally from the
# overarching goal of navigating in as
# few steps as possible.

# This is mostly to give a relatively
# quick demonstration of reinforcement learning,
# but you can do it in the more impressive
# way of letting these strategies emerge
# as follows:

# set STEP_REWARD, WALL_REWARD and GOOD_STEP_REWARD
# to the same value (e.g -1). This way, we're not
# cheating.

# Then, be much more conservative with
# decreasing epsilon during training, so that
# the agent keeps exploring more and for longer
# before getting full agency over its behavior.

# This will have the agent learn lessons like
# "don't walk into the wall" on its own more
# easily, instead of us telling it so.

GOAL_REWARD = 1
GOOD_STEP_REWARD = -1
STEP_REWARD = -3
WALL_REWARD = -7

AGENT_INDICATOR = 1
GOAL_INDICATOR = -1
