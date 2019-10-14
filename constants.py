UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

AGENT_INDICATOR = 1
GOAL_INDICATOR = -1

# GOAL_REWARD must have a unique value
# the way that training is currently
# implemented

GOAL_REWARD = 1
GOOD_STEP_REWARD = -1
STEP_REWARD = -3
WALL_REWARD = -7

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

# The reason it's good to find ways of
# avoiding this sort of cheating isn't
# just to impress people, but also
# quite practical;
# for this specific example, we may be
# absolutely sure that walking into a
# wall is a bad idea, and that short
# term reduction of distance is a good
# idea. But for more complex tasks,
# it may sometimes be that we should
# let the agent explore strategies that
# seem bad to us at first, as it may find
# advantages we didn't realize ourselves.

# the idea is that if we are right about
# such strategies being bad, then this 
# will still be penalized in the long run,
# teaching the agent to avoid them,
# while if we were wrong, the agent can
# find a good strategy we were unaware of.
