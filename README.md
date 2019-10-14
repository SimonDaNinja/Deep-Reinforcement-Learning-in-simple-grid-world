This is basically just an implementation of deep reinforcement learning, using experience replay, for teaching an agent to navigate to a goal in a grid world with no obstacles... as of yet.

Note that the fact that little change happens in number of steps during the first 200 iterations isn't necessarilly due to lack of learning, but high epsilon. This is necessary, as a low epsilon this early will result in much, much slower learning.
