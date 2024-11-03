import collections
import random
import numpy as np


class Storage:
    """
    经验回放池，储存每次交互的数据，作为一个双头队列，陈旧的数据会逐步被新数据取代，
    如果actor和critic的观测值相同，则储存5元素元组，(state, action, reward, next_state, done)
    如果actor和critic的观测值不同，则储存7元素元组，(actor_state, critic_state, action, reward, next_actor_state, next_critic_state, done)
    会根据要求随机给出batch_size个元组
    """

    def __init__(self, total_length):
        self.tuples_deque = collections.deque(maxlen=total_length)

    def append(self, rl_tuple):
        self.tuples_deque.append(rl_tuple)

    def sample(self, batch_size):
        samples = random.sample(self.tuples_deque, batch_size)
        if len(self.tuples_deque[0]) == 5:
            state, action, reward, next_state, done = zip(*samples)  # un-zipped
            return (
                np.array(state),
                np.array(action),
                np.array(reward),
                np.array(next_state),
                np.array(done),
            )

        elif len(self.tuples_deque[0]) == 7:
            (
                actor_state,
                critic_state,
                action,
                reward,
                next_actor_state,
                next_critic_state,
                done,
            ) = zip(*samples)
            return (
                np.array(actor_state),
                np.array(critic_state),
                np.array(action),
                np.array(reward),
                np.array(next_actor_state),
                np.array(next_critic_state),
                np.array(done),
            )
        else:
            raise ValueError("rl tuples wrong")

    def length(self):
        return len(self.tuples_deque)
