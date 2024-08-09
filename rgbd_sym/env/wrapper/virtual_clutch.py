from rgbd_sym.env.wrapper.base import BaseWrapper


class VirtualClutch(BaseWrapper):
    """ Virtual Clutch """

    def __init__(self,
                 env,
                 start=6,
                 **kwargs):
        super().__init__(env)
        self._start = start
        self._clutch = False  # False for open, True for closed

    def reset(self,):
        self._clutch = False
        _obs = self.env.reset()
        return _obs

    def step(self, action):
        self._clutch = self.unwrapped.timestep >= self._start  # time-dependent clutch
        return self.env.step(action, skip=not self._clutch)

    @property
    def clutch_state(self):
        return self._clutch
