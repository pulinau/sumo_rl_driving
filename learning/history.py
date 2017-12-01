import numpy as np

class History:
  def __init__(self, config):
    self.history = np.zeros([config.history_length, config.state_length], dtype=np.float32)
  
  def add(self, current):
      self.history[:-1] = self.history[1:]
      self.history[-1] = current

  def reset(self):
    self.history *= 0

  def get(self):
    return self.history
