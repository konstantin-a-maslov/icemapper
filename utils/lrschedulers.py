import tensorflow as tf
import numpy as np


class LRRestartsWithCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, start, restart_steps, verbose=0):
        super(LRRestartsWithCosineDecay, self).__init__()
        self.steps = 0
        self.start = start
        self.restart_steps = [0] + restart_steps
        self.stage = 0
        self.verbose = verbose

    def on_batch_end(self, batch, logs=None):
        self.steps = self.steps + 1

    def on_batch_begin(self, batch, logs=None):
        if self.stage >= len(self.restart_steps):
            return

        next_stage_steps = self.restart_steps[self.stage]
        if self.steps < next_stage_steps:
            current_stage_steps = self.restart_steps[self.stage - 1]
            lr = 0.5 * self.start * (1 + np.cos(np.pi * (self.steps - current_stage_steps) / (next_stage_steps - current_stage_steps)))
        else:
            self.stage += 1
            lr = self.start
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print(f"\nLRRestartsWithCosineDecay callback: set learning rate to {lr}")
