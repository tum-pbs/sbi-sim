from typing import Dict


def constant_schedule(lr):
    def schedule(i: int):
        return lr

    return schedule

class ReduceLROnPlateau:

    def __init__(self, learning_rate: float, patience: int, factor: float, min_lr: float = 0.0):

        self.learning_rate = learning_rate
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr

        self.best_loss = None
        self.counter = 0

    def __call__(self, *args, **kwargs):
        return self.learning_rate

    def update(self, logs: Dict):

        if 'loss' not in logs:
            return self.learning_rate

        loss = logs['loss']

        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience + 1:

            previous_lr = self.learning_rate
            self.learning_rate *= self.factor
            self.learning_rate = max(self.learning_rate, self.min_lr)

            if self.learning_rate < previous_lr:
                print(f"Reducing learning rate to {self.learning_rate}")

            self.counter = 0

        return self.learning_rate


