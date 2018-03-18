import torch


class Session:

    CHECKPOINT = 'logs/checkpoints/checkpoint.pt'

    class TrainingContext:

        EPOCHS = 10

        def __init__(self, task, path=None):
            self._task = task
            self._path = path

            self.epoch = None
            self.loss = None

            self._state = {
                'loss': self.loss,
                'epoch': self.epoch
            }

        def __enter__(self):
            if self._path is not None:
                state = Session.load(self._path)
                self.state = state

            map(lambda x: getattr(x, 'train')(x), self._task.readers)

        def __exit__(self, exc_type, exc_val, exc_tb):
            map(lambda x: getattr(x, 'train')(x, False), self._task.readers)
            Session.save(self.state)

        def train(self):
            outputs = self._task.train()

        def evaluate(self):
            outputs = self._task.evaluate()

        @property
        def state(self):
            return {
                'task': self._task.state,
                'context': self._state
            }

        @state.setter
        def state(self, state):
            self._task.state = state['task']
            self._state = state['context']

    class EvaluationContext:

        def __init__(self, task, path=None):
            self._task = task
            self._path = path

        def __enter__(self):
            if self._path is not None:
                state = Session.load(self._path)
                self._task.state = state['task']

            map(lambda x: getattr(x, 'eval')(x), self._task.readers)

        def __exit__(self, exc_type, exc_val, exc_tb):
            map(lambda x: getattr(x, 'eval')(x, False), self._task.readers)

    @staticmethod
    def load(path=None):
        if path is not None:
            return torch.load(path)
        else:
            return None

    @staticmethod
    def save(state):
        torch.save(obj=state, f=Session.CHECKPOINT)

    @staticmethod
    def evaluate(task):
        with Session.EvaluationContext(task) as ec:
            ec.evaluate()

    @staticmethod
    def train(task):
        with Session.TrainingContext(task) as tc:

            for epoch in range(tc.EPOCHS):
                tc.train()

                with Session.EvaluationContext(task) as ec:
                    ec.evaluate()

                tc.epoch = epoch
                Session.save(tc.state)
