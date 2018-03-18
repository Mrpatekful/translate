import torch


class Session:

    class TrainingContext:

        EPOCHS = 10

        def __init__(self, task, path=None):
            self._task = task
            self._path = path

            self._state = {
                'loss': None,
                'epoch': None
            }

        def __enter__(self):
            if self._path is not None:
                state = Session.load(self._path)
                self._task.state = state

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def fit(self):
            outputs = self._task.fit()
            self._task.optimizers.step()

        def eval(self):
            outputs = self._task.evaluate()
            self._task.optimizers.adjust()

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
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    @staticmethod
    def load(path=None):
        if path is not None:
            return torch.load(path)
        else:
            return None

    @staticmethod
    def save(task):
        pass

    @staticmethod
    def evaluation(task):
        with Session.EvaluationContext(task) as ec:
            ec.eval()

    @staticmethod
    def training(task):
        with Session.TrainingContext(task) as tc:

            for epoch in range(tc.EPOCHS):
                tc.fit()

                with Session.EvaluationContext(task) as ec:
                    ec.eval()
