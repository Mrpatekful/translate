
class Trainer:

    class TrainingContext:

        def __init__(self, task):
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def fit(self):
            pass

    def __init__(self, task):

        self._task = task

    def create_session(self):

        with Trainer.TrainingContext(self._task) as tc:
            tc.fit()
            pass

    def start(self):
        pass
