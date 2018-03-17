
class Session:

    class TrainingContext:

        def __init__(self, task):
            self._task = task

        def __enter__(self):
            pass

        def __exit__(self):
            pass

        def fit(self):
            pass

    def __init__(self, task):

        self._task = task

    def load(self):
        pass

    def start(self):

        with Session.TrainingContext(self._task) as tc:
            tc.fit()
