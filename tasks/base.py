
class Task:

    def fit_model(self, *args, **kwargs):
        return NotImplementedError

    def test_model(self, *args, **kwargs):
        return NotImplementedError

    @classmethod
    def assemble(cls, params):
        return NotImplementedError

    @classmethod
    def abstract(cls):
        return True
