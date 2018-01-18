
class stats():
    def __init__(self, name):
        self.loss = {}
        self.lists = {}
        self.default_name = name

    def __str__(self):
        return str(self.loss[self.default_name])

    def setList(self, name, l):
        self.lists[name] = l

    def setLoss(self, name, loss):
        self.loss[name] = loss

    def setList(self, name, l):
        self.lists[name] = l

    def getLoss(self, name='mse'):
        return self.loss[name]
