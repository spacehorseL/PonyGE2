class stats():
    def __init__(self, name):
        self.loss = {}
        self.lists = {}
        self.default_name = name

    def __str__(self):
        return str(self.loss[self.default_name])

    def __setitem__(self, key, value):
        self.loss[key] = value

    def __getitem__(self, key):
        return self.loss[key]

    def setList(self, name, l):
        self.lists[name] = l

    def setLoss(self, name, loss):
        self.loss[name] = loss

    def getList(self, name):
        return self.lists[name]

    def getLoss(self, name='mse'):
        return self.loss[name]
