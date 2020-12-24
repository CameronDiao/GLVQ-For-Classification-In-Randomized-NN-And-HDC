class BaseClassifier:
    def __init__(self, train_set):
        self.train_set = train_set

    def preprocess(self, dataset):
        return dataset.drop(["clase"], axis=1).values

    def model(self, inputs, labels):
        pass

    def train(self):
        pass

    def score(self, test_set):
        pass