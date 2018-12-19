def get_optimizer(name):
    name = name.lower()
    if name == 'adam':
        return Adam()
    else:
        raise ValueError('unknown optimizer: ' + name)

class Adam():
    DEFAULT_LEARNING_RATE = 0.0002

    def save(self, path):
        pass

    def load(self, path):
        pass

    def path_for_model(self, model_path):
        return ''
