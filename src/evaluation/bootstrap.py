
class Bootstrap():

    def __init__(self, data):
        self.data = data

    def get_data_sample(self):
        return self.data.sample(n=len(self.data), replace=True, axis=0)