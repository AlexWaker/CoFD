class dataset_split():
    def __init__(self, dataset, num_clients, strategy):
        self.dataset = dataset
        self.num_clients = num_clients
        self.strategy = strategy
        self.result = self.split()
    
    def split(self):
        if self.strategy == 'dilikelei':
            