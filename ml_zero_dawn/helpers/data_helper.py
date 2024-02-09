from torch.utils.data import Dataset, DataLoader

class XyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

class DataHelper:
    def __init__(self, hardaware_helper, X, Y, batch_size):
        print("Initialising the data helper..")
        self.hardaware_helper = hardaware_helper
        self.dataset = XyDataset(X, Y)
        self.batch_size = batch_size
        self.dataset_iter = iter(DataLoader(self.dataset, self.batch_size, shuffle=True))

    def get_batch(self):
        try:
            X, Y = next(self.dataset_iter)
        except StopIteration:
            # print("Restarting dataset iteration..")
            self.dataset_iter = iter(DataLoader(self.dataset, self.batch_size, shuffle=True))
            X, Y = next(self.dataset_iter)
        X, Y =  self.hardaware_helper.send_to_device(X, Y)
        return (X, Y)