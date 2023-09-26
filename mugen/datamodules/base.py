from torch.utils.data import Dataset

class BaseDataModule:
    def get_training_dataset(self) -> Dataset:
        raise NotImplementedError
    
    def get_validation_dataset(self) -> Dataset:
        raise NotImplementedError
