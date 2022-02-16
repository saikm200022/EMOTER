from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt


class SmilesDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []

        # Negative examples
        with open(dataset_path + '/negatives/smiles_01_neg.idx') as file:
            for line in file:
                image = Image.open(dataset_path + '/negatives/' + line.strip())
                transform = transforms.ToTensor()
                tensor = transform(image)

                self.data.append((tensor, 0))
        
        # Positive examples
        dir = os.fsencode(dataset_path + '/positives/positives7/')
        for file in os.listdir(dir):
            image = Image.open(dataset_path + '/positives/positives7/' + (file.strip()).decode("utf-8"))
            transform = transforms.ToTensor()
            tensor = transform(image)

            self.data.append((tensor, 1))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_data(dataset_path, num_workers=0, batch_size=64):
    dataset = SmilesDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)

def viz_data(dataloader):
    data = next(iter(dataloader))
    f, axarr = plt.subplots(8,8)
    im = 0
    for r in range(8):
        for c in range(8):
            axarr[r][c].axis('off')
            axarr[r,c].imshow(data[0][im].squeeze(), cmap = 'gray')
            im += 1
    plt.show()  

obj = SmilesDataset('./data/SMILEs')
assert len(obj) > 0, "Dataset is empty!"