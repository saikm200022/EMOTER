import datetime
from models import SmileDetector
from utils import load_data
from tqdm import tqdm
import torch

def train(lr = 1e-4, batch_size = 128, epochs = 20, pre_trained = None):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmileDetector()
    model.to(device)
    if pre_trained != None:
        model = load_model(pre_trained)

    loss_f = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-6)

    train_data = load_data('./data/SMILEs', batch_size=batch_size)

    for epoch in range(epochs):
        epoch_loss = 0
        n = 0
        for images, labels in tqdm(train_data):
            n += 1
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = loss_f(output, labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            epoch_loss = epoch_loss + (loss.item() - epoch_loss) / n
        print("EPOCH ", epoch, " LOSS :", loss.item())
        if epoch % 4 == 0:
            save_model(model, "RES" + str(epoch) + ".th")
            print("ACCURACY: " , assess_accuracy(model))


def save_model(model, file_name):
    from torch import save
    from os import path
    print("saving", file_name)
    return save(model.state_dict(), file_name)

def load_model(file):
    from torch import load
    from os import path
    gen = SmileDetector()
    gen.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), file), map_location='cpu'))
    return gen

def assess_accuracy(model):
    train_data = load_data('./data/SMILEs')
    accuracies = []
    total = 0
    positives = 0
    negatives = 0
    for images, labels in tqdm(train_data):
        output = model(images)
        positives += labels.sum().item()
        negatives += labels.size(0) - labels.sum().item()
        total += labels.size(0)
        accuracy = (output.argmax(1) == labels).float().mean().item()
        accuracies.append(accuracy)
    
    print("POSITIVIES (SMILING): ", positives, "NEGATIVES (NOT SMILING): ", negatives)
    return torch.FloatTensor(accuracies).mean().item()

if __name__ == '__main__':
    train(lr = 1e-4, batch_size = 128, epochs = 21)