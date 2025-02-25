import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class TorchMultiplicationNetwork:
    def __init__(self):
        self.device = self.get_gpu()
        model = self.create_model()
        
        # initialize model weights
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
        self.model = model.to(self.device)
        self.optimizer, self.criterion = self.create_optimizer(self.model)

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        return model
    
    def create_optimizer(self, model):
        optimizer = optim.Adam(model.parameters(), lr=0.00002)
        criterion = nn.MSELoss()
        return optimizer, criterion

    def train(self, input_data, target, epoch):
        all_loss = 0
        
        for e in tqdm(range(epoch)):
            epoch_loss = 0
            for i in range(len(input_data)):
                self.optimizer.zero_grad()

                input_i = torch.unsqueeze(input_data[i], 0)
                output_i = self.model(input_i)
                target_i = torch.unsqueeze(target[i], 0)

                loss = self.criterion(output_i, target_i)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(input_data)
            all_loss += epoch_loss
            if (e + 1) % 100 == 0:
                print(f"Epoch [{e+1}/{epoch}], Loss: {epoch_loss/len(input_data):.4f}")
        return all_loss/epoch

    def save_model(self, path, epoch: float, loss: float):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }, path)

    def test(self):
        print("      ", end="")
        for j in range(9):
            print(j+1, end="     ")
        print()
        for i in range(9):
            print(i+1, end=" ")
            for j in range(9):
                input_i = torch.tensor([i+1, j+1], dtype=torch.float32).unsqueeze(0)
                output = self.model(input_i)
                output_round = round(output.item(), 8)
                output_str = str(output_round)[:5]
                print(output_str, end=" ")
            print()

    def get_gpu(self, use_gpu=True):
        if not use_gpu:
            device = torch.device('cpu')
            print("Using CPU")
            return device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device

    def create_training_data(self):
        input_data = [(a+1, b+1) for a in range(9) for b in range(9)]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        
        output_data = [x[0] * x[1] for x in input_data]
        output_tensor = torch.tensor(output_data, dtype=torch.float32).to(self.device)
        output_tensor = torch.unsqueeze(output_tensor, 1)
        
        print(input_tensor.shape, output_tensor.shape)
        return input_tensor, output_tensor
