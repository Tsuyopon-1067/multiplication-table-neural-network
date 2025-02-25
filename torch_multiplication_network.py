import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class TorchMultiplicationNetwork:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.device = self._get_device(False)
        model = self.create_model()
        
        # initialize model weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
        self.model = model.to(self.device)
        self.optimizer, self.criterion = self.create_optimizer(self.model)

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        return model
    
    def create_optimizer(self, model):
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        return optimizer, criterion

    def train(self, input_data, target, epoch):
        all_loss = 0
        dataset_size = len(input_data)
        
        for e in tqdm(range(epoch)):
            epoch_loss = 0
            indices = torch.randperm(dataset_size)

            for start_idx in range(0, dataset_size, self.batch_size):
                batch_indices = indices[start_idx:min(start_idx + self.batch_size, dataset_size)]
                batch_input = input_data[batch_indices]
                batch_target = target[batch_indices]

                self.optimizer.zero_grad()

                outputs = self.model(batch_input)
                loss = self.criterion(outputs, batch_target)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_indices)

            avg_epoch_loss = epoch_loss / dataset_size
            all_loss += avg_epoch_loss
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
                input_i = torch.tensor([i+1, j+1], dtype=torch.float32).unsqueeze(0).to(self.device)
                output = self.model(input_i)
                output_round = round(output.item(), 8)
                output_str = str(output_round)[:5]
                print(output_str, end=" ")
            print()

    def _get_device(self, use_gpu=True):
        if not use_gpu:
            print("Using CPU")
            return torch.device('cpu')

        if torch.cuda.is_available():
            print("Using CUDA")
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            print("Using MPS")
            return torch.device('mps')
        else:
            print("Using CPU")
            return torch.device('cpu')

    def create_training_data(self):
        input_data = [(a+1, b+1) for a in range(9) for b in range(9)]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        
        output_data = [x[0] * x[1] for x in input_data]
        output_tensor = torch.tensor(output_data, dtype=torch.float32).to(self.device)
        output_tensor = torch.unsqueeze(output_tensor, 1)
        
        print(input_tensor.shape, output_tensor.shape)
        return input_tensor, output_tensor
