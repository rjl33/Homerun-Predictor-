import torch 
import torch.nn as nn
import torch.optim as optim
from Preprocessing_baseball import X, y  #Import data from preprocessing file
import matplotlib.pyplot as plt

class BaseballNet(nn.Module):         #creat a class for our Neural Net
    def __init__(self, input_size):
        super(BaseballNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, 64) #input layer
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(64, 32) #hidden layer
        self.act2 = nn.ReLU()
        self.output = nn.Linear(32, 1)  #output layer (One out put node either homerun or no homerun)
        self.sigmoid = nn.Sigmoid()   #For output layer which is either 0 or 1

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.sigmoid(self.output(x))
        return x

input_size = X.shape[1]
model = BaseballNet(input_size)

loss_fn = nn.BCELoss() #binaery cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10
losses = []


for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        

    losses.append(loss.item())
    print(f'Finished epoch {epoch}, latest loss {loss}')

# compute accuracy 
with torch.no_grad():
    y_pred = model(X)
    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accuracy {accuracy}")

    # L2 Error (norm of prediction error)
    l2_error = torch.norm(y - y_pred, p=2).item()
    print(f"L2 Error: {l2_error:.6f}")


    mse = torch.mean((y - y_pred) ** 2).item()
    print(f"Mean Squared Error: {mse:.6f}")


plt.figure()
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross Entropy Loss')
plt.title('Loss over Epochs using SGD')
plt.legend()
plt.grid(True)
plt.show()

    # compute accuracy 
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# L2 Error (norm of prediction error)
l2_error = torch.norm(y - y_pred, p=2).item()
print(f"L2 Error: {l2_error:.6f}")

# Optional: Mean Squared Error
mse = torch.mean((y - y_pred) ** 2).item()
print(f"Mean Squared Error: {mse:.6f}")