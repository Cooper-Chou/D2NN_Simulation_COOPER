# Defined an single layer perceptron to fit the amplitude and phase response of metasurface
# class ResponseFitting(nn.Module):
#     def __init__(self, M, N):
#         super(ResponseFitting, self).__init__()
#         self.M = M
#         self.N = N
#         self.fc1 = nn.Linear(M, 100)
#         self.fc2 = nn.Linear(100, 100)
#         self.fc3 = nn.Linear(100, N)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x
#
# # Hyperparameters
# learning_rate = 0.003
# epochs = 6
# batch_size = 128
#
# # Initialize network, loss, and optimizer
# model = ResponseFitting(M, N).to(device)
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Compare this snippet from update/inco_train.py:
#
# # Hyperparameters
# learning_rate = 0.003
# epochs = 6
# batch_size = 128
#
# # Initialize network, loss, and optimizer
# model = ResponseFitting(M, N).to(device)
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # Training loop
# for epoch in range(epochs):
#     print("epoch: ", epoch+1)
#     # Training phase
#     model.train()
#     running_loss = 0.0
#     turns = 0
#     for batch_data, batch_labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_data)
#         loss = loss_function(outputs, batch_labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         turns += 1
#         if turns % 100 == 0:
#             print("Finished ", turns, "batches")
#             turns = 0
#
#     avg_train_loss = running_loss / len(train_loader) # 每个 epoch 计算一次
#
#
