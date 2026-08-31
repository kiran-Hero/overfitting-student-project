import numpy as np
data = np.genfromtxt(
  "dataset/student_dta.csv",
  delimer=",",
  skip_header=1
)
x=data[:,0:4]
y=data[:,4].reshape(-1,1)
print("number of students:",len(x))
print("number of input feautres:",x.shape[1])
np.random.seed(10)
indices = np.random.permutation(len(X))
np.random.seed(10)
indices = np.random.permutation(len(X))
train_size = int(0.5 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
np.random.seed(10)
W1 = np.random.randn(4, 16) * 0.1
b1 = np.zeros((1, 16))
W2 = np.random.randn(16, 8) * 0.1
b2 = np.zeros((1, 8))
W3 = np.random.randn(8, 1) * 0.1
b3 = np.zeros((1, 1))
W1 = np.random.randn(4, 16) * 0.1
W2 = np.random.randn(16, 8) * 0.1
W3 = np.random.randn(8, 1) * 0.1
def forward(X):
    z1 = X @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = np.tanh(z2)
    z3 = a2 @ W3 + b3
    output = sigmoid(z3)
    return a1, a2, output
  a1 = np.tanh(z1)
a2 = np.tanh(z2)
output = sigmoid(z3)
_, _, sample_output = forward(X_train)
print("Sample predictions:")
print(sample_output[:5])
def calculate_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )
    return loss
def train_network(X, y, learning_rate):
    global W1, b1, W2, b2, W3, b3
    a1, a2, output = forward(X)
    dz3 = output - y
    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0, keepdims=True)
    dz2 = (dz3 @ W3.T) * (1 - a2 ** 2)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    dz1 = (dz2 @ W2.T) * (1 - a1 ** 2)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    return calculate_loss(y, output)
epochs = 5000
learning_rate = 0.05
loss_history = []
for epoch in range(epochs):
    loss = train_network(X_train, y_train, learning_rate)
    loss_history.append(loss)
    if epoch % 500 == 0:
        print("Epoch:", epoch, "Loss:", round(loss, 4))
def predict(X):
    _, _, output = forward(X)
    return (output >= 0.5).astype(int)
def calculate_accuracy(X, y):
    predictions = predict(X)
    accuracy = np.mean(predictions == y)
    return accuracy * 100
train_accuracy = calculate_accuracy(X_train, y_train)
test_accuracy = calculate_accuracy(X_test, y_test)
print("\nFinal Results")
print("Training Accuracy:", round(train_accuracy, 2), "%")
print("Unseen Test Accuracy:", round(test_accuracy, 2), "%")
difference = train_accuracy - test_accuracy
print("Accuracy Difference:", round(difference, 2), "%")
if difference > 15:
    print("Result: The model shows signs of overfitting.")
else:
    print("Result: Strong overfitting is not observed.")
