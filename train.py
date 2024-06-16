import numpy as np
from sklearn import datasets, metrics, svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from micrograd.nn import MLP

digits = load_digits()
data = digits.data
labels = digits.target

print(data.shape, labels.shape, len(np.unique(labels)))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=False)

print(len(X_train), len(X_test))

model = MLP(data.shape[-1], [10, 10, 10])

print(f"The number of trainable params in {len(model.parameters())}")

epochs = 200

def compute_softmax(model_outputs):
    exp_outputs = [o.exp() for o in model_outputs]
    denominator = sum(exp_outputs)
    probs = [exp_output/denominator for exp_output in exp_outputs]
    return probs

batch_size = 64
learning_rate = 0.01

for epoch in range(epochs):
    print(f"Current epoch:{epoch}")
    avg_loss = []
    for batch_idx in tqdm(range(0, len(X_train), batch_size)):
        batch_X, batch_y = X_train[batch_idx:batch_idx+batch_size], y_train[batch_idx:batch_idx+batch_size]

        model_outputs = [model.forward(image) for image in batch_X]
        model_probs = [compute_softmax(model_output) for model_output in model_outputs]
        loss = sum([-(o_i[y_i].log()) for o_i, y_i in zip(model_probs, batch_y )])
        loss = loss/(len(batch_X))

        model.zero_grad()

        loss.backward()

        model.update_parameters(learning_rate)

        avg_loss.append(loss.data)

    print(np.mean(avg_loss))

    acc = 0
    for batch_idx in tqdm(range(0, len(X_test), batch_size)):
        batch_X, batch_y = X_test[batch_idx:batch_idx+batch_size], y_test[batch_idx:batch_idx+batch_size]

        model_outputs = [model.forward(image) for image in batch_X]
        model_probs = [compute_softmax(model_output) for model_output in model_outputs]
        preds = []
        for model_output in model_outputs:
            probs = []
            for o in model_output:
                probs.append(o.data)
            preds.append(np.argmax(probs))
        acc += np.sum(np.array(preds) == batch_y)
    
    print(f" The accuracy on external dataset {acc/len(X_test)}")
