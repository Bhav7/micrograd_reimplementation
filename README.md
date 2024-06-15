# Micrograd Reimplementation

This is a reimplementation of Andrej Karpathy's [Micrograd repository](#https://github.com/karpathy/micrograd), a scalar based autograd engine with a PyTorch-like API which can be used to build deep neural networks. Within this implementation, I was able to build up neural networks to faciliate multi-class classification of handwritten digits.


## Simple Training Loop

```python   
from micrograd.nn import Module


data = [[2,3], [1,2], [4,5]]
labels = [0, 1, 2]

model = Module(2, [10, 10, 3])

def compute_softmax(model_outputs):
    exp_outputs = [o.exp() for o in model_outputs]
    denominator = sum(exp_outputs)
    probs = [exp_output/denominator for exp_output in exp_outputs]
    return probs

learning_rate = 0.01
epochs = 200

for epoch in range(epochs):

    model_outputs = [model.forward(x_i) for x_i in data]
    model_probs = [compute_softmax(model_output) for model_output in model_outputs]
    loss = sum([-(o_i[y_i].log()) for o_i, y_i in zip(model_probs, labels)])
    loss = loss/(len(data))

    model.zero_grad()

    loss.backward()

    model.update_parameters(learning_rate)
```