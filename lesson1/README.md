[Link to the lesson](https://www.youtube.com/watch?v=VMj-3S1tku0&t=2452s) 

The notebook in this repo showcases how backpropagation works to optimise a simple net. You can manually create a graph like [this one](https://github.com/francesco-losciale/nn-backpropagation/blob/main/graph.jpg?raw=true) by doing forward and backward pass and use it a reference for understanding/changing the code. 
Remember that the derivatives can be estimated numerically running the code of f by using `(f(x+h)-f(x))/h`. 


### Notes

Autograd is a automatic gradient engine that implements backpropagation, which is an algorithm that allows to evaluate the gradient of a loss function with respect to the weights of a neural network.
Backpropagation only cares about mathematical expression and we can leverage it so that we can iteratively tune the weights to minimize the loss function and improve the accuracy of the net.

[Micrograd](https://github.com/karpathy/micrograd) allows to build mathematical expressions and use backpropagation. Backpropagation is at the core of PyTorch.

We can identify two phases: 
- forward pass: we move from the inputs and weights through the nodes to calculate the output
- backward pass: we now start from the final node and recursively calculate the gradient/derivative at each step moving back to the inputs of the net. The gradient leverages the chain rule of calculus to calculate the derivatives at each node. They tell us how the values are affecting the overall mathematical expression (loss).

Our goal is to find weights that minimize the result of the loss function and to achieve that we need to re-iterate these two steps after tweaking the weights appropriately. Inputs instead are fixed and since we can't change them (it's the data used for training).

- x0 (input) -> w0 (synapse) -> x0*w0 -> [cell body = sum(wi*xi)+b] -> tanh -> output
- x1         -> w1           -> x1*w1 ->
- ...

tanh it's just an example of a squashing function. b is a constant and it's called the bias.
 
