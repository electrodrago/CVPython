import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # Initialize the list of weights matrices, the store the
        # network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # Starting looping from the index of the first layer but
        # stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # Randomly initialize a weight matrix connecting the
            # number of nodes in each respective layer together
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

            # The last two layers are a special case where the input
            # connections need a bias term but the output does not
            w = np.random.randn(layers[-2] + 1, layers[-1])
            self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # Construct and return a string that represents the network
        # architecture
        return "Neural Network: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # Compute and return sigmoid activation value for a
        # given input value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display_update=100):
        # Insert a column of 1's as the last entry in the feature
        # matrix -- this little trick allows us to treat the bias
        # as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # Loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # Loop over each individual data point and train
            # the network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # Check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        # Construct our list of output activations for each layer
        # as our data point flows through the network; the first
        # activation is a special case -- it's just the input
        # feature vector itself
        A = [np.atleast_2d(x)]

        # FEEDFORWARD
        # Loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # Feed forward the activation at the current layer by
            # taking the dot product between activation and the
            # weight matrix -- this is called the "net input"
            # to the current layer
            net = A[layer].dot(self.W[layer])

            # Computing the "net output" is simply applying our
            # nonlinear activation function to the net output
            out = self.sigmoid(net)

            # Once we have the net output, add it to our list of
            # activations
            A.append(out)

        # BACKPROPAGATION
        # The first phase of backpropagation is to compute the
        # difference between our *prediction* (the final output
        # activation in the activation list) and the true target
        # value
        error = A[-1] - y

        # From here, we need to apply the chain rule and build our
        # list of deltas 'D'; the first entry in the deltas is
        # simply the error of the output layer times the derivative
        # of our activation function for the output value
        D = [error * self.sigmoid_derivative(A[-1])]

        # Loop oer the layers in reverse order (ignoring the last
        # two layer since already use it) -- chain rule
        for layer in np.arange(len(A) - 2, 0, -1):
            # The delta for the current layer is equal to the delta
            # of the *previous layer* dotted with the weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivative of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_derivative(A[layer])
            D.append(delta)

        # Reverse the deltas as we looped over the reverse order
        D = D[::-1]

        # WEIGHT UPDATE
        # Loop over the layers
        for layer in np.arange(0, len(self.W)):
            # Update our weights by taking the dot product of the layer
            # activations with their respective deltas, then multiplying
            # this value by some small learning rate and adding to our
            # weight matrix -- this is where the actual "learning" take
            # place
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss


# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])
#
# nn = NeuralNetwork([2, 2, 1], alpha=0.5)
# nn.fit(X, y, epochs=2000)
#
# for (x, target) in zip(X, y):
#     pred = nn.predict(x)[0][0]
#     step = 1 if pred > 0.5 else 0
#     print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))
#
# print(nn.W)

