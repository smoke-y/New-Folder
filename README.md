# Neuron
neuron is described by: $N(x) = f(Wx + B)$, where $f()$ is a non-linear function called activation function, $W$ and $b$ is called weight and bias respectively. These are computed during trainnig.

### Non-linear vs linear function
linear: $f(sx) = sf(x)$<br>
non-linear: $f(sx) \neq sf(x)$

### Why are using non-linear instead of linear?
$z = wx + y$ is the equation of line. This line cannot model curves present in our data. On adding non-linear function, we can now model curves as it is no longer the equation of a line. If we use a linear function, then no matter how many layers of neuron we have in our model, the equation of our model is still a line.

# Loss function
Loss function compares output by our model and the correct answer by outputing an error value. Then we change our model's parameters by "some method" to reduce the loss value. Usually the method is stochastic gradient descent. We use backpropogation to find how much we should change our model's parameters in order to reduce the loss value and then we update our parameters and we keep doing this until our model converges.

### Backpropogation
Forward pass:<br>
$M(x) = f(Wx + B) = f(a + B) = f(d) = y$ <br>
where $f(x) = \frac{1}{1+e^-x}$ also know as the sigmoid function

Loss function:<br>
$MSE(x) = \frac{1}{n}\sum({y* - y})^2 = E$<br>
where $y*$ is truth

Backward pass:<br>
$\frac{\partial E}{\partial y} = -(y*-y)$<br>
$\frac{\partial y}{\partial d} = f(d)(1-f(d))$<br>
$\frac{\partial E}{\partial d} = \frac{\partial E}{\partial y}\frac{\partial y}{\partial d}$<br>
$\frac{\partial d}{\partial w} = x$<br>
$\frac{\partial E}{\partial w} = \frac{\partial E}{\partial d}\frac{\partial d}{\partial w} = \frac{\partial E}{\partial d}x$<br>
$\frac{\partial d}{\partial b} = 1$<br>
$\frac{\partial E}{\partial w} = \frac{\partial E}{\partial d}\frac{\partial d}{\partial b} = \frac{\partial E}{\partial d}$<br>

# Regularization
### L1 regularization
$loss = lossFunction + \sum W$<br>
This penalizes when the weights are high, and can drive the weights of few neurons to zero.
### L2 regularization
$loss = lossFunction + \sum W^2$<br>
This also penalizes when the weights are high, and can drive the weights of few neurons to NEAR zero, but never zero.
Imagine loss function to be of some shape in a 2d(i.e. 2 parameters. Usually its higher dimension) space. Now $W^2$ is a circle at origin and the intersection between the circle and the loss function is the solution which the model will converge to. But if it is $W$, then we have a square whose each vertex is on an axis. Hence, the value that the model will converge to is when one of the weights are 0.

# Optimizer
Now that we have calculated gradient, we can simply reduce the loss by updating the parameters.
### SGD
$p_t = p_{t-1} - \alpha G$ where $\alpha$ is the learning rate(hyperparameter: set before training and usually does not change while training).
and $G$ is the gradient. The stochastic in stochastic gradient descent refers to random sampling of data from the dataset inorder to calculate gradient.
### SGD with momentum
$v_t = \beta v_{t-1} + (1-\beta)G$
p_t = p_{t-1} - \alpha v_t$<br>
By keeping track of the general direction of the descent, any noise present in the dataset won't cause a "shock" to the model.
### RMSprop
$v_t = \beta v_{t-1} + (1-\beta)G^2$
p_t = p_{t-1} - \alpha \frac{G}{\sqrt{v_t} + \epsilon}$
This allows for an adaptive learning rate. If the gradient is large, the step size is small and vice verca.
Adagrad is another optimizer that looks similar to the above except $v_t = v_{t-1} + G^2$. This results in a diminishing lr.
### Adam
Adam optimizer combines momentum and adaptive lr idea. <br>
$m_t = \beta_1 m_{t-1} + (1-\beta_1)G$<br>
$v_t = \beta_2 v_{t-1} + (1-\beta_2)G^2$<br>
$p_t = p_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$<br>

# Normalization
### Layer normalization
$var(x) = \frac{\sum(x-\bar x)^2}{N}$<br>
$x = \frac{x - \bar x}{var(x) + \epsilon}$<br>
Layer normalization helps the model to find the underlying pattern in the data irrespective of how rotated, skewed, streched, etc... it is.
### Batch normalization
The above equations are now applied in the batch dimension of the tensor. This smoothens the loss landscape and makes it easier to navigate. If one of the inputs in a batch is noisy, batch normalization smoothens it out.

# Transformer
Helps pass data from 1 vector to another vector when required.<br>
We multiply the embedding with 3 matrix to get $Q,K,V$.<br>
$Q$: This asks a question<br>
$K$: This raises its hand and says "Hey I have the answer!"<br>
We take $softmax(QK)$ inorder to get a normalized heatmap and finally multiply $V$ to deliver the information. $softmax(QK)$ tells us where to look, but it is $V$ that delivers the required information.<br>
$attention = softmax(\frac{QK^T}{\sqrt d})V$<br>
In multi head attention, we divide the embedding into $\frac{d}{nheads}$ chunks and apply attention on each of these chunks. Think of it as a team of specialists. The team of specialist are working on parts of the project instead of the entire project.

The nlp layer after the attention helps information to flow within embeddings itself.