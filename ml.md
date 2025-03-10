# Probability
$bayes \space theorem = P(A|B) = \frac{P(B|A)P(A)}{P(B)}$<br>
$P(A,B) = P(A|B)P(B)$<br>
$expectation = \sum xP(x)$<br>
$KL \space divergence = KL(P||Q) = \sum P(x)\log\frac{P(x)}{Q(x)}$

# Metric
$precision = \frac{true positive}{true positive + false positive}$<br>
$recall = \frac{true positive}{true positive + false negative}$<br>
$F1 = 2\frac{precision*recall}{precision+recall}$

# Methods
### Cross-Validation
Keep splitting the dataset and training and testing it on the new split.
### Ensemble
Combine weaker models to create a stronger model.<br>
Bagging: trains multiple models on different parts of the data and aggregates the results. Example: Random forest
Boosting: trains models sequentlially where each model focuses on correcting the errors of the previous model.

# Linear regression
$y = mx + b$ where $m, b$ is slope and y-intercept respectively. This is an equation of a line and will capture any linear relationship between x and y.

$slope=\frac{y_2 - y_1}{x_2-x_1}$<br>
$intercept=\bar y - m\bar x$

### std deviation and variance
$variance = \frac{\sum (x-\bar x)^2}{N}$<br>
$std deviation = \sqrt{variance}$

Both measure the dispersion of data from the avg. Std deviation is easier to interpret because its unit of measurement is the same as the data.

# Logistic regression
In linear regression we assume continuous data. In Logistic regression the outcome is binary. We feed the output from the line to sigmoid function. $m,b$ is found through gradient descent.

### loss function
Binary cross entropy: $-\frac{1}{N}\sum y_i\log(p_i) + (1-y_i)\log(1-p_i)$<br>
where $y,p$ is ground truth and predicted respectively. This is modified version of cross entropy loss: $-\sum y_i \log p_i$. Cross entropy loss compares 2 probability distribuition i.e. the output vector of the model should sum to 1, whereas BCE specializes for "Obama" or "not Obama".

# Decision Tree
2 main ways to measure if the split(question being asked) is good or bad.<br>
$gini = 1 - \sum_{i=0} ^ c p_i^2$<br>
$p^2$ tells us the probability of picking 2 items of the same class back to back and $1-p^2$ tells us the inverse, i.e the probability of not picking 2 items of the same classes back to back. By picking items of the same class back to back, we can measure how good the question being asked is.<br>
$entropy = -\sum_{i=0} ^ c p_i \log p_i$<br>
These 2 ways try to tell us how good the question being asked is. Imagine c=2, "yes" or "no". By asking question A, we may be able to put 1 "yes" in one side and 4 "no" on the other side. Both gini and entropy will be high for this question. If we ask question B, 2 "yes" on one side and 3 "no" on the other results in low gini and entropy.<br>
$Information \space gain = entropy_{parent} - \sum \frac{n_{child}}{n_{parent}}entropy_{child}$<br>
$n_{child}$ is the number of samples in the current node and $n_{parent}$ is the number of samples in the root node. The same formula can be applied for gini impurity.

### Pruning
Pruning help in reducing over-fitting. During training we can set max depth, etc... to prevent overfitting. You can also prune after the tree has been constructed.

# Support Vector Machine
SVM tries to draw a hyperplane through datapoints, with highest amount of breathing room.
Consider 2d dataset<br>
$hyperplane= wx -b =0$<br>
$positive \space plane = wx - b = 1$<br>
$negative \space plane = wx - b = -1$<br>
$w$ is the perpendicular vector from the hyperplane towards positive plane.
Consider we are on hyperplane. Take k steps to positive line. Hence, <br>
$wx - b=0$<br>
$w(x+k\frac{w}{\rVert w \rVert})- b=1$<br>
$k=\frac{1}{\rVert w \rVert}$<br>
Hence we have to $minimize(\rVert w \rVert)$ subjected to $y(wx+b) \geq 1$<br>
We are going to introduce a slack variable $\epsilon$ as in real world the usuallly data is not hard margin seperable.<br>
$y(wx+b) \geq 1 - \epsilon$<br>
$\epsilon \geq 1 - y(wx+b)$<br>
$\epsilon = max(0,1 - y(wx+b))$<br>
This is the loss used in SVM.
### Kernel trick
The above works only for linear relationship. For non-linear we have to apply the kernel trick.
Instead of transforming each data point to a higher space and applying non linear function, we just apply the kernel directly to the data, saving computation and space, generating new datapoints that can be linearly seperated.
The kernel takes the dot product in the higher dimension, hence gives us the effect of high dimension seperatability.

# K-Nearest Neighbors
KNN picks the k closest nearest neighbors to the current data and makes them vote for the label. The distance is given by $\sqrt{\sum_{i=0}^N(x_i - y_i)^2}$ where $N$ is the number of features.

# K-Means Clustering
Assing k centroids randomly. For each points mark the closest centroid. Update the centroid by taking average distance of respective marked centroids. Keep doing this until centroids don't move. This is an unsupervised algorithm.