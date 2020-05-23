# Machine-Learning
Different techniques and optimizations in the world of ML

In case github doesn't loads the jupyter notebook, use online notebook viewer.<br>
For example : https://nbviewer.jupyter.org/github/guru3/Machine-Learning/blob/master/Linear%20Regression.ipynb

### The universal machine-learning workflow

1. Define the problem: What data is available, and what are you trying to predict? Will you need to collect more data or hire people to manually label a dataset? 
2. Identify  a  way  to  reliably  measure  success  on  your  goal.  For  simple  tasks,  this may  be  prediction  accuracy,  but  in  many  cases  it  will  require  sophisticated domain-specific metrics.
3. Prepare the validation process that you’ll use to evaluate your models. In particular, you should define a training set, a validation set, and a test set. The validation-set and test-set labels shouldn’t leak into the training data: for instance, with temporal  prediction,  the  validation  and  test  data  should  be  posterior  to  the training data.
4, Vectorize the data by turning it into vectors and preprocessing it in a way that makes it more easily approachable by a neural network (normalization, and so on).
5. Develop a first model that beats a trivial common-sense baseline, *thus demonstrating that machine learning can work on your problem*. This may not always be the case!
6. Gradually refine your model architecture by tuning hyperparameters and adding regularization. Make changes based on performance on the validation data only, not the test data or the training data. Remember that you should get your model to overfit (thus identifying a model capacity level that’s greater than you need) and only then begin to add regularization or downsize your model.
7. Be  aware  of  validation-set  overfitting  when  turning  hyperparameters:  the  fact that your hyperparameters may end up being over specialized to the validation set. Avoiding this is the purpose of having a separate test set!
