# Automating Hyperparameters Tuning and Model Benchmarking with ScikitLearn

## NON-TECHNICAL EXPLANATION OF THE PROJECT

The main objective of this project is to show how the search, for the best hyperparameters, for different families of models, can be automated. I.e, we want to be able to easily - automatically - compare the best models of each family. Besides, we are interested in models for image classification. But we are more interested in the search and benchmarking techniques than in the sort of family model to use for image classification, and how performant could be for the task at hand. That can be seen as an introduction to AutoML.

As a secondary objective, it is intended to show the limitations of this type of experiments on PCs and explain the need for AutoML tools in the cloud.

A final objective is to identify and learn the sort of training and search patterns for hyperparameters tuning and model benchmarking. These patterns are mostly based on pipes and are the same or very similar both, locally (PC) and in the cloud. ScikitLearn is designed around the pipe pattern and is a great framework to comprehend these techniques.

## DATA

Because we will train many times the same model using different hyperparameters, we will choose a very simple training image dataset. We select as image dataset the Modified National Institute of Standards and Technology (MNIST) Dataset. The reason is that we require very simple images which will use very few computational resources and thus will facilitate each training step.

The MNIST database is also widely used for training and testing in the field of machine learning, and it became the classic standard the facto to compare performance between models of different families, architectures or hyperparameters. It is an extremely good database for people who want to try machine learning techniques and pattern recognition methods on real-world data while spending minimal time and effort on data preprocessing and formatting. Its simplicity and ease of use are what make this dataset so widely used and deeply understood.

The MNIST database is a large collection of handwritten digit images, each of size 28x28 pixels with 256 grey levels. The digits have been size-normalized and centered in a fixed-size image.

The MNIST database has a training set of 60,000 examples and a test set of 10,000 examples. The training set consists of handwritten numbers from 250 different people, of which 50% are high school students and 50% are from the Census Bureau.

You can find and download MNIST database here: http://yann.lecun.com/exdb/mnist/

The original MNIST dataset that you can find in the former url, is in a format that is difficult for beginners to use. The final MNIST dataset that we have been used, takes advantage of the work of [Joseph Redmon](https://pjreddie.com/) to provide the [MNIST dataset in a CSV format](https://pjreddie.com/projects/mnist-in-csv/).

The dataset consists of two files:

- `mnist_train.csv`
- `mnist_test.csv`

The `mnist_train.csv` file contains the 60,000 training examples and labels. The `mnist_test.csv` contains 10,000 test examples and labels. Each row consists of 785 values: the first value is the label (a number from 0 to 9) and the remaining 784 values are the pixel grey-scale values (a number from 0 to 255).

## MODEL

For that project, some valid family model candidates are K-Nearest Neighbors, Random Forest, Support-vector machines (SVM), Deep neural networks (DNN) and Convolutional neural networks (CNN), among others.

Because we will train many times the same model using different hyperparameters, we have chosen the SMV family as one of the simplest models able to recognize image patterns. The idea is to use a very simple model, hopefully, one that will require very little training time at each tuning step. In our particular case, we will see that with an undersampled MNIST training set (5_000 images), each training step will last 2-3 secs only.

## HYPERPARAMETER OPTIMISATION

Search for parameters of machine learning models that result in best cross-validation performance is necessary in almost all practical cases to get a model with best generalization estimate.

The description of which hyperparameters we have and how we have chosen to optimise them is the core of that project. By using cross-validation, we will compare SVC models with different Kernels, parameters, as well as one-versus-one approach. We will report the test error of the model picked.

- First, we use **Grid Search optimization**, on the following **search space**:

  ```python
  kernels = ('linear', 'poly', 'rbf')
  Cs = (0.001, 0.01, 0.1, 1, 10, 100, 1000)
  degrees = (3, 8, 10)
  gammas = ('auto', 'scale')

  param_grid = {'kernel': kernels, 'C': Cs, 'degree': degrees, 'gamma': gammas}
  param_grid
  ```

  A standard approach in scikit-learn is using **sklearn.model_selection.GridSearchCV** class, which takes a set of values for every parameter to try, and simply enumerates all combinations of parameter values. A more scalable approach is using **sklearn.model_selection.RandomizedSearchCV**, which however does not take advantage of the structure of a search space.

- Second, we use **Bayes Search optimization**, on the following **search space**:

  ```python
  bayes_opt = BayesSearchCV(
    svm.SVC(),
    {
        'C': (1e-6, 1e+6, 'log-uniform'),     # float valued parameter
        'gamma': (1e-6, 1e+1, 'log-uniform'), # float valued parameter
        'degree': (1, 8),                     # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    },
    n_iter=64,
    cv=5,
    verbose=True
  )
  ```

  Scikit-optimize provides **skopt.BayesSearchCV** as a drop-in replacement for **sklearn.model_selection.GridSearchCV**, which ***utilizes Bayesian Optimization where a predictive model referred to as “surrogate” is used to model the search space and utilized to arrive at good parameter values combination as soon as possible***.

  Note that **Grid Search Optimization can use discrete dimensions, and thus a completely or partially discrete search space**. Now, **with Bayesian Optimization, we can naturally use continuous dimensions, and a more comprehensive search space**. In our example, `C` and `gamma` now are considered as continuous values between `0.000_001` and `1_000_000`. That means that searched space is "denser" and solved optimization problems are harder.

Finally, we **benchmark two models** (of the same family, we could have done the same with two models of different families): in practice, **one wants to enumerate over multiple predictive model classes, with different search spaces and number of evaluations per class.** We present an example of such search over parameters of `Linear SVM` and `RBF SVM` models. Both models, despite beign from the same family, present different hyperparameters and training challenges (such the number of iterations to converge):

```python
# single categorical value of 'model' parameter sets the model class
linsvc_search = {
    'model': [svm.LinearSVC(max_iter=1_000_000)],              # Make max_iter very high: enough to avoid convergence problems.
                                                               # Also make the training verbose to know whether liblinear converges or
                                                               # not: if not, results are not valid.
                                                               # Note that not having convergence warnings makes the fitting process quicker
    'model__C': (1e-6, 1e+6, 'log-uniform'),                   # log-uniform: understand as search over p = exp(x) by varying x
}
# We could get ConvergenceWarnings because the problem is not well-conditioned. But that's fine, this is just an example.

# explicit dimension classes - Real, Categorical, Integer - can be specified like this:
svc_search = {
    'model': Categorical([svm.SVC()]),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),         # log-uniform: understand as search over p = exp(x) by varying x
    'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),     # log-uniform: understand as search over p = exp(x) by varying x
    'model__degree': Integer(1,8),
    'model__kernel': Categorical(['linear', 'poly', 'rbf']),
}
```

To fully automate this kind of experiments, we use **pipes**. That pattern is extensively used by ScikitLearn and is part of its core design. Using pipes we can automate the end-to-end benchmarking process, from getting and transforming the data to use, to launching experiments and collecting results. The `pipe` object (which includes all data preparation steps, model selection and model training steps), can be used as a replacement for the `model` object and thus be used as an estimator per-se:

```python
# pipeline class is used as estimator to enable search over different model types
pipe = Pipeline(steps = [('preprocess', column_transf),
                         ('model', svm.SVC())])                # model here is a kind of variable: its value
                                                               # will be redefined in the next search spaces.
```

Finally, **we perform Bayes Optimization on our new `pipe` estimator, using 5-folds cross-validation:**

```python
bayes_opt2 = BayesSearchCV(
    pipe,
    [(linsvc_search, 16), (svc_search, 64)],                   # (parameter space, # of evaluations) = (grid_search, n_iter)
    cv=5,
    verbose=3,                                                 # verbose=3 better than verbose=True only: you have to know
                                                               # whether there are convergence problems or not.
    random_state=1
)
```

Observe how important is to observe the logs of the process. You have to know whether there are or not convergence problems. When you have convergence problems, results (accuracy scores) are not valid, and the saved model at the end of the process might not be able to predict anything.


## RESULTS

That is a brief summary of the project results and some things learnt from it.

First let's compare results of the different approaches we have proposed:

**2. SVM Model Training using two different Kernels:**

Here we trained two SVM models on the same training set and with default parameters. Accuracy and confusion matrix results are:

- **Linear SVM:**
  - test set accuracy_score = 0.8606
  - test confusion matrix:

    <img src='./img/2.2.2_Linerar_SVM_Test_Confusion_Matrix.png' alt='' width=350>

- **RBF SVM:**
  - test set accuracy_score = 0.9542
  - test confusion matrix:

    <img src='./img/2.4.2_RBF_SVM_Test_Confusion_Matrix.png' alt='' width=350>

**3. Hyper-parameter search by using cross-validation:**

- **Grid Search:**
  - test set accuracy_score = 0.9642
  - Grid search best model result: {'C': 0.001, 'degree': 10, 'gamma': 'scale', 'kernel': 'poly'}

- **Bayes Search:**
  - test set accuracy_score = 0.9656
  - Bayes search best model result: [('C', 337597.3274086411), ('degree', 7), ('gamma', 10.0), ('kernel', 'poly')]

**4. Evaluating different SVM model families with Sklearn BayesSearchCV and Pipes: a more comprehensive example**

- Bayes Search 2: test set accuracy_score = 0.9638  --> Bayes search 2 best result: [('model', SVC(C=1_754.5053606495005, degree=8, gamma=1.2240709533286132, kernel='poly'))]
- Bayes Search 2: test set accuracy_score = 0.1000  --> Bayes search 2 best result: [('model', SVC(C=1_000_000.0, degree=6, gamma=0.0001822281872335297, kernel='poly'))]

## (OPTIONAL: CONTACT DETAILS)
If you are planning on making your github repo public you may wish to include some contact information such as a link to your twitter or an email address.

