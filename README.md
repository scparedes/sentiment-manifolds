# Sentiment Manifolds
The research paper for this project can be found by navigating to the [sentiment manifolds pdf](https://github.com/scparedes/sentiment-manifolds/blob/master/Sentiment%20Manifolds.pdf) in this repo.

## How to work on this project
After downloading IMBD data from kaggle site (https://goo.gl/1of8KR), copy the directory train into the training_data directory.

So the directory structure should be:
training_data : train : {pos,neg}
where pos and neg are directories containing reviews.

To train network do the following: 
modelData is a tuple containing compiled model,
Xtrain,ytrain,Xtest,ytest  where xtest,ytest are dev sets:

from network_architecture_7 import build_CNN_model,train_CNN_model

modelData = build_CNN_model()

train_CNN_model(modelData)


