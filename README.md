### Implementation of orthogonal/inverted matrix-based homomorphic encrpytion for somewhat-encrypyted machine learning

_For experimental purposes only_

Taking the UCI credit default dataset, we built a benchmark classification model (~75%).

Then encrypted the dataset using a set of matrix transformations based on the homomorphic encryption schemata [here](https://www.cs.cmu.edu/~rjhall/JOS_revised_May_31a.pdf).

Running a backpropagation neural network model on encrypted data yielded similar accuracy (~74%) to the vanilla model on non-encrypted data, indicating no loss of insight/pattern during encryption.
