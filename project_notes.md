## Handling missing data

- Putting values to 0
> - Makes the attribute irrelevant
> - Might be an issue if 0 makes sense for some attribute

## Handling sequence

- Recurrent neural network

- LSTM (Long Short-Term Memory)

---



# Evaluation

Split the data into a train set and a test set. The test set would be used only as a final result.  
Then split the train set into a true train set and validation set to test different nn architectures and different hyperparameters.  
To have more robust statistics, cross-validation and/or multiple trials can be done which would provide mean and variance estimations.

# Sources

### Papers
- [Multi-Modal Probabilistic Indoor Localization on a Smartphone](https://infoscience.epfl.ch/record/270245?ln=en)
- [https://ai.stanford.edu/~ang/papers/icml11-MultimodalDeepLearning.pdf](Multimodal Deep Learning)

### Neural network solutions
- [A Beginner’s Guide on Recurrent Neural Networks with PyTorch](https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/)
- [Sequence Models and Long-Short Term Memory Networks](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [Sequence-to-sequence neural networks](https://medium.com/@culurciello/sequence-to-sequence-neural-networks-3d27e72290fe)

### Datasets
- [Wireless Indoor Localization Data Set](https://code.datasciencedojo.com/datasciencedojo/datasets/tree/master/Wireless%20Indoor%20Localization)
- [UjiIndoorLoc: An indoor localization dataset](https://www.kaggle.com/giantuji/UjiIndoorLoc)
