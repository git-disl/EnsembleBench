# EnsembleBench Notebooks

These notebooks utilize the EnsembleBench library to perform ensembles of multiple pre-trained models. They also include various diversity-based metrics for evaluation and support different voting strategies.

## Instructions

The `modelTraining` directory contains notebooks that were used to train and validate MNIST and FashionMNIST datasets. 

The `training.ipynb` notebook trains the model and saves the corresponding predictions and weight files locally.

Validation predictions saved from the `training.ipynb` notebook can be used in the `BaselineDiversityBasedEnsembleSelection.ipynb` and `FocalDiversityBasedEnsembleSelection.ipynb` notebooks. Based on the performance, a suitable set of ensemble models can be then selected for testing.

### Example (FashionMNIST):
#### Inference and Ensemble Selection
##### Final Code Cell from `FocalDiversityBasedEnsembleSelection.ipynb`:
```python
# Create a list of tuples (member, accuracy)
member_accuracy_pairs = [(member, teamAccuracyDict[member]) for member in EQ_members if member in teamAccuracyDict]

# Sort the list by accuracy in descending order
sorted_member_accuracy_pairs = sorted(member_accuracy_pairs, key=lambda x: x[1], reverse=True)

# Check if there are fewer than 3 members
if len(sorted_member_accuracy_pairs) < 3:
    top_3_members = sorted_member_accuracy_pairs
    print("Less than 3 members are available in EQ_members.")
else:
    # Get the top 3 members with highest accuracy
    top_3_members = sorted_member_accuracy_pairs[:3]

# Extract the top 3 accuracies
top_3_accuracies = [accuracy for member, accuracy in top_3_members]

print("Top EQ_members with their accuracies:", top_3_members)
```
##### Output:
```
Top EQ_members with their accuracies:
- ('4,5,6,9', 95.50000762939453)
- ('5,6,9,10', 95.4416732788086)
- ('2,4,7,9,10', 95.41667175292969)
```

To evaluate individual or ensemble testing accuracy, simply modify the `model_names` variable in `inference.ipynb`:

```python
model_names = ['resnet18', 'resnet34']  # Modify this line to select desired models
```
Ensure that the `model_paths` variable is updated to match the selected models:
```python
model_paths = {
    'resnet18': r'./resnet18_best_model.pth',
    'resnet34': r'./resnet34_best_model.pth'
}
```

#### Comparison 
##### Individual Model Accuracies:

| Index | Model         | Testing Accuracy (%) |
|-------|---------------|----------------------|
| 0     | ResNet34      | 94.18                |
| 1     | ResNet50      | 93.66                |
| 2     | ResNet101     | 93.65                |
| 3     | ResNet152     | 93.50                |
| 4     | AlexNet       | 92.22                |
| 5     | DenseNet121   | 94.51                |
| 6     | DenseNet161   | 94.60                |
| 7     | DenseNet169   | 94.73                |
| 8     | SqueezeNet1_1 | 93.01                |
| 9     | GoogleNet     | 94.52                |
| 10    | VGG11         | 92.14                |
| 11    | VGG13         | 92.27                |
| 12    | ConvNeXt Tiny | 94.05                |


##### Ensemble Accuracies:
| Members (Indices) | Val Accuracy (%) | Testing Accuracy (%) |
|-------------------|------------------|----------------------|
| 4, 5, 6, 9        | 95.50            | 95.08                |
| 5, 6, 9, 10       | 95.44            | 95.17                |
| 2, 4, 7, 9, 10    | 95.42            | 95.11                |


The inference notebook (`inference.ipynb`) loads weight files from models pretrained in the training notebook. Links to these weight files and predictions obtained after training are included inside the notebooks.