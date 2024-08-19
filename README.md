<!--- Project Logo --->
# EnsembleBench
<!--- a href=""><img src="" alt=""></a --->
-----------------
[![GitHub license](https://img.shields.io/badge/license-apache-green.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)
[![Version](https://img.shields.io/badge/version-0.0.1-red.svg?style=flat)]()
<!---
[![Travis Status]()]()
[![Jenkins Status]()]()
[![Coverage Status]()]()
--->
## Introduction

A set of tools for building high diversity ensembles.

* a set of quantitative metrics for assessing the quality of ensembles;
* a suite of baseline diversity metrics and optimized diversity metrics for identifying and selecting ensembles with high diversity and high quality;
* representative ensemble consensus methods: soft voting (model averaging), majority voting, plurality voting and boosting voting.

CogMI 2020 Presentation Video: https://youtu.be/ErZj_OxyYxc

If you find this work useful in your research, please cite the following papers:

**Bibtex**:
```bibtex
@INPROCEEDINGS{ensemblebench,
    author={Y. {Wu} and L. {Liu} and Z. {Xie} and J. {Bae} and K. -H. {Chow} and W. {Wei}},
    booktitle={2020 IEEE Second International Conference on Cognitive Machine Intelligence (CogMI)},
    title={Promoting High Diversity Ensemble Learning with EnsembleBench},
    year={2020},
    volume={},
    number={},
    pages={208-217},
    doi={10.1109/CogMI50398.2020.00034}
}
@INPROCEEDINGS{dp-ensemble,
    author={Wu, Yanzhao and Liu, Ling and Xie, Zhongwei and Chow, Ka-Ho and Wei, Wenqi},
    booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    title={Boosting Ensemble Accuracy by Revisiting Ensemble Diversity Metrics}, 
    year={2021},
    volume={},
    number={},
    pages={16464-16472},
    doi={10.1109/CVPR46437.2021.01620}
}
@INPROCEEDINGS{hq-ensemble,
    author={Wu, Yanzhao and Liu, Ling},
    booktitle={2021 IEEE International Conference on Data Mining (ICDM)}, 
    title={Boosting Deep Ensemble Performance with Hierarchical Pruning}, 
    year={2021},
    volume={},
    number={},
    pages={1433-1438},
    doi={10.1109/ICDM51629.2021.00184}
}
```

## Instructions


### Installation

1. It is recommended to clone this git repo and refer to the demo folder for building your own projects using EnsembleBench.

       git clone https://github.com/git-disl/EnsembleBench.git
    
2. Initialize the environmental variables:

       source env.sh

3. Install the Python dependencies through the following command: 

       pip install -r requirements.txt


4. Run the demos under the demo folder.


If you would like to simply use some functions provided by EnsembleBench, you may install it using the following command.
    
    pip install EnsembleBench



### Usage
After installation, simply import the library for usage. The `training.ipynb` notebook trains the model and saves the corresponding predictions and weight files locally.

Validation predictions saved from the `training.ipynb` notebook can be used in the notebooks within the demo directory, based on which, a suitable set of ensemble models can then be selected for testing.


Testing predictions are also saved through the `training.ipynb` notebook. These can be used to perform ensembling by simply modifying the variable `mode` to switch between training and testing configurations.

To use testing predictions for ensembling modify the `mode` variable in your script to `testing` to use the saved testing predictions.

This setup allows you to seamlessly transition between validation and testing phases for your model evaluations and ensemble selections.

###### Example (FashionMNIST):
###### Ensemble Selection
###### Code Output from `FocalDiversityBasedEnsembleSelection.ipynb`:
<!-- ```python
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
``` -->
<!-- ##### Output: -->
```
Top EQ_members with their accuracies:
- ('4,5,6,9', 95.50000762939453)
- ('5,6,9,10', 95.4416732788086)
- ('2,4,7,9,10', 95.41667175292969)
```

To evaluate individual or ensemble testing accuracy, simply modify the `model_names`.
<!-- ```python
model_names = ['resnet18', 'resnet34']  # Modify this line to select desired models
``` -->
Ensure that the `model_paths` variable is updated to match the selected models:
```python
model_paths = {
    'resnet18': r'./resnet18_best_model.pth',
    'resnet34': r'./resnet34_best_model.pth'
}
```

###### Comparison (Inference)

<div style="display: flex; justify-content: space-between;">

<div style="flex: 0 0 48%;">
Individual Model Accuracies:
<table style="font-size: 12px;">
  <tr>
    <th>#</th>
    <th>Model</th>
    <th>Test Acc (%)</th>
  </tr>
  <tr>
    <td>0</td>
    <td>ResNet34</td>
    <td>94.18</td>
  </tr>
  <tr>
    <td>1</td>
    <td>ResNet50</td>
    <td>93.66</td>
  </tr>
  <tr>
    <td>2</td>
    <td>ResNet101</td>
    <td>93.65</td>
  </tr>
  <tr>
    <td>3</td>
    <td>ResNet152</td>
    <td>93.50</td>
  </tr>
  <tr>
    <td>4</td>
    <td>AlexNet</td>
    <td>92.22</td>
  </tr>
  <tr>
    <td>5</td>
    <td>DenseNet121</td>
    <td>94.51</td>
  </tr>
  <tr>
    <td>6</td>
    <td>DenseNet161</td>
    <td>94.60</td>
  </tr>
  <tr>
    <td>7</td>
    <td>DenseNet169</td>
    <td>94.73</td>
  </tr>
  <tr>
    <td>8</td>
    <td>SqzNet1_1</td>
    <td>93.01</td>
  </tr>
  <tr>
    <td>9</td>
    <td>GoogleNet</td>
    <td>94.52</td>
  </tr>
  <tr>
    <td>10</td>
    <td>VGG11</td>
    <td>92.14</td>
  </tr>
  <tr>
    <td>11</td>
    <td>VGG13</td>
    <td>92.27</td>
  </tr>
  <tr>
    <td>12</td>
    <td>CvNeXtTiny</td>
    <td>94.05</td>
  </tr>
</table>
</div>

<div style="flex: 0 0 48%;">
Ensemble Accuracies:
<table style="font-size: 12px;">
  <tr>
    <th>Members (Idx)</th>
    <th>Val Acc (%)</th>
    <th>Test Acc (%)</th>
  </tr>
  <tr>
    <td>4, 5, 6, 9</td>
    <td>95.50</td>
    <td>95.08</td>
  </tr>
  <tr>
    <td>5, 6, 9, 10</td>
    <td>95.44</td>
    <td>95.17</td>
  </tr>
  <tr>
    <td>2, 4, 7, 9, 10</td>
    <td>95.42</td>
    <td>95.11</td>
  </tr>
</table>
</div>

</div>


Links to the corresponding weight files and predictions obtained after training are included inside the notebooks.


## Supported Platforms

The source codes have been tested on Ubuntu 16.04 and Ubuntu 20.04.



## Development / Contributing


## Issues


## Status


## Contributors

See the [people page](https://github.com/git-disl/EnsembleBench/graphs/contributors) for the full listing of contributors.

## License

Copyright (c) 20XX-20XX [Georgia Tech DiSL](https://github.com/git-disl)  
Licensed under the [Apache License](LICENSE).

