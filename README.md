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

3. Install the Python dependencies.

4. Run the demos under the demo folder.


If you would like simply use some functions provided by EnsembleBench, you may install it using the following 
    
    pip install EnsembleBench



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

