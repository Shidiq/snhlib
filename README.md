202201102140
Status: #library
Tags: [[contents#Misc]]

# The SNH's Misc. Tools: snhlib

Install: `pip install git+https://github.com/Shidiq/snhlib.git`

> Latest version: **0.0.1-alpha.14**-


## Features

| Library | Class       | Function          | Params                                                             | Return                                   | Information                                                                   |
| ------- | ----------- | ----------------- | ------------------------------------------------------------------ | ---------------------------------------- | ----------------------------------------------------------------------------- |
| finder  | FinderScan  | scanroot          | root, pattern=`'*.csv'`, verbose=0                                 | full-path-files,  (path, subdirs, files) | get spesific pattern's files in root and sub-root                             |
|         |             | openDataGeNose    | item, cols=`None`                                                  | Pandas dataframe                         | Open item full-path file (csv or json) for open data GeNose                   |
| stats   | Significant | KolmogorovSmirnov | kontrol, test                                                      | value and information                    | Significant hypothesis test using K-S method                                  |
|         |             | PSI               | kontrol, test                                                      | value and information                    | Significant hypothesis test using PSI method                                  |
|         |             | cohend            | kontrol, test                                                      | value and information                    | Calculate effect size using Cohen's method                                    |
| image   | Style       | reset             | -                                                                  | -                                        | Reset rcParams (matplotlib)                                                   |
|         |             | paper             | loc='best', classic=True, figsize=[10.72, 8.205]                   | figure, axes                             | Update rcParams with custom matplotlib theme                                  |
| dataviz | -           | boxplot           | data, id_vars, value_vars, hue=None, hue_order=None, options       | figure, axes                             | Boxplot using pandas dataframe as input and spesicif columns for target value |
|         | CalcPCA     | init              | round_=1, featurename=None, scaler=StandardScaler, colors, markers |                                          | Initial configuration                                                         |
|         |             | fit               | x, y                                                               | pca model                                | analyze PCA using input x and target y                                        |
|         |             | getvarpc          | -                                                                  | pc score, variance , eigen value         |                                                                               |
|         |             | getcomponents     | -                                                                  | loading score                            |                                                                               |
|         |             | getbestfeature    | PC=0, n=3                                                          | top n loading score                      |                                                                               |
|         |             | plotpc            | PC, size, ellipse, ascending, legend, loc                          | figure                                   | Plot PCA analysis                                                             |
|         |             | screenplot        | PC                                                                 | figure                                   |                                                                               |
|         | CalcLDA     | init              | round_=1, scaler, colors, markers, cv                              |                                          |                                                                               |
|         |             | fit               | [x, y]   or [xtrain, xtest, ytrain, ytest]                         |                                          |                                                                               |
|         |             | getvarld          | -                                                                  | lda score and variance                   |                                                                               |
|         |             | getscore          | -                                                                  | cross-validation scores                  |                                                                               |
|         |             | plotlda           | ellipse, ascending, legend, loc                                    | figure                                   | Plot LDA analysis                                                             | 


## Requirements

- pandas
- scipy
- numpy
- matplotlib
- seaborn
- scikit-learn


---
# Release Notes:

- 0.0.1-alpha.1 (2021122800):
  - finder - FinderScan - scanroot
- 0.0.1-alpha.10 (2021122900):
  - stats - Significant - [KolmogorovSmirnov, PSI, cohend]
