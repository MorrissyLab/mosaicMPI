# Guidelines for multi-resolution factorization

MosaicMPI will integrate datasets with large differences in the resolution (ie., single-cell and bulk). However, the number of programs that you can reasonably expect to discover for each dataset will depend on the number of samples and the diversity of cell types and states.

## Selecting rank ranges for any dataset

When factorizing a dataset in mosaicMPI, you will need to select a rank range and spacing to identify programs. The following rule of thumb can help you balance the number of ranks you need to factorize with the resolution you wish to achieve.
1. Identify the largest rank you want to run. For example, for single-cell brain, we might expect up to 150 cell types/states. For bulk datasets or single-cell datasets with less diversity, the highest rank you might want to run could be a lower number like 20 or 50.
2. Take that number and divide it by 10 to get the spacing. For this example, that would mean a spacing of 15. The reason for such large spacing is that k=149 and k=150 are going to be nearly identical, as are k=145 and k=150. Bigger spacing will make it faster to run and perhaps simplify the integration without any negative consequences.

## Implications for integration

When integrating datasets that have been factorized over different rank ranges and/or with different spacing, you may notice that each dataset has a different or unbalanced number of nodes in the program graph. Althought the number of nodes may vary, this should not alter the community structure, provided that each dataset has appropriate rank ranges to identify the programs of interest.