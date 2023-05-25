# GeoGNN Molecular Representation Prediction

This model uses Graph Neural Networks to represent atoms, bonds, and bond angles in a molecule which can be used to predict properties of molecules. The model combines two graphs, one of which uses atoms as nodes and bonds as edges, while the other graph uses bonds as the nodes and bond angles as the edges. This model was pretrained used the ZINC demo dataset and and currently predicts properties for SMILES in the tox21 dataset. However, this model can be finetuned to predict properties of several other molecules. To determine how to do this, look through the source code from the github repository linked below. From there, we can downstream finetune the model so that the model can make better predictions for specific properties. 

## Identifiers

* EOS model ID: `eos1bba`
* Slug: `gem-representation-learning`

## Characteristics

* Input: `Compound`
* Input Shape: `Single`
* Task: `Representation`
* Output: `Descriptor`
* Output Type: `Float`
* Output Shape: `List`
* Interpretation: The output gives us a decimal value. This model currently uses regression to determine the outputs. The outputs that we test are from the tox21 dataset and are the following: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53.

## References

* [Publication](https://www.nature.com/articles/s42256-021-00438-4)
* [Source Code](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/pretrained_compound/ChemRL/GEM)
* Ersilia contributor: [karthikjetty](https://github.com/karthikjetty)

## Citation

If you use this model, please cite the [original authors](https://www.nature.com/articles/s42256-021-00438-4) of the model and the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license. The model contained within this package is licensed under a Apache-2.0 license.

Notice: Ersilia grants access to these models 'as is' provided by the original authors, please refer to the original code repository and/or publication if you use the model in your research.

## About Us

The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission!