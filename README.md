# Mitigating cold start problems in drug-target affinity prediction with interaction knowledge transferring

**Motivation**: Predicting the drug-target interaction is crucial for drug discovery as well as drug repurposing. Machine learning is commonly used in drug-target affinity (DTA) problem. However, machine learning model faces the cold-start problem where the model performance drops when predicting the interaction of a novel drug or target. Previous works try to solve the cold start problem by learning the drug or target representation using unsupervised learning. While the drug or target representation can be learned in an unsupervised manner, it still lacks the interaction information, which is critical in drug-target interaction.
**Results**: To incorporate the interaction information into the drug and protein interaction, we proposed using transfer learning from chemical-chemical interaction (CCI) and protein-protein interaction (PPI) task to drug-target interaction task. The representation learned by CCI and PPI tasks can be transferred smoothly to the DTA task due to the similar nature of the tasks. The result on the drug-target affinity datasets shows that our proposed method has advantages compared to other pretraining methods in the DTA task.

# Usage

# Citation
If you find this work useful, please cite our paper:
```
@article{nguyen2021mitigating,
  title={Mitigating cold start problems in drug-target affinity prediction with interaction knowledge transferring},
  author={Nguyen, Tri Minh and Nguyen, Thin and Tran, Truyen},
  year={2022}
}
```
