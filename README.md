# Summary 
This set of Matlab codes reproduce the experimental results in our paper: 
> **Finding a Sparse Vector in a Subspace: Linear Sparsity using Alternating Directions**   
> Qing Qu, Ju Sun, and John Wright. In Advances in Neural Information Processing Systems (NIPS), 2014. Full version on arXiv: http://arxiv.org/abs/1412.4659

### Descriptions
+ *psv_pt.m*: phase transition for planted sparse vector (PSV) model
+ *dl_pt.m*: phase transition for dictionary learning (DL) model
+ *ADM.m*: alternating direction method for solving the key optimization problem:

> min_{q,x} 1/2*||Y*q - x||_2^2 + lambda * ||x||_1, s.t. ||q||_2 = 1 

### Contact
Codes written by Qing Qu, Ju Sun, and John Wright. Questions or bug reports please send email to Qing Qu, qingqu87@gmail.com
