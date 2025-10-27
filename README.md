# FSFT6mA

The code is the implementation of our method described in the paper “Hong-Jin Yu, Ying Zhang, Dong-Jun Yu, and Guansheng Zheng, FSFT6mA: A Feature-Synthesis Fine-tuning Framework for DNA 6mA Site Prediction”.

DNA N6-methyladenine (6mA) is an important epigenetic modification that plays a critical role in gene expression regulation and has been associated with diverse biological processes and diseases. Accurate identification of 6mA sites is essential for understanding its functional significance. Although an increasing number of computational approaches have been proposed, they almost exclusively rely on sequence-derived features. The potential of novel feature representations to further enhance predictive performance remains an important research problem. In this study, we propose FAST6mA, a novel deep learning-based framework designed to improve 6mA site prediction through feature synthesis. The model is initially trained on the original dataset using a deep convolutional neural network. Subsequently, a Generative Adversarial Network (GAN) is employed to generate synthetic features from intermediate network layers, which are then used to fine-tune the well-trained model in the first stage. Independent validation experiments demonstrate that FAST6mA achieves superior performance compared to existing state-of-the-art predictors.

## Contact
If you are interested in our work or have any suggestions and questions about our research work, please feel free to contact us. E-mail: 
202383290182@nuist.edu.cn.
