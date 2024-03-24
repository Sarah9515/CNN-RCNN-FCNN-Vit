Part 1: CNN Classifier 

 

Comparative Analysis of CNN and Faster R-CNN Models for Image Classification and Object Detection 

  

Objective: 

The main objective of this study is to explore the capabilities of the PyTorch library for building neural architectures such as Convolutional Neural Networks (CNN) and Region-based Convolutional Neural Networks (Faster R-CNN) for computer vision tasks. Specifically, the study aims to: 

  

1. Implement a CNN architecture for classifying the MNIST dataset. 

2. Implement a Faster R-CNN architecture for object detection. 

3. Compare the performance of the CNN and Faster R-CNN models using various metrics such as accuracy, F1 score, loss, and training time. 

4. Fine-tune pre-trained models (VGG16 and AlexNet) on the MNIST dataset and compare the results with CNN and Faster R-CNN models. 

   

Dataset: MNIST Dataset (Digit Recognizer | Kaggle) 


 

  

1. Establish a CNN Architecture: 

- The CNN architecture consists of convolutional layers, and fully connected layers. 

- Hyperparameters such as kernel size, padding, stride, and optimizers are defined. 

- The model is trained in GPU mode for faster computation. 

Data Augmentation :  

 

 

CNN Model : 

 

 

 

Adam Optimizer : 

 

 

Metrics : 

 

Analysing Result : 

 

  

2. Implementation of Faster R-CNN: 

- Faster R-CNN architecture is implemented for object detection tasks. 

- This architecture consists of a backbone network (e.g., ResNet50), a Region Proposal Network (RPN), and a detection network. 

- Hyperparameters are set for the RPN and detection network. 

 

  

3. Model Comparison: 

- The performance of the CNN and Faster R-CNN models is compared using metrics such as accuracy, F1 score, loss, and training time. 

- Evaluation metrics are calculated on a validation dataset to assess the models' performance. 

  

4. Fine-tuning Pre-trained Models: 

- Pre-trained models like VGG16 and AlexNet are fine-tuned on the MNIST dataset. 

- The fine-tuned models are compared with the CNN and Faster R-CNN models in terms of classification accuracy and other relevant metrics. 

 

 

 

Train AlexNet Model & print the Metrics 

 

Metrics : 

 

 

 

 

Train VGG16 Model 

 

Metrics : 

 

 

Conclusion: 

- The CNN model demonstrates strong performance in classifying images from the MNIST dataset, achieving high accuracy and relatively low training time. However, the performance may vary depending on the complexity of the dataset and task. 

- While Faster R-CNN shows promising results in detecting objects within images, it may require more computational resources compared to CNN due to its complex architecture. 

- Fine-tuning pre-trained models such as VGG16 on the MNIST dataset yields the best metrics, indicating the effectiveness of transfer learning. VGG16, with its deeper architecture and pre-learned features, excels in capturing intricate patterns present in the MNIST dataset, leading to superior classification performance. 

- Overall, the choice between CNN, Faster R-CNN, and fine-tuned pre-trained models depends on the specific requirements of the task, dataset complexity, and available computational resources. For image classification tasks with structured datasets like MNIST, leveraging pre-trained models such as VGG16 could offer the best trade-off between performance and computational efficiency. 

 

 

 

Part 2: Vision Transformer (ViT) 

Comparative Analysis of Vision Transformers (ViT) and Convolutional Neural Networks (CNN) for Image Classification 

 

Since their introduction by Dosovitskiy et al. in 2020, Vision Transformers (ViT) have emerged as a dominant architecture in the field of computer vision, achieving state-of-the-art performance in image classification and various other tasks. 

  

1. Establishing a Vision Transformer Architecture: 

- Following the tutorial provided (https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c), a Vision Transformer model architecture is established from scratch. 

- The architecture includes self-attention mechanisms and positional encodings crucial for capturing global dependencies in images. 

- The Vision Transformer model is trained on the MNIST dataset for the classification task. 

  

2. Interpretation and Comparison of Results: 

- The obtained results from the Vision Transformer model on the MNIST dataset are interpreted in terms of classification accuracy, F1 score, and any other relevant metrics. 

- The results are compared with those obtained in the first part, where CNN and Faster R-CNN architectures were employed for image classification and object detection tasks, respectively. 

  

Comparison with Part 1: 

- The performance of the Vision Transformer model is compared with that of the CNN model used in Part 1 for the MNIST classification task. 

- Metrics such as classification accuracy, F1 score, and training time are considered for the comparison. 

- Any significant differences in performance, computational efficiency, or other aspects between the Vision Transformer and CNN models are highlighted. 

  

Conclusion: 

Do Vision Transformers See Like Convolutional Neural Networks? 

Convolutional neural networks (CNNs) have so far been the de-facto model for visual data. Recent work has shown that (Vision) Transformer models (ViT) can achieve comparable or even superior performance on image classification tasks. This raises a central question: how are Vision Transformers solving these tasks? Are they acting like convolutional networks, or learning entirely different visual representations? Analyzing the internal representation structure of ViTs and CNNs on image classification benchmarks, we find striking differences between the two architectures, such as ViT having more uniform representations across all layers. We explore how these differences arise, finding crucial roles played by self-attention, which enables early aggregation of global information, and ViT residual connections, which strongly propagate features from lower to higher layers. We study the ramifications for spatial localization, demonstrating ViTs successfully preserve input spatial information, with noticeable effects from different classification methods. Finally, we study the effect of (pretraining) dataset scale on intermediate features and transfer learning, and conclude with a discussion on connections to new architectures such as the MLP-Mixer. 
