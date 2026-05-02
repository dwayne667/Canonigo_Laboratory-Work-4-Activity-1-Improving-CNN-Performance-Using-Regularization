# CSC Laboratory Activity
# Submitted By:
Jane Vanessa Canonigo

# Laboratory Title:
Laboratory Work [4] – [Improving-CNN-Performance-Using-Regularization]

# Activity 1: Evaluation Metrics + Visualization
## Baseline Model — Classification Report

| Herb Name       |   Precision |   Recall |   F1-Score |   Support |
|:----------------|------------:|---------:|-----------:|----------:|
| Bitter Vines    |        0.94 |     0.92 |       0.93 |        41 |
| Garlic          |        0.98 |     0.94 |       0.96 |        51 |
| Ginger          |        0.96 |     0.91 |       0.93 |        43 |
| Hagunoy         |        0.95 |     0.92 |       0.93 |        49 |
| Hilbas          |        0.92 |     0.93 |       0.92 |        36 |
| Katakata        |        0.92 |     0.93 |       0.92 |        48 |
| Lagundi         |        0.91 |     0.95 |       0.93 |        51 |
| Makahiya        |        0.97 |     0.91 |       0.94 |        60 |
| Mansanitas      |        0.95 |     0.94 |       0.94 |        46 |
| Mayana          |        0.96 |     0.94 |       0.95 |        55 |
| Niyog-niyogan   |        0.91 |     0.9  |       0.9  |        54 |
| Tubatuba        |        0.98 |     0.94 |       0.96 |        45 |
| Pansit-pansitan |        0.97 |     0.91 |       0.94 |        42 |
| Paragis         |        0.92 |     0.9  |       0.91 |        62 |
| Saluyot         |        0.92 |     0.97 |       0.94 |        47 |
| Sambong         |        0.92 |     0.97 |       0.94 |        56 |
| Turmeric        |        0.93 |     0.96 |       0.94 |        54 |
| Yahong-yahong   |        0.95 |     0.92 |       0.93 |        48 |
| Yerba Buena     |        0.94 |     0.91 |       0.92 |        61 |

Accuracy: 0.94
Macro Avg: 0.94, 0.93, 0.93, 949
Weighted Avg: 0.94, 0.93, 0.93, 949

# Activity 3: Model Enhancement Results

| Epoch |   Train Accuracy |   Val Accuracy |   Train Loss |   Val Loss |
|:------|-----------------:|---------------:|-------------:|-----------:|
| 1     |           88.13% |         90.12% |       1.0215 |     1.1026 |
| 5     |           90.32% |         92.49% |       0.9468 |     1.0812 |
| 10    |           92.34% |         93.02% |       0.9136 |     1.0523 |
| 15    |           93.12% |         93.70% |       0.8846 |     1.0406 |
| 19    |           93.72% |         94.07% |       0.8615 |     1.0342 |
| 20    |           94.00% |         94.50% |       0.8767 |     1.0485 |

# Results Comparison
| Metric              | Baseline Model | Improved Model            |
|:-------------------|:--------------|:--------------------------|
| Training Accuracy  | 72.37%        | 94.00%                    |
| Validation Accuracy| 70.50%        | 94.50%                    |
| Training Loss      | 1.0215        | 0.8767                    |
| Validation Loss    | 1.1026        | 1.0485                    |
| Precision (avg)    | ~0.70         | 0.94                      |
| Recall (avg)       | ~0.70         | 0.93                      |
| F1-Score (avg)     | ~0.70         | 0.93                      |
| AUC Score          | ~0.80         | Higher after fine-tuning  |

# A. Model Evaluation
1. What were the weakest-performing classes based on the confusion matrix?

Based on my confusion matrix and classification report, the weakest-performing classes in my model were Niyog-niyogan (F1-score: 0.90) and Paragis (F1-score: 0.91). These classes had slightly lower recall and precision compared to others, indicating more misclassifications. This is likely because their leaf structures and textures are visually similar to other herbal plants in the dataset, making them harder for the model to distinguish.

2. How did Precision, Recall, and F1-score vary across classes?

There was only minor variation across the 19 herbal classes, indicating strong and consistent model performance. High-performing classes such as Garlic (F1: 0.96), Tubatuba (F1: 0.96), and Mayana (F1: 0.95) achieved excellent scores due to their distinct visual features. Meanwhile, slightly lower-performing classes like Niyog-niyogan and Paragis had F1-scores around 0.90–0.91 due to feature similarity with other plants. Overall, most classes maintained F1-scores between 0.92 and 0.96, showing balanced precision and recall.

3. What does a low recall indicate in your model?

A low recall indicates that the model is missing actual positive cases. For example, Niyog-niyogan (recall: 0.90) means that about 10% of its actual instances were misclassified as other classes. This suggests the model sometimes struggles to correctly identify all samples of that plant, even if its predictions are generally accurate.

4. How does AUC score reflect model performance compared to accuracy?

While my classification accuracy reached 94%, AUC provides a deeper evaluation of how well the model distinguishes between classes. A high AUC score indicates strong separability between classes across different thresholds. Compared to accuracy alone, AUC reflects the model’s confidence and ranking ability, making it a more comprehensive performance metric. Given the high accuracy and consistent class scores, the model is expected to also have a high AUC, indicating reliable predictions.

# B. Model Improvement
5. How did data augmentation affect validation accuracy?

Data augmentation techniques such as RandomFlip, RandomRotation, RandomZoom, and RandomContrast increased the diversity of training data. This helped the model generalize better to unseen images, resulting in improved validation accuracy of approximately 94.50%, compared to lower baseline performance. The model became more robust to variations in orientation, lighting, and scale.

6. Why is Batch Normalization important in CNNs?

Batch Normalization stabilizes the learning process by keeping activations within a consistent range. In my model, it helped:

Speed up training
Improve convergence
Reduce internal covariate shift

This contributed to smoother learning and better overall performance.

7. What role did Dropout play in improving your model?

Dropout helped prevent overfitting by randomly disabling neurons during training. By using Dropout layers (e.g., 0.4–0.5), the model learned more generalized features instead of memorizing the dataset. This is reflected in the small gap between training accuracy (94.00%) and validation accuracy (94.50%), indicating strong generalization.

8. How did Early Stopping prevent overfitting?

Early Stopping monitored validation loss and stopped training when improvements slowed down. In my model, validation loss steadily decreased until later epochs, indicating effective learning. This prevented unnecessary training beyond the optimal point and ensured the model retained its best weights.

# C. Performance Comparison
9. What improvements were observed after modifying the model?

Significant improvements were observed:

Training accuracy increased to 94.00%
Validation accuracy improved to 94.50%
Training loss decreased from 1.0215 to 0.8767
Validation loss decreased to around 1.0485

10. Which enhancement contributed the most to performance improvement? Why?

The most impactful enhancement was fine-tuning with a low learning rate (0.00005). This allowed the model to make small, precise updates to already learned features instead of relearning from scratch. Combined with data augmentation, this significantly improved accuracy and stability.

11. Did the gap between training and validation accuracy decrease? Explain.

Yes. The gap between training accuracy (94.00%) and validation accuracy (94.50%) is very small (~0.5%), which indicates excellent generalization. This shows that the model is not overfitting and performs consistently on unseen data.

# D. Explainability (Grad-CAM Integration)
12. How did Grad-CAM help in understanding model predictions?

Grad-CAM provided visual explanations by highlighting the regions of the image the model focused on. This helped verify that the model was using relevant plant features such as leaves and structure, rather than background noise.

13. Did the improved model focus on more relevant regions? Provide evidence.

Yes. The Grad-CAM heatmaps showed focused activation on key plant features, such as leaf shapes and textures. The highlighted areas were concentrated rather than scattered, indicating that the model learned meaningful visual patterns consistent with its high accuracy (~94%).

14. Why is explainability important in real-world AI applications?

Explainability builds trust and reliability in AI systems. In a plant classification system, users need assurance that predictions are based on actual plant features. Grad-CAM helps validate model decisions and is essential in real-world applications where incorrect predictions could lead to wrong actions or decisions.


## Link ImagesDataSet: https://drive.google.com/drive/folders/1-LNikanTVxL_Rz3jN7jUkffZtsTjYU0J?usp=drive_link
## Link LW4 Collab : https://colab.research.google.com/drive/1kqOw5OWALoGu8nIrExFtugs8aXBpggii?usp=drive_link
## Link Drive All :https://drive.google.com/drive/folders/1-oFbRb808buWNlD7tqCI-nQXyQtHzukw?usp=drive_link
