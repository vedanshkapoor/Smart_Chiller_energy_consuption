![Screenshot 2025-06-02 192425](https://github.com/user-attachments/assets/e10bfa24-0621-49d6-a9ba-fabc1da07911)# Energy Demand Forecasting using Teaching-Learning-Based Optimization and Feedforward Neural Networks

This repository contains the code and resources for a research study on short-term energy demand forecasting. The project leverages a Feedforward Neural Network (FFNN) optimized with the Teaching-Learning-Based Optimization (TLBO) algorithm to achieve high accuracy in predicting energy consumption, with a particular focus on handling high-output value ranges effectively.

## Abstract

This study presents an efficient approach to short-term energy demand forecasting using a feedforward neural network (FFNN) trained with the Teaching-Learning-Based Optimization (TLBO) algorithm. The methodology leverages a publicly available dataset derived from a recent study on hybrid energy forecasting models. The experimental workflow includes data preprocessing, correlation analysis, and visualization. A specialized data-sampling strategy was implemented to improve the model's generalization on higher energy consumption values. The TLBO-optimized model achieved superior accuracy with an $R^{2}$ of 0.9636, RMSE of 5.86, and MAE of 4.03, demonstrating strong predictive performance and practical applicability in demand management systems.

## 1. Introduction

Accurate short-term energy demand forecasting is crucial for efficient power system operation, demand-side management, and load balancing. Traditional models often struggle with generalizing to high-demand scenarios. This study explores the use of the Teaching-Learning-Based Optimization (TLBO) algorithm to train a feedforward neural network (FFNN) for enhanced accuracy and robustness in energy prediction, with particular emphasis on handling high-output value ranges effectively.

## 2. Data Collection and Preprocessing

### 2.1 Data Source

The dataset used in this study was obtained from Kaggle and corresponds to the experimental data featured in the 2024 Energy journal article titled "Short-term energy demand forecasting based on a hybrid optimization algorithm integrating teaching-learning-based optimization with neural networks." The data includes temporal energy demand values along with associated meteorological and calendar-based features.

### 2.2 Data Cleaning

No missing values were present in the dataset, which negated the need for imputation techniques. Data types were verified and properly encoded, and all feature entries were retained due to the integrity of the source data.

### 2.3 Preprocessing Strategy

Normalization was intentionally omitted due to concerns regarding disproportionate error scaling when modeling high-magnitude values. In conventional regression settings, normalization can compress high-value targets, inadvertently reducing model sensitivity to extreme demand peaks. Since the target variable spans a broad numerical range, retaining raw scale was deemed advantageous for interpretability and performance.

### 2.4 Training Data Construction

To improve the model's ability to generalize to both typical and peak demand levels, a custom training dataset was constructed. Specifically:

-   80% of the training data comprised samples where the target energy demand was greater than 200 units.
-   The remaining 20% included samples with demand below 200 units.

This stratified sampling approach provided the model with sufficient exposure to peak values, often underrepresented in uniform sampling, thus mitigating bias toward lower demand predictions.

## 3. Data Exploration and Visualization

![Screenshot 2025-06-02 194732](https://github.com/user-attachments/assets/1f903886-66d7-4526-a3fd-ef6bf540da77)


### 3.1 Task Abstraction

Exploratory data analysis (EDA) was conducted to identify trends, correlations, and patterns relevant to short-term energy forecasting. This phase sought to answer critical questions:

-   What features most strongly influence energy demand?
-   Are there identifiable daily or hourly consumption patterns?
-   Can peak periods be visually isolated?

To validate the utility of these visualizations, feedback was obtained from five individuals with varying degrees of familiarity with data analytics. All users reported improved interpretability and relevance of the visual outputs.

### 3.2 Visual Encoding and Rationale

#### 3.2.1 Correlation Heatmap

A correlation matrix heatmap was generated to highlight statistical relationships between features and the target variable. Strong linear correlations, especially with time-of-day and temperature-related features, were observed. This guided feature prioritization in model training.

#### 3.2.2 Energy Demand over Time Plot
![Screenshot 2025-06-02 194933](https://github.com/user-attachments/assets/c63c875e-8824-4d6b-842c-9380e937a360)

A time-series line plot was created to display energy demand across a 48-hour period. This visualization revealed clear demand cycles, with noticeable peaks during typical daytime hours. Such patterns justify the selection of temporal features and support the feasibility of short-term prediction using FFNN architectures.

## 4. Modelling and Optimization

### 4.1 Model Architecture
![Screenshot 2025-06-02 192405](https://github.com/user-attachments/assets/b79b5832-5c9e-4dae-a480-4aa030a1b429)

A Feedforward Neural Network (FFNN) was employed for energy demand forecasting due to its simplicity, computational efficiency, and strong performance in structured data regression tasks. The architecture consisted of a multi-layer dense network enhanced with dropout regularization and batch normalization to improve generalization and convergence stability. The complete architecture is summarized below:

-   **Input Layer:** Fully connected dense layer with 96 neurons and Swish activation, with L2 regularization ($1e-4$).
-   **Hidden Layers:**
    -   First hidden layer: 64 neurons → Dropout (0.25) → Batch Normalization → Swish activation → L2 regularization ($1e-4$).
    -   Second hidden layer: 48 neurons → Dropout (0.2) → Batch Normalization → Swish activation → L2 regularization ($1e-4$).
    -   Third hidden layer: 24 neurons → Dropout (0.1) → Batch Normalization → Swish activation → L2 regularization ($1e-4$).
-   **Output Layer:** Single neuron with linear activation to predict energy demand as a continuous value.

**Total trainable parameters:** 11,665
**Non-trainable parameters (e.g., from batch normalization):** 464
**Overall model size:** ~47.38 KB

The design balances depth and regularization, ensuring the model can capture complex temporal patterns while minimizing the risk of overfitting.

### 4.2 Optimization Algorithm: Teaching-Learning-Based Optimization (TLBO)

The model was optimized using the Teaching-Learning-Based Optimization (TLBO) algorithm, a population-based metaheuristic inspired by the real-world teaching-learning interaction in a classroom. TLBO operates in two main phases:

1.  **Teacher Phase:** Learners improve their knowledge based on the teacher's (best solution's) knowledge.
2.  **Learner Phase:** Learners improve their knowledge by interacting with other learners in the population.

This approach allows the model to efficiently explore the parameter space and converge to an optimal solution without requiring specific algorithm parameters (like mutation or crossover rates in genetic algorithms), making it robust and easy to implement.

## 5. Results and Discussion
![Screenshot 2025-06-02 192425](https://github.com/user-attachments/assets/e7afe259-21a0-40d0-aa70-fd04c01f62f9)
![Screenshot 2025-06-02 192434](https://github.com/user-attachments/assets/e487e66d-4276-4ea2-8fef-df9236cd4778)

The TLBO-optimized FFNN model demonstrated superior accuracy in short-term energy demand forecasting. The key performance metrics are summarized below:

| Metric         | Value      | Unit      |
| :------------- | :--------- | :-------- |
| $R^{2}$ Score  | 0.96355    | -         |
| RMSE           | 5.85637    | kWh       |
| MAE            | 4.02526    | kWh       |


![Screenshot 2025-06-02 192206](https://github.com/user-attachments/assets/351c89fd-51ff-4460-9f8c-def637ac9d22)

These metrics indicate a strong predictive performance, with the model explaining over 96% of the variance in energy consumption and maintaining low average errors. The specialized data-sampling strategy proved effective in enhancing the model's generalization, particularly for higher energy consumption values, which are critical for demand management systems.

## 6. Reproducibility and Code Availability

All supporting code is provided in the accompanying `.ipynb` files. The repository includes:

-   Data loading and preprocessing scripts
-   Visualization notebooks
-   TLBO implementation code
-   Model evaluation metrics

The codebase is fully reproducible and can be extended to support other optimizers or forecasting horizons.

## 7. Conclusion

This study demonstrates that a TLBO-optimized FFNN model can provide highly accurate short-term energy forecasts with minimal parameter tuning. The custom sampling and careful preprocessing were instrumental in improving the model's generalization, especially for high-demand scenarios. This approach offers a robust and practical solution for energy demand management.
