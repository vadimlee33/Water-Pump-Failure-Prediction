# Water Pump Failure Prediction
## Project overview
The goal of the project is to predict the failure of a water pump that experienced frequent failures during the spring and summer of 2018.

The dataset for this project was collected from the pump and consists of 52 sensors that measure various physical properties of the system.

## Tools
In this project we use a lot of different libraries.
#### Data Preprocesing: 
- Data Scaling: MinMaxScaler (from scikit-learn)
- Missing Data Imputation: RandomSampleImputer (from feature-engine)
- Feature Selection: DropCorrelatedFeatures, DropDuplicateFeatures, DropCorrelatedFeatures (from feature-engine)
- Also we use Pipelines to provide a convenient way to organize the preprocessing steps in a structured way and apply them to the input data in a single call.
#### For model training we use Machine Learning Algorithms and Deep Learning(ANN):
- We use Machine Learning Algorithms from open-source library scikit-learn.
- ML Algorithms: LogisticRegression, KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBoost Classifier.
- In case of Deep Learning we use TensorFlow.
- Also we find hyperparameters use Grid Search for ML Algorithms and Keras Tuner for Deep Learning.
#### Metrics for Evaluation:
We are working with classification problem, thats why we use these metrics:
- Accuracy
- Roc-auc
- Recall
- F1
- Precision
- Confusion Matrix
All of them we get from scikit-learn library.
