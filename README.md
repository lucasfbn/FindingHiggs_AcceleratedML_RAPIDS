# Detecting the Higgs boson using accelerated machine learning

The Large Hadron Collider (LHC) is the world's largest particle accelerator and allows physicists to test the predictions of different theories of particle physics. Storing all the data generated by the LHC is not feasible. As a result, technologies are in place to classify the data on whether it holds relevant information or not. Statistical models and machine learning algorithms are among the technologies employed to accomplish this. 

This notebook shows how accelerated machine learning can be used to accomplish this. Accelerated machine learning uses GPUs to speed up processing-intensive operations and can be used to execute pre-processing steps and the training of machine learning algorithms more efficiently.

The used dataset is comprised of around 11 million observations and consists of a variety of properties related to collisions in the LHC. The aim is to distinguish between a signal- or background event. A signal event refers to the detection of the Higgs boson, while a background event is the detection of any other particle. 

To achieve this, three different machine learning algorithms are used: SVM, random forest and XGBoost. After an exploratory data analysis and various pre-processing steps, the selected algorithms will be trained using RAPIDS. Furthermore, hyperparameter tuning over an extensive search space and model evaluation are discussed.