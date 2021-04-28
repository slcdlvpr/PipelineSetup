# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The classification goal of this project is to predict if the client will subscribe (yes/no) a term deposit (variable y).  The data used for the project was collected from 
direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 


The project was done both with a hyperparameter run and a autoML run with the objective to compare the results using both methods for accuracy.  The most accurate method was VotingEnsemble with an accuracy of .9164.  This was model was obtained as a result of the AutomML run. 

Below is a diagram showing the major componenets of the project. 

<img src ="https://github.com/slcdlvpr/PipelineSetup/blob/main/Images/Summary.JPG"/>


## Scikit-learn Pipeline
The major steps for preparing the pipeline are common for all machine learning.  Import, Clean, Split Data and select a model 
<ol>
  <li>Import and Clean and Split the Data
    <ul>
      <li> Import data using <i>TabularDatasetFactory</i> </li>
      <li> Cleaning of data -  handling NULL values, processing dates </li>
      <li> Splitting the cleaned data into train and test data </li>
      <li> For this part of the experiment we selected scikit-learn logistic regression model for classification </li> 
    </ul>
  </li><br>
  <li>Setup the estimator and pass that data to the Hyperdrive</li><br>
  <li> Configuration of Hyperdrive 
    <ul>
      <li> Setup of parameter sampler </li>
      <li> Setup of primary metric </li>
      <li> Setup of early termination policy </li>
      <li> Setup of estimator (SKLearn) </li>
      <li> Setup resources </li>
   </ul>
  </li><br>  
  <li>Save the trained optimized model</li>
</ol>

<p>As specified in the project requirements, we have used logistic regression model for our binary classification problem and Hyperdrive tool to choose the best hyperparameter values from the range of values provided. 
The best Model from the Hyperdrive run is listed below. </p> 

<strong>Best Hyperdrive Model</strong>
<i>ID :  HD_5355379d-5443-42ae-9375-61cf4708bf81_1
Metrics :  {'Regularization Strength:': 10.0, 'Max iterations:': 50, 'Accuracy': 0.914921598381386} </i>

<img src = "https://github.com/slcdlvpr/PipelineSetup/blob/main/Images/hyperparameter.JPG" />



<strong>Parameter Sampling</strong>
I used RandomParameterSampling because it supports both discrete and continuous hyperparameters. It supports early termination of low-performance runs and supports early stopping policies.  In random sampling, hyperparameter values are randomly selected from the defined search space. This sampling technique was shown to be comparable to full grid sampling results in earlier labs. 

## HyperParameter Definitions
In our random sampling approach the following are explanations of the hyperparameters used;  C: smaller values specify stronger regularization, 
max_iter : maximum number of iterations taken for the solvers to converge values are randomly selected from the defined value ranges. 


<strong>Early Stopping Policy</strong>
<p> The early stopping policy I chose was Bandit Policy because it is based on slack factor and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor compared to the best performing run. <a href = 'https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py&preserve-view=true#&preserve-view=truedefinition'> More Info</a></p



## AutoML Setup
AutoML setup process was similar to setup for the hyperparameter run. 
<ol>
  <li> Import data using <i>TabularDatasetFactory</i></li>
  <li> Cleaning of data -  handling NULL values etc. </li>
  <li> Splitting of data into train and test data </li>
  <li> Configuration of AutoML parameters </li>
  <li> Save the best model generated </li>
</ol>

## AutoML 

The below explanation gives the details of the best model prediction by highlighting feature importance values and discovering patterns in data at training time. It also shows differnt metrics and their value for model.

<img src = "https://github.com/slcdlvpr/PipelineSetup/blob/main/Images/AutoML.JPG" />

## Model Explanation:

<img src = "https://github.com/slcdlvpr/PipelineSetup/blob/main/Images/ModelExplanationAML.JPG" />

## Pipeline comparison
The two methods produced very similar results with a slight edge in accuracy going to AutoML.  The AutoML run took longer and used a wider range of potential models to test for the most accurate.  The steps for preparing each run were similar.  Overall the results from each run where very close in Accuracy.  


## Approach Comparsion
Both approaches use similiar preparation steps, but very differnt processes during their respective runs. Regression + Hyperdrive uses a fixed model then uses hyperdrive to find optimal parameters from the range of values provided.  
In AutoML the processes generates different models with their own parameters and then selects best model. The Hyperdrive run was signficantly faster and took about half the time of the AutoML run.  In this experiment the accuracy was comparable between the two Hyperdrive: .914  AutoML: .916.  


## Future work
One of the DATA GUARDRAILS triggered during the AutoML run.  The nature of the trigger was -- Imbalanced data can lead to a falsely perceived positive effect of a model's accuracy -- so there is probably some tweaks that could be made to the split on the training/testing data. 


## Proof of cluster clean up
<img src = "https://github.com/slcdlvpr/PipelineSetup/blob/main/Images/Cleanup.JPG"/>
