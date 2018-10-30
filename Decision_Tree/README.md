# COL341: Assignment 4
##Support Vector Machine

**Name**

svm - Run the executable program for SVM

**Synopsis**
*Part a*`./svm <part> <tr> <ts> <out> <c_value>`
*Part b*`./svm <part> <tr> <ts> <out> <c_value> <gamma>`
*Part c*`./svm <part> <tr> <ts> <out> <c_value> <gamma>`

**Description**

This program will train svm model using given code on train data, make predictions on test data and write final predictions in given output file.

**Options**

-part  
    Part as per question i.e. a/b/c.  
-tr  
    File containing training data in csv format where 1st entry is the target  
-ts  
    File containing test data in csv format where 1st entry is the target  
-out  
    Output file for predictions. One value in each line.
-c_value  
    C is a regularization parameter that controls the trade-off between maximizing the margin and minimizing the training error.    
-gamma  
    Bandwidth parameter for RBF kernel

**Example**
    
`./svm a DHC_train.csv DHC_test.csv output 10`
`./svm b DHC_train.csv DHC_test.csv output 10 0.01`
`./svm c DHC_train.csv DHC_test.csv output 10 0.01`
    
**Data**

- DHC_train.csv: Train data  
- DHC_test.csv: Test data
    
**Marking scheme**

Marks will be given based on following categories:

1. Part-a/b/c(i): Run/format error (0 points) 
2. Part-a/b/c(i): Runs fine but predictions are incorrect within some threshold (half of specified points) 
3. Part-a/b/c(i): Works as expected with correct predictions(Full points) 
4. Part-c: Relative marking will be done based on specified error
5. For part-c(ii), you can get 0 (wrong plot) and full (plot is as expected)
6. For part-d, You can get full (correct observation) or 0 (wrong observation)

**Checking Program**

Accuracy on test data will be used as an evaluation criterion.

**Submission**

1. Your submission should be "ENTRY_NO.zip".
2. Make sure you clean up extra files/directories such as "__MACOSX"
3. Command "unzip ENTRY_NO.zip", should result in a single directory "ENTRY_NO".

-------------------------------------------------------------------------
-------------------------------------------------------------------------
##Decision Tree

**Name**

dtree - Run the executable program for Decision Tree

**Synopsis**

`./dtree <part> <tr> <vs> <ts> <out> <plot>`

**Description**

This program will train a decision tree model using given code on train data, make predictions on test data and write final predictions
in given output file. You must save the plot also with given png name.

**Options**

-part  
    Part as per question i.e. a or b.  
-tr  
    File containing training data in csv format where 1st entry is the target  
-vs  
    File containing validation data in csv format where 1st entry is the target
-ts  
    File containing test data in csv format where 1st entry is the target  
-out  
    Output file (write your predictions in this file) 
-plot  
    Save your plot in png format to  this location 


**Example**
    
`./dtree a train.csv valid.csv test.csv output plo1.png`
`./dtree b train.csv valid.csv test.csv output plot2.png`

**Data**

- train.csv: Train data
- valid.csv: Validation data
- test.csv: Test data
    
**Marking scheme**

Marks will be given based on following categories:

1. Code: 80% points , Plot: 20% points
2. Run/format error (0 points in both)
3. Runs fine but predictions are incorrect within some threshold (50% in code) 
4. Works as expected with correct predictions (100% in code)
5. For plot:  you can get 0 (wrong plot),half (plot is partially correct) and full (plot is as expected).You have to include plot in your report and generate it at run-time also. 

**Checking Program**

F_Score on test set will be used as evaluation criterion.

**Submission**

1. Your submission should be "ENTRY_NO.zip".
2. Make sure you clean up extra files/directories such as "__MACOSX"
3. Command "unzip ENTRY_NO.zip", should result in a single directory "ENTRY_NO".
