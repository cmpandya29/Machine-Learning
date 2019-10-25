# Build a linear regression model for iris data set

Problem: Build a linear regression model for iris data set
Data: Iris flower data set consists of total 150 samples. Out of 150, 50 samples from each of three species of Iris (Setosa, virginica and versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Below integer values map to the each of the label type:
Iris-setosa → 0, Iris-versicolor → 1 and Iris-virginica → 2
Method:
1. Data collection:
This program use built in python function to read “iris.data” file.
2. Data shuffling:
After reading the input data file, program will shuffle the data and write it to another file name called “shuffled_data.txt”. The reason for data shuffling is that the input file contains labeled attribute in sequence so if we use K-fold method for cross validation then the test data divided chunks can have same type of labeled attribute together.
For an example, if the value of k is 3 in k-fold method then data will be divided into 3 chunks with 50 samples in it. In that case each chunk has same type of classification label. So for this case, the accuracy will be very less as the model will be trained on two types of values only and will be testing third value on that model.
3. Model Creation:
Model is created using below equation:
β = (X`X)−1X`Y
Where β is model parameter matrix, X is predictor variable matrix and Y is response variable matrix.
4. Training and testing the data:
For the testing of the model, K-fold cross validation method is used in the project. Three different values of k (3,5,10) is used for the testing. For every iteration in k-fold method, above formula mentioned in step 3 is used to find out the model parameter using training data. Later, same parameter is used to find value of response variable of test data.
The big advantage that comes with K-Fold Cross Validation is that its much less prone to selection bias since training and testing is performed on several different parts. In particular, if we increase the value of K, we can be even more sure of the robustness of our model since we’ve trained and tested on so many different sub-datasets.

