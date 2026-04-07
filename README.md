# XAI Final assignment: SHAP/DeepLIFT comparison 
In this project the XAI methods SHAP and DeepLIFT were compared on a tremor severity classification dataset.  

## Data
The data consisted of 6 features containing values from sensor data (gyroscope and accelerometer) with a label that indicated tremor severity. In the code, the labels
were substracted by 1 for simplicity, so, e.g., tremor severity 'Mild' went from value 1 to 0. 

## Use 
Download the files that have been put in the repository. These include the code, the saved model path for reproducibility, and the dataset that was retreived from 
https://www.kaggle.com/datasets/manahilsiddique/parkinsons-imu-tremor-severity-dataset. Change the file path in the XAI_CNN_COMPLETE.py paths to your own directory. Install the requirements .txt and run the command pip install -r requirements.txt.

The code should be cpu friendly. However, if gpu can be used this is preferred. If the code runs too long with KernelShap of 400, change it to a smaller number (e.g, 50) for code demonstration.


