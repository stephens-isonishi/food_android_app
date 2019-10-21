# Food Classification Android App 
An android app that can recognize food. Focus is more on the backend/model.

## Background
It would be useful to have an app that can recognize a variety of different foods through a smartphone app. The thought process was, find a food dataset, train a model on it, find an app framework, insert model on it, and voilà. 

## Dataset
Found the largest dataset that seemed polished and professional--the [Food-475 Database](http://www.ivl.disco.unimib.it/activities/food475db/). Larger is better right? This database is actually a combination of four datasets that include: UECFOOD256, VIREO, Food-101, and Food-50. Of these, I needed to get permission from the administrator for the Food-101 dataset (a professor at a Chinese university). I had no problem downloading the train/test split spreadsheet, the UECFOOD256 dataset, VIREO, and Food-101, but I could not find any links for the Food-50 dataset. The Food-50 dataset was missing from a certain lab webpage of National Taiwan University which it should have been on. So, I decided to just skip the Food-50 dataset. This may introduce bias because the creators of the Food-475 dataset may have split the data into training and testing in a certain way. 

## Model and Approach
According to papers such as [Hassannejad, Hamid, et al. "Food Image Recognition Using Very Deep Convolutional Networks." Proceedings of the 2nd International Workshop on Multimedia Assisted Dietary Management. ACM, 2016.](https://dl.acm.org/citation.cfm?id=2986042) and [NVIDIA DEEP LEARNING CONTEST 2016, Keun-dong Lee, DaUn Jeong, Seungjae Lee, Hyung Kwan Son (ETRI VisualBrowsing Team), Oct.7, 2016.](https://www.gputechconf.co.kr/assets/files/presentations/2-1650-1710_DL_Contest_%EC%A7%80%EC%A0%95%EC%A3%BC%EC%A0%9C_%EB%8C%80%EC%83%81.pdf), researchers were able to achieve around 90% accuracy on the Food-101 dataset with transfer learning models with Inception V3 and ResNet200. However, these papers never mention anything about mobile deployment so I decided to start off with a simple [SqueezeNet model](https://arxiv.org/pdf/1602.07360.pdf) which is a lightweight model that is suited for mobile development. 

# Training
The model for SqueezeNet is saved as squeezenet_model.py, and the the program that trains the model is squeezenet_main.py. I ran the model on the lab's workstation computer that has 4 NVIDIA GTX 1080TI GPUs with Docker. The system settings were **Python 3.5.1** and **Tensorflow 1.12.0**, and trained for 200 epochs, batch size of 64, and an optimizer of SGD with .001 learning rate. Multi-gpu training was incorporated (commented out in the code for testing an error) in the training, allowing the training to finish within 3 days. The results from training were decent with 72% accuracy and 65% validation accuracy. 

# Conversion to TFLITE


--talk about SqueezeNet
-problems why it didn't work

## Current Approach

## Considering to...

