# Automated-Football-Match-Analysis-using-Advanced-Computer-Vision


## Introduction
This project employs advanced computer vision techniques like YOLOv5 and OpenCV to identify and track players, referees, and footballs in videos. I have used different techniques in order to process video frame for including various components such as Speed and Distance Travelled for players. <br><br>
In addition, we also employed K-means clustering for pixel segmentation, enabling us to assign players to their respective teams based on the color of their jerseys. This data will allow us to calculate the percentage of ball possession for each team during a match, providing valuable insights into the game dynamics.<br><br>
I also incorporated optical flow techniques to measure the movement of the camera between frames, which will facilitate accurate tracking of player movements. To further refine our measurements, we will implement perspective transformation to depict the depth and perspective of the scene. This will allow us to measure player movements in real-world units (meters) instead of pixels, providing a more realistic representation of the game.<br><br>
Finally, we will compute the speed of each player and the total distance they cover during the match. This comprehensive project encompasses a wide range of concepts and tackles real-world challenges, making it an excellent learning opportunity for me. <br><br>
Here, we will also be fine-tuning YOLO model on [football dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1) before video processing to enhance its performance and ensure precise detection.


This project is inspired from a kaggle competition [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout).<br>
<br>
![Screenshot](https://github.com/user-attachments/assets/d55bb7db-7a0a-44e5-9ad3-5fcb694eb8ae)

## Techniques Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective

## Trained Models
I have already fine tuned the YOLOv5 on Roboflow dataset which can be accessed from:
- [Trained Yolo v5](https://huggingface.co/abhi1304/Automated-Football-Analysis/tree/main)

## Results
This model tracks all the players, referee and ball quite well. Refer to below videos for results.
<br>
-  [Input Video 1](https://drive.google.com/file/d/1R9Cd92g5cNvf7jV0Dp7KDC10CkvIMoS5/view?usp=sharing)<br>
[output_Video_1](https://drive.google.com/file/d/1BS1fFRF3qfu7vuk-gRCA6EY-jeDk8i-e/view?usp=sharing)<br><br>
-  [Input Video 2](https://drive.google.com/file/d/1B8Km-KgHnwUrcPRRWNdYpBRDV1Ws0oZc/view?usp=sharing)<br>
[output_Video_2](https://drive.google.com/file/d/1oQtMA3su17xPhCRRqyTSxVxgDMSjDUE-/view?usp=sharing)

## Project Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas
