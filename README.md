# Automated-Football-Match-Analysis-using-Advanced-Computer-Vision


## Introduction
This project employs advanced computer vision techniques like YOLOv8 and OpenCV to identify and track players, referees, and footballs in videos. I have used different techniques in order to process video frame for including various components such as Speed and Distance Travelled for players. <br><br>
In addition, we also employed K-means clustering for pixel segmentation, enabling us to assign players to their respective teams based on the color of their jerseys. This data will allow us to calculate the percentage of ball possession for each team during a match, providing valuable insights into the game dynamics.<br><br>
I also incorporated optical flow techniques to measure the movement of the camera between frames, which will facilitate accurate tracking of player movements. To further refine our measurements, we will implement perspective transformation to depict the depth and perspective of the scene. This will allow us to measure player movements in real-world units (meters) instead of pixels, providing a more realistic representation of the game.<br><br>
Finally, we will compute the speed of each player and the total distance they cover during the match. This comprehensive project encompasses a wide range of concepts and tackles real-world challenges, making it an excellent learning opportunity for both novice and seasoned machine learning engineers. It is a blend of theory and practice, designed to provide a hands-on experience in applying machine learning techniques to solve complex problems.<br><br>
Here, we will also be fine-tuning YOLO model on [football dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1) before video processing to enhance its performance and ensure precise detection.


This project is inspired from a kaggle competition [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout).<br>
<br>
![Screenshot](https://github.com/user-attachments/assets/d55bb7db-7a0a-44e5-9ad3-5fcb694eb8ae)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models
- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video
-  [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas
