---
layout: post
title: WIP- Computer Vision Project- Airplane Contrail Detection
subtitle: Masters Capstone Project
cover-img: /assets/img/index.jpg
tags: [AWS]
---

This is my capstone project for my Masters from George Mason University. This project was done in a group setting. While I was involved in every aspect of the project to some degree, I was more involved in some aspects than others. This is a very simplified version of a 16-week long project that culminated in a 50 page report. 

# Context
Contrails are are created when airplanes fly through an ice supersaturated regsion (ISSR), which have sufficiently cold and humid conditions to turn the exhaust gas discharged from the airplane into ice crystals. These contrails appear as thin white lines that form behind the plane, and many only last a few seconds. However, certain conditions cause these contrails to remain longer. Contrails that last up to 10 minutes are called short-lived contrails, and contrails that last longer than 10 minutes (and up to 10 hours) are called long-lived contrails. When a long-lived contrail spreads out and loses it's linear shape, it is called a cirrus contrail. 

Long-lived contrails and cirrus contrails are the primary contributors to global warming. Aviation generates 4% of global warming, and since more than half of the aviation industry's contribution to gloval warming is from contrails, contrails contribute 2%of the total anthropogenic global heating.

Only about 10% of flights create long-lived and cirrus contrails. These contrails can be avoided by flying at a different altitude when there is an ISSR. However, we do not currently have an accurate way to detect where these ISSRs occur. 

This project aims to identify the correlations between weather patterns and contrail appearance so that we can better predict when an ISSR can occur and divert airplanes accordingly. This is one of the few ways we can see an immediate affect on global warming, so this is a very important topic to understand.

To do this, our computer vision model had to be able to accurately identify contrails in images, as well as the type of contrails that appear and how many of each type.

## The Datasets

Most of our datasets were relatively clean and needed minimal pre-processing. The Sky Images Dataset and Training Labels Spreadsheet are not publicly available datasets.

# Sky Images Dataset
This dataset is a set of images of the sky taken near Dulles International Airport. The images were taken every hour on the hour starting July 31, 2022. Images were still being added to the dataset at the time of this project.
The dataset consists of each image and a key, which is the name of the image file. The names are based on the date and time the image was taken, but multiple naming conventions were used that had to be standardized. These images are uploaded to a google drive.

# Training Labels Spreadsheet
Many images were labeled by experts and put into a spreadsheet. Some of the columns in this spreadsheet are:
- Key: This key aligns to the file names of the images.
- Exclude indicator: An indicator to include or exclude the image in the algorithm. Some images were excluded due to poor quality.
- Long-lived contrail count: The number of long-lived contrails identified in the image.
- Cirrus contrail count: The number of cirrus contrails identified in the image. 
- Day cirrus indicator: An indicator of whether any cirrus clouds were present at any point that day. Cirrus clouds also occur in ISSR regions, similar to long-lived and cirrus contrails.

# Ground Weather Dataset
This dataset, downloaded from the Visual Crossing weather data site, has information on ground weather. Some of the relevant columns include:
- Date and time
- Temperature: Ground temperature in Farenheit. 
- Dew: Dew point in Farenheit. 
- Humidity: Relative humidity percentage. 
- Precipitation: How much precipitation occured in inches. 
- Wind speed: The maximum wind speed in miles per hour. 



# Data Storage and Model Deployment Using AWS
## Data storage 

![]({{ "/assets/table1.png" | absolute_url }})

   _Table 1_

## Model Deployment


# System Architecture

# Computer Vision Algorithms and Roboflow
## Annotations with VOTT
## Training the Roboflow Model

# Statistical Testing

# Machine Learning

# Visualizations and Dashboard Development

# Findings


#### Thank you so much for reading my project. I appreciate your time. If you have any advice for improving this project, I would love to hear it! Please email me at ahowe615@gmail.com with any comments.
