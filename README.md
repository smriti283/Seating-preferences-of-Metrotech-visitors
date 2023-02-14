# Seating-preferences-of-Metrotech-visitors
Analysis of seating preferences through temperature, humidity and video data.


## Overview


The focus of this project is to better understand what measurable factors play a role in where citizens choose to sit in public spaces. Public space and outdoor seating has become an especially important topic in New York due to the airborne COVID-19 pandemic; outdoor space became the only place in the last year plus for folks to safely socialize. Even early in the pandemic, many experts knew that being outside was safer than inside as people flocked to parks and outdoor recreation areas. 

Though the pandemic looks to be calmer in 2021 than 2020, people are still gathering outdoors, and New York, just like other metropolitan areas, will have to keep up in its efforts to satiate this growing demand. 

Our hope is that by using the sensors available to us, we can quantitatively identify why some seating areas in MetroTech Plaza are preferable to others - and then generalize these takeaways for the city to take advantage of across the 5 boroughs.

We decided to cconduct this study on the Lawrence St alleyway of the Metro Tech Plaza in Downtown Brooklyn. A 2016 New York Times and NYU Tandon collaboration provides clear evidence that the plaza receives different amounts of light in the springtime throughout the day (“Mapping the Shadows of New York City: Every Building, Every Block”). The Lawrence St. alleyway that we were able to study is mostly in shade throughout the Spring/Fall, but during approximately 10% of the day receives direct sunlight


![Study](https://user-images.githubusercontent.com/78453405/218828955-2b99713d-2c91-47eb-b3dc-137c0566b4a2.png)


## Hypothesis

In the moderate spring weather, people will prefer to utilize seats and tables that are in directly-lit, warmer areas during the daytime compared to shaded areas. Using existing open data, we anticipate that the directly-sunlit part of the plaza will be more utilized than shaded counterparts.


## Methodology

1. Data Collection
2. Data Processing - Video Imagery
3. Data Processing - Temperature and Humidity
4. Hypothesis Testing

### 1. Data Collection

We recorded video imagery using a Samsung Galaxy S10 mobile phone (with 1080p video at 30 fps) from the northeast corner of the alleyway labeled in Figure 2a (adjacent to the Five Guys store). We found a relatively unobstructed view of both shaded and sunlit bench areas. Approximately 20 minutes of video data was captured on Wednesday, May 12, 2021 from 12:12 PM  to 12:32 PM.  


![Lawrence St Corner 2](https://user-images.githubusercontent.com/78453405/218829137-fa96870b-ee96-45fd-beda-3fe5a5976243.png)


For temperature and humidity, recordings were taken between 12pm and 12:39pm (EST) for between 7 and 9 minutes of each section (excluding transition time before and after each measurement period). This data was collected via a Raspberry Pi 3 (Model B+) and DHT11 Humidity Sensor to get readings of the seating locations in both the sunlit and shaded areas.


### 2. Data Processing - Video Imagery


We used the Mobilenet-SSD Model to detect people and chairs in the video, and track the behavior of people in terms of their seating choices in sunlit and shaded areas. The Mobilenet-SSD model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. This model is implemented using the Caffe framework. The model input is a blob that consists of a single resized image frame in BGR order. Thereafter, mean subtraction of given BGR values is performed to help combat illumination changes in the frame, before passing the image blob into the network. The model output is a typical vector containing the tracked object data. 


We made many adjustments and modifications to the script to suit the needs of our Video data, and to make sure our objects were tracked efficiently. These include, but not limited to:
- Detection of seating along with people
- Confidence threshold
- Bounding boxes display
- Direction and Detection


Since the algorithm is trained to detect walking/standing people, detection of sitting people was a challenge. Because of this, we relied on manual counting to analyze the seating preferences of people. However, we recognize the need of an automated counter for more extensive study on the topic. To improve the performance of the counter, the algorithm must be trained on seated people for detection, which is a time-intensive and computationally expensive process.


### 3. Data Processing - Temperature and Humidity


The temperature and humidity data collected was logged and extracted as 22 distinct CSVs, representing each session of measurement via a Raspberry Pi 3 and corresponding sensors. Each CSV featured timestamps and measures for temperature and humidity every 2 seconds. Collation of these CSVs, data cleaning and visualization was completed in pandas and matplotlib in Jupyter notebooks with Python.


## Analysis and Results


### Video Imagery

Our video tracking results show that in the span of 20 minutes at midday on May 12, 2021, more than 200 people walked past the captured area. Out of them, two people decided to sit in the plaza. The first person was resting and the 2nd person sat for a meal.

Both of them chose the sunlit side for sitting as opposed to the shaded side. A third person briefly proceeded towards the chairs but only to check his belongings and did not sit. However, he also chose the sunlit side for this purpose.

The sunlit side was thus favored over the shaded side 100% of the time during the short span of collected data. 


![mbnet ssd](https://user-images.githubusercontent.com/78453405/218831213-04c390aa-4d05-482a-a359-7850ba3de90b.png)


### Temperature and Humidity

We see noticeable differences between the directly lit and shaded seating areas in terms of temperature and humidity.

![temperature](https://user-images.githubusercontent.com/78453405/218834011-1fab12b9-b51c-440b-b8a9-d45fbe614e34.png) 
*Temperature - Metro Tech Seating Locations*



While neither of these findings are novel discoveries, we were surprised to see such high variance between the two lighting types. Humidity is much lower in the sun, while temperature is much lower in the shade. This matches common sense about shade vs direct light, though the delta between the two is noteworthy. The difference of mean temperatures in either section was 4.3 degrees Celsius, which can represent a noticeable difference in human comfort.

