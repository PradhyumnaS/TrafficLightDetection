# Traffic Light Detection and Color Classification Using Yolo v8

![Model Output Traffic Light Image](images/traffic_light.jpg "Traffic Light Detection")

This repository includes three trained Yolo v8 models that are designed to detect traffic lights and classify their colors.  
The different file sizes indicate the size of the model; the available sizes are small, medium and nano. The speeds that each model works at varies depending on the size of the model and the hardware being used. 

This README file provides necessary instructions to load, use, and understand the models in this repository. 

## File Structure 

* `best_traffic_med_yolo_v8.pt`
   The trained medium-sized Yolo v8 model file.
  
* `best_traffic_small_yolo.pt`
  The trained small-sized Yolo v8 model file.
  
* `best_traffic_nano_yolo.pt`
  The trained nano-sized Yolo v8 model file.
  
* `livetest.py`
  This Python script provides a live webcam test for the model that you choose.

## Model Performance 

| Model Size | Execution Time(MS) | Hardware Used |
| --- | --- | --- |
| Nano   | 43.7 | Mac M1 Max (CPU Only) |
| Small  | 72.1 | Mac M1 Max (CPU Only) |
| Medium | 130.2 | Mac M1 Max (CPU Only) |


## Usage
Run `livetest.py` script file to perform a live test of each model. Make sure to include the paths to desired `.pt` file in the script to ensure that the proper model is loaded.
