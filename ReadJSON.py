import os
import json

"""
The goal of this script is to combine all the annotations for individual clips into one. 

Each Clip has a folder with the name of the clip number and has a directory inside it named "annotations". This contains 
the annotation file in the following format: "{Clip Number}_Annotations.json".

All these individual annotations files needs to be combined into one file before feeding to the detectron2 model.

"""


