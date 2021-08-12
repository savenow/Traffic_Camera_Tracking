import json
json_format = {
    "licenses": [{
        "name": "",
        "id": 0,
        "url": ""
    }],
    "info": {
        "contributor": "Vishal Balaji",
        "date_created": "",
        "description": "Escooter Dataset",
        "url": "",
        "version": "",
        "year": ""
    },
    "categories": [{
        "id": 1,
        "name": "Escooter",
        "supercategory": ""
    }]
}
"""
"images":[
    {
        "id":1,
        "width": 1920,
        "height": 1080,
        "file_name":"sdfa.PNG",
        "license":0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0
    }
]

"annotations":[
    {
        "id": 1,
        "image_id": 55,
        "category_id": 1,
        "segmentation": [[]],
        "area": {some area number in float},
        "bbox": [].
        "iscrowd": 0
    }
]
"""
images_list = []
annotations_list = []

for _ in range(5):
    image_dict = {"id": (_+1), "width": 1920, "height":1080, "file_name": str(_) + '.PNG', "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}
    images_list.append(image_dict)

    #anno_dict= {"id": (_+1), "image_id"}


json_format["images"] = images_list

with open("JSON_Testing/Test_3.json", "w") as file:
    json.dump(json_format, file)

print(json_format)