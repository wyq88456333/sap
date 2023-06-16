# import os
# import json
# from xml.etree.ElementTree import Element, SubElement, tostring
# from xml.dom.minidom import parseString
# from pycocotools.coco import COCO

# # Define the COCO and VOC directories
# coco_dir = '/home/xunxun/workspace/sada/sa-da-faster-master/maskrcnn_benchmark/datasets/spine/mianyang'
# voc_dir = '/home/xunxun/workspace/dcdet/SAPN/sap-da-detectron2/voc'

# # Define the class names
# coco_classes = ['0']
# voc_classes = ['background'] + coco_classes

# # Load the COCO dataset
# annFile = os.path.join(coco_dir, 'coco_sp.json')
# coco = COCO(annFile)

# # Create the VOC dataset directories
# for split in ['train', 'val']:
#     split_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
#     os.makedirs(split_dir, exist_ok=True)

# # Convert the annotations to VOC format
# for img_id in coco.imgs:
#     # Load the image file
#     img_info = coco.imgs[img_id]
#     img_path = os.path.join(coco_dir, 'imgs', img_info['file_name'])

#     # Load the annotations
#     ann_ids = coco.getAnnIds(imgIds=[img_id])
#     anns = coco.loadAnns(ann_ids)

#     # Create the VOC annotation file
#     voc_file = os.path.join(voc_dir, 'Annotations', f"{img_info['file_name'][:-4]}.xml")
#     root = Element('annotation')
#     SubElement(root, 'filename').text = img_info['file_name']
#     size_node = SubElement(root, 'size')
#     SubElement(size_node, 'width').text = str(img_info['width'])
#     SubElement(size_node, 'height').text = str(img_info['height'])
#     SubElement(size_node, 'depth').text = '3'

#     for ann in anns:
#         # Get the class name
#         coco_class_id = ann['category_id']
#         coco_class_name = coco.loadCats(coco_class_id)[0]['name']
#         voc_class_id = voc_classes.index(coco_class_name)

#         # Create the VOC annotation node
#         obj_node = SubElement(root, 'object')
#         SubElement(obj_node, 'name').text = coco_class_name
#         SubElement(obj_node, 'pose').text = 'Unspecified'
#         SubElement(obj_node, 'truncated').text = '0'
#         SubElement(obj_node, 'difficult').text = '0'
#         bbox_node = SubElement(obj_node, 'bndbox')
#         SubElement(bbox_node, 'xmin').text = str(int(ann['bbox'][0]))
#         SubElement(bbox_node, 'ymin').text = str(int(ann['bbox'][1]))
#         SubElement(bbox_node, 'xmax').text = str(int(ann['bbox'][0] + ann['bbox'][2]))
#         SubElement(bbox_node, 'ymax').text = str(int(ann['bbox'][1] + ann['bbox'][3]))

#     # Save the VOC annotation file
#     with open(voc_file, 'w') as f:
#         xml_string = parseString(tostring(root)).toprettyxml()
#         f.write(xml_string)
from pycocotools.coco import COCO

# Load the annotation file
ann_file = "/home/xunxun/workspace/dcdet/SAPN/sap-da-detectron2/datasets/mianyang/coco_miangyang_1600_val.json"
coco = COCO(ann_file)

# Validate the annotations
coco.check_annotations()