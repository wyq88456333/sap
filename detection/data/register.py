from detectron2.data.datasets import register_pascal_voc, register_coco_instances
from pathlib import Path

dataset_base_dir = Path(__file__).parent.parent.parent / 'datasets'



dataset_dir = str(dataset_base_dir/ 'itri-taiwan-416-VOCdevkit2007')
classes = ('person', 'two-wheels', 'four-wheels')
years = 2007
split = 'train' # "train", "test", "val", "trainval"
meta_name = 'itri-taiwan-416_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)
split = 'val'
meta_name = 'itri-taiwan-416_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)


dataset_dir = str(dataset_base_dir/ 'tokyo-320-v2-VOCdevkit2007')
classes = ('person', 'two-wheels', 'four-wheels')
split = 'train' # "train", "test", "val", "trainval"
meta_name = 'tokyo-320-v2_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)
split = 'val'
meta_name = 'tokyo-320-v2_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)


dataset_dir = str(dataset_base_dir/ 'tokyo-320-test-only-VOCdevkit2007')
split = 'test' # "train", "test", "val", "trainval"
classes = ('person', 'two-wheels', 'four-wheels')
years = 2007
meta_name = 'tokyo-320_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

json_file_path = dataset_base_dir/'tokyo-320-v2-VOCdevkit2007'/ 'tokyo-320-v2-val-coco-instances.json'
image_root =  dataset_base_dir/'tokyo-320-v2-VOCdevkit2007'/'JPEGImages'
split = 'cpaug'
meta_name = 'tokyo-320_{}'.format(split)
meta_data = {'thing_classes':['person','two-wheels','four-wheels']}
register_coco_instances(meta_name, meta_data, json_file_path, image_root)

json_file_path = dataset_base_dir/'Cityscapes-coco'/'instancesonly_filtered_gtFine_train.json'
image_root =  dataset_base_dir/'Cityscapes-coco'/'train'
split = 'train'
meta_name = 'cityscapes_{}'.format(split)
register_coco_instances(meta_name, {}, json_file_path, image_root)

# json_file_path = dataset_base_dir/'Cityscapes-coco'/'instancesonly_filtered_gtFine_val.json'
# image_root =  dataset_base_dir/'Cityscapes-coco'/'val'
# split = 'val'
# meta_name = 'cityscapes_{}'.format(split)
# register_coco_instances(meta_name, {}, json_file_path, image_root)

json_file_path = dataset_base_dir/'Foggy-cityscapes-coco'/'instancesonly_filtered_gtFine_train.json'
image_root =  dataset_base_dir/'Foggy-cityscapes-coco'/'train'
split = 'train'
meta_name = 'foggy-cityscapes_{}'.format(split)
register_coco_instances(meta_name, {}, json_file_path, image_root)

# json_file_path = dataset_base_dir/'Foggy-cityscapes-coco'/'instancesonly_filtered_gtFine_val.json'
# image_root =  dataset_base_dir/'Foggy-cityscapes-coco'/'val'
# split = 'val'
# meta_name = 'foggy-cityscapes_{}'.format(split)
# register_coco_instances(meta_name, {}, json_file_path, image_root)

dataset_dir = str(dataset_base_dir/ 'Foggy-cityscapes-coco'/'VOC2007')
split = 'val' # "train", "test", "val", "trainval"
classes = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
years = 2007
meta_name = 'foggy-cityscapes_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

# dataset_dir = str(dataset_base_dir/ 'Foggy-Cityscapes-VOCdevkit2007')
# split = 'train' # "train", "test", "val", "trainval"
# classes = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorbike', 'bicycle')
# years = 2007
# meta_name = 'foggy-cityscapes_{}'.format(split)
# register_pascal_voc(meta_name, dataset_dir, split, years, classes)

# dataset_dir = str(dataset_base_dir/ 'Foggy-Cityscapes-VOCdevkit2007')
# split = 'test' # "train", "test", "val", "trainval"
# classes = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorbike', 'bicycle')
# years = 2007
# meta_name = 'foggy-cityscapes_{}'.format(split)
# register_pascal_voc(meta_name, dataset_dir, split, years, classes)

json_file_path = '/home/xunxun/workspace/sada/sa-da-faster-master/maskrcnn_benchmark/datasets/spine/3yuan/train_one.json'
image_root =  '/home/xunxun/workspace/sada/sa-da-faster-master/maskrcnn_benchmark/datasets/spine/3yuan/trainimages'
split = 'train'
meta_name = 'spines_3yuan_{}'.format(split)
register_coco_instances(meta_name, {}, json_file_path, image_root)

json_file_path = '/home/xunxun/workspace/dcdet/SAPN/sap-da-detectron2/datasets/mianyang/coco_miangyang_1600_train.json'
image_root =  '/home/xunxun/workspace/dcdet/SAPN/sap-da-detectron2/datasets/mianyang/1600valimages'
split = 'train'
meta_name = 'spines_mianyang_{}'.format(split)
register_coco_instances(meta_name, {}, json_file_path, image_root)

json_file_path = '/home/xunxun/workspace/dcdet/SAPN/sap-da-detectron2/datasets/mianyang/coco_miangyang_1600_val.json'
image_root =  '/home/xunxun/workspace/dcdet/SAPN/sap-da-detectron2/datasets/mianyang/1600valimages'
# json_file_path = '/home/xunxun/workspace/sada/sa-da-faster-master/maskrcnn_benchmark/datasets/spine/mianyang/coco_sp_val.json'
# image_root = '/home/xunxun/workspace/sada/sa-da-faster-master/maskrcnn_benchmark/datasets/spine/mianyang/valimages'
split = 'val'
meta_name = 'spines_mianyang_{}'.format(split)
register_coco_instances(meta_name, {}, json_file_path, image_root)

json_file_path = '/home/xunxun/workspace/yh_det_pro/mm_dt/related_files/linyi/linyi_60_hash.json'
image_root =  '/home/xunxun/workspace/yh_det_pro/mm_dt/data/linyi/anned_ori/total_hash'
split = 'val'
meta_name = 'spines_linyi_{}'.format(split)
register_coco_instances(meta_name, {}, json_file_path, image_root)