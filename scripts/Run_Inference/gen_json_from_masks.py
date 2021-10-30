import glob

from src.create_annotations import *

# Label ids of the dataset
category_ids = {
    "road": 0,
    "sidewalk": 1,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "pole": 5,
    "traffic light": 6,
    "traffic sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10,
    "person": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18
}

# Define which colors match which categories in the images
category_colors = {
    "(128, 64, 128)": 0, # road
    "(244, 35, 232)": 1, # sidewalk
    "(70, 70, 70)": 2, # building
    "(102, 102, 156)": 3, # wall
    "(190, 153, 153)": 4, # fence
    "(153, 153, 153)": 5, # pole
    "(250, 170, 30)": 6, # traffic light
    "(220, 220, 0)": 7, # traffic sign
    "(107, 142, 35)": 8, # vegetation
    "(152, 251, 152)": 9, # terrain
    "(70, 130, 180)": 10, # sky
    "(220, 20, 60)": 11, #person
    "(255, 0, 0)": 12, #rider
    "(0, 0, 142)": 13, # car
    "(0, 0, 70)": 14, # truck
    "(0, 60, 100)": 15, # bus
    "(0, 80, 100)": 16, # train
    "(0, 0, 230)": 17, # motorcycle
    "(119, 11, 32)": 18, # bicycle
}


# Define the ids that are a multiplolygon. In our case: wall, roof and sky
multipolygon_ids = [0, 2, 3, 4, 8, 9, 12, 13, 14, 15, 16, 17, 18]

# Get "images" and "annotations" info 
def images_annotations_info(maskpath):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    
    for mask_image in glob.glob(maskpath + "*.png"):
        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file
        original_file_name = os.path.basename(mask_image).split(".")[0] + ".jpg"

        # Open the image and (to be sure) we convert it to RGB
        mask_image_open = Image.open(mask_image).convert("RGB")
        w, h = mask_image_open.size
        
        # "images" info 
        image = create_image_annotation(original_file_name, w, h, image_id)
        images.append(image)

        sub_masks = create_sub_masks(mask_image_open, w, h)
        for color, sub_mask in sub_masks.items():
            category_id = category_colors[color]

            # "annotations" info
            polygons, segmentations = create_sub_mask_annotation(sub_mask)

            # Check if we have classes that are a multipolygon
            if category_id in multipolygon_ids:
                # Combine the polygons to calculate the bounding box and area
                multi_poly = MultiPolygon(polygons)
                                
                annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

                annotations.append(annotation)
                annotation_id += 1
            else:
                for i in range(len(polygons)):
                    # Cleaner to recalculate this variable
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    
                    annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                    
                    annotations.append(annotation)
                    annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id

if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()
    
    for keyword in ["train", "val"]:
        mask_path = "dataset/{}_mask/".format(keyword)
        
        # Create category section
        coco_format["categories"] = create_category_annotation(category_ids)
    
        # Create images and annotations sections
        coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

        with open("output/{}.json".format(keyword),"w") as outfile:
            json.dump(coco_format, outfile)
        
        print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))
