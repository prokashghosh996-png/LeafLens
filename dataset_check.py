from pathlib import Path
import json
import cv2

def count_images(class_folder):
    image_count = 0
    for entry in class_folder.iterdir():
        if entry.is_file():
            if entry.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_count += 1


    return image_count 
def check_images(image_paths):
    loaded_image = 0
    failed_image = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print("Failed! - ",image_path.name)
            failed_image+=1
        else:
            loaded_image+=1

    return loaded_image,failed_image 
def get_image_paths(class_folder):
    image_paths = []
    for entry in class_folder.iterdir():
        if entry.is_file():
            if entry.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_paths.append(entry)
    return image_paths                  

                      
folder = Path(r"C:\Users\USER\Desktop\leaf-image-procession\leaf-image-procession\archive\plantvillage dataset\color")


class_folders = []



for entry in folder.iterdir():
    if entry.is_dir():
        class_folders.append(entry)#storing entry in the list to access the images full path later

print("Total classes:", len(class_folders))        

dataset = {}
for class_folder in class_folders:
    count_1 = count_images(class_folder)
    
    dataset[class_folder.name] = count_1
    

#print(dataset)
total_images = sum(dataset.values())
print("Total images:", total_images)
counts_text = json.dumps(dataset, indent=4)

output_path = Path(r"D:\LeafLens\dataset_counts.json")
output_path.write_text(counts_text, encoding="utf-8")

validation_results = {}

for class_folder in class_folders:
    image_paths = get_image_paths(class_folder)
    loaded, failed = check_images(image_paths)
    validation_results[class_folder.name] = {
        "Loaded": loaded ,
        "Failed": failed
    }

    #print("Loaded:", loaded)
    #print("Failed:", failed)

#results_text = json.dumps(validation_results, indent=4)
#results_path = Path(r"D:\LeafLens\image_readability_report.json")
#results_path.write_text(results_text, encoding="utf-8")


report_path = Path(r"D:\LeafLens\image_readability_report.json")
report = json.loads(report_path.read_text(encoding="utf-8"))

print("Classes in report:", len(report))
selected_classes = [
    "Apple___healthy",
    "Apple___Apple_scab"
]
for class_name in selected_classes:
    print(class_name,dataset[class_name])

