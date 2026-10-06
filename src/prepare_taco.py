from pathlib import Path
from PIL import Image
import json
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TACO_PATH = PROJECT_ROOT / "datasets" / "taco"
ANNOTATIONS_FILE = TACO_PATH / "annotations.json"

OUTPUT_PATH = PROJECT_ROOT / "processed" / "taco_crops"
METADATA_FILE = OUTPUT_PATH / "metadata.csv"


TACO_MAPPING = {
    "Aluminium foil": "Metal",

    "Other plastic bottle": "Plastic",
    "Clear plastic bottle": "Plastic",
    "Plastic bottle cap": "Plastic",

    "Glass bottle": "Glass",
    "Broken glass": "Glass",
    "Glass cup": "Glass",
    "Glass jar": "Glass",

    "Metal bottle cap": "Metal",
    "Food Can": "Metal",
    "Aerosol": "Metal",
    "Drink can": "Metal",
    "Metal lid": "Metal",
    "Pop tab": "Metal",
    "Scrap metal": "Metal",

    "Other carton": "Cardboard",
    "Egg carton": "Cardboard",
    "Drink carton": "Cardboard",
    "Corrugated carton": "Cardboard",
    "Meal carton": "Cardboard",
    "Pizza box": "Cardboard",

    "Paper cup": "Paper",
    "Magazine paper": "Paper",
    "Tissues": "Paper",
    "Wrapping paper": "Paper",
    "Normal paper": "Paper",
    "Paper bag": "Paper",
    "Paper straw": "Paper",

    "Disposable plastic cup": "Plastic",
    "Foam cup": "Plastic",
    "Other plastic cup": "Plastic",
    "Plastic lid": "Plastic",
    "Other plastic": "Plastic",
    "Plastic film": "Plastic",
    "Six pack rings": "Plastic",
    "Garbage bag": "Plastic",
    "Other plastic wrapper": "Plastic",
    "Single-use carrier bag": "Plastic",
    "Polypropylene bag": "Plastic",
    "Crisp packet": "Plastic",
    "Spread tub": "Plastic",
    "Tupperware": "Plastic",
    "Disposable food container": "Plastic",
    "Foam food container": "Plastic",
    "Other plastic container": "Plastic",
    "Plastic glooves": "Plastic",
    "Plastic utensils": "Plastic",
    "Squeezable tube": "Plastic",
    "Plastic straw": "Plastic",
    "Styrofoam piece": "Plastic",

    "Food waste": "Food Organics",

    "Unlabeled litter": "Miscellaneous Trash",
    "Cigarette": "Miscellaneous Trash",
}


with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


images = {
    image["id"]: image
    for image in data["images"]
}

categories = {
    category["id"]: category["name"]
    for category in data["categories"]
}

annotations = data["annotations"]


OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

for class_name in set(TACO_MAPPING.values()):
    (OUTPUT_PATH / class_name).mkdir(
        parents=True,
        exist_ok=True
    )


metadata_rows = []

total_annotations = len(annotations)
mapped_annotations = 0
saved_crops = 0
skipped_annotations = 0
failed_images = 0


for annotation in annotations:

    category_id = annotation["category_id"]

    if category_id not in categories:
        skipped_annotations += 1
        continue

    taco_class = categories[category_id]

    if taco_class not in TACO_MAPPING:
        skipped_annotations += 1
        continue

    common_class = TACO_MAPPING[taco_class]

    image_id = annotation["image_id"]

    image_info = images.get(image_id)

    if image_info is None:
        failed_images += 1
        continue


    # Exact path from annotations.json

    image_path = TACO_PATH / image_info["file_name"]


    if not image_path.exists():

        failed_images += 1
        continue


    try:

        image = Image.open(
            image_path
        ).convert("RGB")


        # COCO format:
        # x, y, width, height

        x, y, width, height = annotation["bbox"]


        x1 = max(
            0,
            int(x)
        )

        y1 = max(
            0,
            int(y)
        )

        x2 = min(
            image.width,
            int(x + width)
        )

        y2 = min(
            image.height,
            int(y + height)
        )


        if x2 <= x1 or y2 <= y1:
            skipped_annotations += 1
            continue


        crop_width = x2 - x1
        crop_height = y2 - y1


        if crop_width < 10 or crop_height < 10:
            skipped_annotations += 1
            continue


        crop = image.crop(
            (
                x1,
                y1,
                x2,
                y2
            )
        )


        annotation_id = annotation["id"]


        output_name = (
            f"taco_img{image_id}_"
            f"ann{annotation_id}.jpg"
        )


        output_file = (
            OUTPUT_PATH
            / common_class
            / output_name
        )


        crop.save(
            output_file,
            "JPEG",
            quality=95
        )


        metadata_rows.append({

            "crop_path": str(
                output_file.relative_to(
                    PROJECT_ROOT
                )
            ),

            "source": "TACO",

            "source_image_id": image_id,

            "source_image": image_info["file_name"],

            "taco_category": taco_class,

            "common_class": common_class,

            "annotation_id": annotation_id
        })


        mapped_annotations += 1
        saved_crops += 1


    except Exception:

        failed_images += 1


# Save metadata

with open(
    METADATA_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "crop_path",
        "source",
        "source_image_id",
        "source_image",
        "taco_category",
        "common_class",
        "annotation_id"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        metadata_rows
    )


# Count classes

class_counts = {}

for row in metadata_rows:

    class_name = row["common_class"]

    class_counts[class_name] = (
        class_counts.get(
            class_name,
            0
        ) + 1
    )


print("\nTACO CROPPING COMPLETE")
print("=" * 50)

print(
    f"Total annotations       : "
    f"{total_annotations}"
)

print(
    f"Mapped annotations      : "
    f"{mapped_annotations}"
)

print(
    f"Saved crops             : "
    f"{saved_crops}"
)

print(
    f"Skipped annotations     : "
    f"{skipped_annotations}"
)

print(
    f"Failed images           : "
    f"{failed_images}"
)

print(
    f"\nMetadata file           : "
    f"{METADATA_FILE}"
)

print("\nCROPS BY CLASS")
print("=" * 50)

for class_name in sorted(class_counts):

    print(
        f"{class_name:25} : "
        f"{class_counts[class_name]}"
    )