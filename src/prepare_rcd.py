from pathlib import Path
from PIL import Image
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RCD_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "rcd"
    / "Remondis_Contamination_Dataset (RCD)"
)

OUTPUT_PATH = PROJECT_ROOT / "processed" / "rcd_crops"
METADATA_FILE = OUTPUT_PATH / "metadata.csv"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

PLASTIC_OUTPUT = OUTPUT_PATH / "Plastic"

PLASTIC_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


metadata_rows = []

total_objects = 0
saved_crops = 0
skipped_objects = 0
missing_images = 0


# --------------------------------------------------
# Find matching image for a label file
# --------------------------------------------------

def find_image(image_folder, stem):

    for extension in IMAGE_EXTENSIONS:

        image_path = (
            image_folder
            / f"{stem}{extension}"
        )

        if image_path.exists():
            return image_path

    return None


# --------------------------------------------------
# Process training and validation data
# --------------------------------------------------

splits = {
    "training": {
        "image_dir": RCD_PATH / "training" / "image_2",
        "label_dir": RCD_PATH / "training" / "label_2"
    },

    "val": {
        "image_dir": RCD_PATH / "val" / "image",
        "label_dir": RCD_PATH / "val" / "label"
    }
}


for split_name, folders in splits.items():

    image_dir = folders["image_dir"]
    label_dir = folders["label_dir"]

    print(f"\nProcessing RCD {split_name}...")

    label_files = list(
        label_dir.glob("*")
    )

    for label_file in label_files:

        if not label_file.is_file():
            continue


        image_path = find_image(
            image_dir,
            label_file.stem
        )


        if image_path is None:

            missing_images += 1
            continue


        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception:

            missing_images += 1
            continue


        with open(
            label_file,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()


        plastic_index = 0


        for line in lines:

            parts = line.strip().split()


            if not parts:
                continue


            # RCD contains plastic_bag objects

            if parts[0] != "plastic_bag":
                continue


            total_objects += 1


            # Bounding box fields:
            # parts[4] = left
            # parts[5] = top
            # parts[6] = right
            # parts[7] = bottom

            if len(parts) < 8:

                skipped_objects += 1
                continue


            try:

                x1 = int(float(parts[4]))
                y1 = int(float(parts[5]))
                x2 = int(float(parts[6]))
                y2 = int(float(parts[7]))

            except ValueError:

                skipped_objects += 1
                continue


            # Keep coordinates inside image

            x1 = max(
                0,
                min(x1, image.width)
            )

            y1 = max(
                0,
                min(y1, image.height)
            )

            x2 = max(
                0,
                min(x2, image.width)
            )

            y2 = max(
                0,
                min(y2, image.height)
            )


            if x2 <= x1 or y2 <= y1:

                skipped_objects += 1
                continue


            crop_width = x2 - x1
            crop_height = y2 - y1


            if crop_width < 10 or crop_height < 10:

                skipped_objects += 1
                continue


            crop = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


            output_name = (
                f"rcd_{split_name}_"
                f"{label_file.stem}_"
                f"{plastic_index}.jpg"
            )


            output_file = (
                PLASTIC_OUTPUT
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

                "source": "RCD",

                "source_split": split_name,

                "source_image": image_path.name,

                "label_file": label_file.name,

                "source_class": "plastic_bag",

                "common_class": "Plastic",

                "object_index": plastic_index
            })


            saved_crops += 1

            plastic_index += 1


# --------------------------------------------------
# Save metadata
# --------------------------------------------------

with open(
    METADATA_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "crop_path",
        "source",
        "source_split",
        "source_image",
        "label_file",
        "source_class",
        "common_class",
        "object_index"
    ]


    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        metadata_rows
    )


# --------------------------------------------------
# Final report
# --------------------------------------------------

print("\nRCD CROPPING COMPLETE")
print("=" * 50)

print(
    f"Total plastic_bag objects : "
    f"{total_objects}"
)

print(
    f"Saved crops               : "
    f"{saved_crops}"
)

print(
    f"Skipped objects           : "
    f"{skipped_objects}"
)

print(
    f"Missing images            : "
    f"{missing_images}"
)

print(
    f"\nMetadata file             : "
    f"{METADATA_FILE}"
)

print(
    f"\nPlastic crops             : "
    f"{saved_crops}"
)