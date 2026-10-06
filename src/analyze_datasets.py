from pathlib import Path
from PIL import Image
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REALWASTE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "realwaste"
    / "realwaste-main"
    / "RealWaste"
)

TACO_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "taco"
)

RCD_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "rcd"
    / "Remondis_Contamination_Dataset (RCD)"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def count_images(folder):
    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_image_info(folder):
    total = 0
    corrupted = 0
    extensions = {}

    if not folder.exists():
        return {
            "total": 0,
            "corrupted": 0,
            "extensions": {}
        }

    for file in folder.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total += 1

        extension = file.suffix.lower()

        extensions[extension] = (
            extensions.get(extension, 0) + 1
        )

        try:
            with Image.open(file) as image:
                image.verify()

        except Exception:
            corrupted += 1

    return {
        "total": total,
        "corrupted": corrupted,
        "extensions": extensions
    }


def analyze_realwaste():

    print("\n========== REALWASTE ==========")

    if not REALWASTE_PATH.exists():

        print("RealWaste path not found:")
        print(REALWASTE_PATH)

        return {}

    classes = {}
    total = 0

    for folder in sorted(REALWASTE_PATH.iterdir()):

        if not folder.is_dir():
            continue

        count = count_images(folder)

        classes[folder.name] = count

        total += count

        print(
            f"{folder.name}: {count}"
        )

    print(
        f"\nTotal images: {total}"
    )

    return {
        "classes": classes,
        "total": total
    }


def analyze_taco():

    print("\n========== TACO ==========")

    if not TACO_PATH.exists():

        print("TACO path not found:")
        print(TACO_PATH)

        return {}

    image_folder = (
        TACO_PATH / "images_raw"
    )

    annotation_file = (
        TACO_PATH / "annotations.json"
    )

    image_info = get_image_info(
        image_folder
    )

    print(
        f"Images: {image_info['total']}"
    )

    print(
        f"Corrupted images: "
        f"{image_info['corrupted']}"
    )

    print(
        f"Extensions: "
        f"{image_info['extensions']}"
    )

    result = {
        "image_info": image_info
    }

    if not annotation_file.exists():

        print(
            "annotations.json not found."
        )

        return result

    try:

        with open(
            annotation_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        categories = data.get(
            "categories",
            []
        )

        annotations = data.get(
            "annotations",
            []
        )

        images = data.get(
            "images",
            []
        )

        print(
            f"Annotated images: "
            f"{len(images)}"
        )

        print(
            f"Annotations: "
            f"{len(annotations)}"
        )

        print(
            f"Categories: "
            f"{len(categories)}"
        )

        print("\nTACO categories:")

        category_data = {}

        for category in categories:

            category_id = category.get(
                "id"
            )

            category_name = category.get(
                "name"
            )

            category_data[
                str(category_id)
            ] = category_name

            print(
                f"{category_id}: "
                f"{category_name}"
            )

        result["annotated_images"] = (
            len(images)
        )

        result["annotations"] = (
            len(annotations)
        )

        result["categories"] = (
            category_data
        )

    except Exception as error:

        print(
            "Could not read "
            "annotations.json:"
        )

        print(error)

    return result


def count_label_files(folder):

    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.iterdir()
        if file.is_file()
    )


def analyze_rcd():

    print("\n========== RCD ==========")

    if not RCD_PATH.exists():

        print("RCD path not found:")
        print(RCD_PATH)

        return {}

    training_images = (
        RCD_PATH
        / "training"
        / "image_2"
    )

    training_labels = (
        RCD_PATH
        / "training"
        / "label_2"
    )

    validation_images = (
        RCD_PATH
        / "val"
        / "image"
    )

    validation_labels = (
        RCD_PATH
        / "val"
        / "label"
    )

    training_image_info = (
        get_image_info(
            training_images
        )
    )

    validation_image_info = (
        get_image_info(
            validation_images
        )
    )

    training_label_count = (
        count_label_files(
            training_labels
        )
    )

    validation_label_count = (
        count_label_files(
            validation_labels
        )
    )

    print(
        f"Training images: "
        f"{training_image_info['total']}"
    )

    print(
        f"Training labels: "
        f"{training_label_count}"
    )

    print(
        f"Validation images: "
        f"{validation_image_info['total']}"
    )

    print(
        f"Validation labels: "
        f"{validation_label_count}"
    )

    return {
        "training": {
            "images": training_image_info,
            "labels": training_label_count
        },
        "validation": {
            "images": validation_image_info,
            "labels": validation_label_count
        }
    }


def main():

    print(
        "========================================"
    )

    print(
        "      WASTE DATASET ANALYSIS"
    )

    print(
        "========================================"
    )

    results = {
        "realwaste": analyze_realwaste(),
        "taco": analyze_taco(),
        "rcd": analyze_rcd()
    }

    output_path = (
        PROJECT_ROOT
        / "results"
        / "dataset_analysis.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        "\n========================================"
    )

    print(
        "Analysis completed."
    )

    print(
        f"Report saved to: "
        f"{output_path}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()