from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REALWASTE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "realwaste"
    / "realwaste-main"
    / "RealWaste"
)

TACO_METADATA = (
    PROJECT_ROOT
    / "processed"
    / "taco_crops"
    / "metadata.csv"
)

RCD_METADATA = (
    PROJECT_ROOT
    / "processed"
    / "rcd_crops"
    / "metadata.csv"
)

OUTPUT_PATH = PROJECT_ROOT / "processed" / "dataset"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

CLASSES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]


# --------------------------------------------------
# Clean previous dataset
# --------------------------------------------------

if OUTPUT_PATH.exists():
    shutil.rmtree(OUTPUT_PATH)


for split in ["train", "val", "test"]:

    for class_name in CLASSES:

        (
            OUTPUT_PATH
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# --------------------------------------------------
# RealWaste
# --------------------------------------------------

realwaste_rows = []

for class_name in CLASSES:

    class_path = REALWASTE_PATH / class_name

    if not class_path.exists():
        continue

    for image_path in class_path.iterdir():

        if (
            image_path.is_file()
            and image_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            realwaste_rows.append({
                "source": "RealWaste",
                "source_image": str(image_path),
                "group_id": f"RealWaste_{image_path.name}",
                "class": class_name
            })


realwaste_df = pd.DataFrame(
    realwaste_rows
)


print("\nREALWASTE")
print("=" * 50)
print(
    f"Images found: {len(realwaste_df)}"
)


# --------------------------------------------------
# TACO
# --------------------------------------------------

taco_df = pd.read_csv(
    TACO_METADATA
)

taco_df["source_image"] = taco_df[
    "crop_path"
].apply(
    lambda x: str(PROJECT_ROOT / x)
)

taco_df["class"] = taco_df[
    "common_class"
]

taco_df["group_id"] = (
    "TACO_"
    + taco_df["source_image_id"].astype(str)
)

taco_df = taco_df[
    [
        "source",
        "source_image",
        "group_id",
        "class"
    ]
]


print("\nTACO")
print("=" * 50)
print(
    f"Crops found: {len(taco_df)}"
)


# --------------------------------------------------
# RCD
# --------------------------------------------------

rcd_df = pd.read_csv(
    RCD_METADATA
)

rcd_df["source_image"] = rcd_df[
    "crop_path"
].apply(
    lambda x: str(PROJECT_ROOT / x)
)

rcd_df["class"] = rcd_df[
    "common_class"
]

rcd_df["group_id"] = (
    "RCD_"
    + rcd_df["source_image"].apply(
        lambda x: Path(x).stem
    )
)

rcd_df = rcd_df[
    [
        "source",
        "source_image",
        "group_id",
        "class"
    ]
]


print("\nRCD")
print("=" * 50)
print(
    f"Crops found: {len(rcd_df)}"
)


# --------------------------------------------------
# Combine
# --------------------------------------------------

all_data = pd.concat(
    [
        realwaste_df,
        taco_df,
        rcd_df
    ],
    ignore_index=True
)


all_data = all_data[
    all_data["class"].isin(CLASSES)
].reset_index(drop=True)


print("\nTOTAL DATA")
print("=" * 50)
print(
    f"Total samples: {len(all_data)}"
)


# --------------------------------------------------
# Split by source group
# --------------------------------------------------

groups = (
    all_data[
        [
            "group_id",
            "class"
        ]
    ]
    .drop_duplicates("group_id")
)


train_groups = []
val_groups = []
test_groups = []


for class_name in CLASSES:

    class_groups = groups[
        groups["class"] == class_name
    ]["group_id"].tolist()


    if len(class_groups) < 3:

        train_groups.extend(
            class_groups
        )

        continue


    train_g, temp_g = train_test_split(
        class_groups,
        test_size=(
            VAL_SIZE + TEST_SIZE
        ),
        random_state=RANDOM_STATE
    )


    val_g, test_g = train_test_split(
        temp_g,
        test_size=(
            TEST_SIZE
            / (VAL_SIZE + TEST_SIZE)
        ),
        random_state=RANDOM_STATE
    )


    train_groups.extend(train_g)
    val_groups.extend(val_g)
    test_groups.extend(test_g)


train_groups = set(train_groups)
val_groups = set(val_groups)
test_groups = set(test_groups)


# --------------------------------------------------
# Assign split
# --------------------------------------------------

def get_split(group_id):

    if group_id in train_groups:
        return "train"

    if group_id in val_groups:
        return "val"

    if group_id in test_groups:
        return "test"

    return None


all_data["split"] = all_data[
    "group_id"
].apply(get_split)


# --------------------------------------------------
# Copy images
# --------------------------------------------------

counters = {
    "train": {},
    "val": {},
    "test": {}
}


copied = 0
missing = 0


for index, row in all_data.iterrows():

    split = row["split"]

    class_name = row["class"]

    source_path = Path(
        row["source_image"]
    )


    if not source_path.exists():

        missing += 1
        continue


    class_counter = counters[
        split
    ].get(
        class_name,
        0
    )


    destination_name = (
        f"{row['source']}_"
        f"{index}_"
        f"{class_counter}"
        f"{source_path.suffix.lower()}"
    )


    destination = (
        OUTPUT_PATH
        / split
        / class_name
        / destination_name
    )


    shutil.copy2(
        source_path,
        destination
    )


    counters[
        split
    ][class_name] = class_counter + 1

    copied += 1


# --------------------------------------------------
# Final report
# --------------------------------------------------

print("\nDATASET CREATION COMPLETE")
print("=" * 50)

print(
    f"Total samples : {len(all_data)}"
)

print(
    f"Copied images : {copied}"
)

print(
    f"Missing images: {missing}"
)


print("\nFINAL DISTRIBUTION")
print("=" * 50)


for split in [
    "train",
    "val",
    "test"
]:

    print(
        f"\n{split.upper()}"
    )

    split_data = all_data[
        all_data["split"] == split
    ]

    counts = (
        split_data["class"]
        .value_counts()
        .reindex(
            CLASSES,
            fill_value=0
        )
    )

    for class_name, count in counts.items():

        print(
            f"{class_name:25} : {count}"
        )


# --------------------------------------------------
# Save complete metadata
# --------------------------------------------------

metadata_path = (
    OUTPUT_PATH
    / "dataset_metadata.csv"
)

all_data.to_csv(
    metadata_path,
    index=False
)

print(
    f"\nMetadata saved to:"
)

print(metadata_path)