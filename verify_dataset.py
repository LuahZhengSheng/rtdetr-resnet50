from ultralytics.data.utils import visualize_image_annotations

# Define the label map with all annotated class labels.
# This should match the classes in your dataset.yaml file
label_map = {
    0: "HDPE Plastic",
    1: "Multi-layer Plastic",
    2: "PET Bottle",
    # Add all your classes here
}

# Visualize
visualize_image_annotations(
    r"C:\Users\60174\OneDrive\FYP\dataset\selected_category_dataset\images\val\7_Types_Plastic_valid_2175_20240128_114606_jpg.rf.a2b4b582d76aff9093be214466f516d0.jpg",  # Input image path.
    r"C:\Users\60174\OneDrive\FYP\dataset\selected_category_dataset\labels\val\7_Types_Plastic_valid_2175_20240128_114606_jpg.rf.a2b4b582d76aff9093be214466f516d0.txt",  # Annotation file path for the image (YOLO format).
    label_map,
)
