import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import glob

# Load the trained model we saved earlier
model = load_model('ingredient_classifier.h5')

# Get the class indices, so we know what our model is predicting
class_indices = np.load('class_indices.npy', allow_pickle=True).item()
# Flip the dictionary around, so we can go from index to class name
idx_to_class = {v: k for k, v in class_indices.items()}

# Function to take an image, get it ready for the model, and make a prediction
def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Make pixel values between 0 and 1
        
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = idx_to_class[predicted_class_idx]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, 0

# Function to find all ingredient folders and their image files
def find_all_ingredients(base_dir="./images"):
    test_data = []
    
    # Check if base_dir exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} not found.")
        return test_data
    
    # List all subdirectories (ingredient folders)
    ingredient_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not ingredient_dirs:
        print(f"No ingredient directories found in {base_dir}")
        return test_data
    
    print(f"Found {len(ingredient_dirs)} ingredient types: {', '.join(ingredient_dirs)}")
    
    # For each ingredient directory, find all image files
    for ingredient in ingredient_dirs:
        ingredient_path = os.path.join(base_dir, ingredient)
        
        # Common image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(ingredient_path, ext)))
            # Also check for images in subdirectories (one level down)
            image_files.extend(glob.glob(os.path.join(ingredient_path, '*', ext)))
        
        if image_files:
            print(f"Found {len(image_files)} images for {ingredient}")
            for img_path in image_files:
                test_data.append((img_path, ingredient))
        else:
            print(f"No images found for {ingredient}")
    
    return test_data

# Find all ingredient images
print("Scanning for all available ingredients...")
test_data = find_all_ingredients()

if not test_data:
    print("No test data found. Please check your image directory structure.")
    exit()

print(f"Total images to process: {len(test_data)}")

# Process test images and make predictions
print("Making predictions on all images...")
test_results = []
for i, (img_path, true_label) in enumerate(test_data):
    if i % 10 == 0:  # Progress indicator
        print(f"Processing image {i+1}/{len(test_data)}")
    
    predicted_label, confidence = predict_image(img_path)
    if predicted_label is not None:
        test_results.append({
            'image': img_path,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence
        })

if not test_results:
    print("No valid predictions were made. Please check your images and model.")
    exit()

# Calculate accuracy
correct = sum(1 for result in test_results if result['true_label'] == result['predicted_label'])
accuracy = correct / len(test_results)
print(f"\nOverall Accuracy: {accuracy:.2f} ({correct}/{len(test_results)})")

# Get the true and predicted labels for all test images
y_true = [result['true_label'] for result in test_results]
y_pred = [result['predicted_label'] for result in test_results]

# Get a list of all the unique labels that appeared in our data
unique_labels = sorted(list(set(y_true + y_pred)))
print(f"Found {len(unique_labels)} unique labels in the results")

# Create the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

# Print the confusion matrix as text
print("\n----- CONFUSION MATRIX -----")
print("Confusion Matrix:")

# Determine column width based on longest label
max_label_len = max(len(label) for label in unique_labels)
col_width = max(9, max_label_len + 2)  # At least 9 chars wide

    # Print the header with class labels
header = "True/Pred"  # Using forward slash instead of backslash
print(f"{header:<{col_width}}", end="")
for label in unique_labels:
    print_label = label[:col_width-2] + ".." if len(label) > col_width-2 else label
    print(f"{print_label:<{col_width}}", end="")
print()

# Print each row of the confusion matrix with its label
for i, label in enumerate(unique_labels):
    print_label = label[:col_width-2] + ".." if len(label) > col_width-2 else label
    print(f"{print_label:<{col_width}}", end="")
    for j in range(len(unique_labels)):
        print(f"{cm[i, j]:<{col_width}}", end="")
    print()  # New line after each row

# For large matrices, also save to CSV for easier analysis
if len(unique_labels) > 10:
    print(f"\nSaving large confusion matrix ({len(unique_labels)}x{len(unique_labels)}) to CSV...")
    with open('confusion_matrix.csv', 'w') as f:
        # Write header
        f.write('True/Pred,' + ','.join(unique_labels) + '\n')
        # Write each row
        for i, label in enumerate(unique_labels):
            f.write(label + ',' + ','.join(str(cm[i, j]) for j in range(len(unique_labels))) + '\n')
    print("Saved to confusion_matrix.csv")

# Calculate per-class metrics
print("\nPer-Class Performance:")
print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
print("-" * 60)

class_metrics = {}
for i, label in enumerate(unique_labels):
    # True positives, false positives, false negatives
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    
    # Class metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    support = cm[i, :].sum()
    
    # Store metrics
    class_metrics[label] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }
    
    # Print metrics
    print(f"{label:<20} {precision:.4f}     {recall:.4f}     {f1:.4f}     {support}")

# Plot the confusion matrix as a heatmap
# For large matrices, we might need to adjust the figure size
plt_size = min(20, max(10, len(unique_labels) / 2))
plt.figure(figsize=(plt_size, plt_size))

# For large matrices, we might want to not show annotations
show_annot = len(unique_labels) <= 25
sns.heatmap(cm, annot=show_annot, fmt='d', cmap='Blues', 
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)

# Also create a normalized version for better visibility of patterns
plt.figure(figsize=(plt_size, plt_size))
# Normalize by row (true labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0

sns.heatmap(cm_normalized, annot=show_annot, fmt='.2f', cmap='Blues', 
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Normalized Confusion Matrix (Row-wise)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('normalized_confusion_matrix.png', dpi=300)

print("\nConfusion matrix plots saved as:")
print("- confusion_matrix.png")
print("- normalized_confusion_matrix.png")

# Print top confusions (most confused pairs)
if len(unique_labels) > 5:  # Only for larger matrices
    print("\nTop Confusions (True → Predicted):")
    confusions = []
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            if i != j and cm[i, j] > 0:
                confusions.append((unique_labels[i], unique_labels[j], cm[i, j]))
    
    # Sort by confusion count, descending
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    # Print top 10 or fewer
    for i, (true_label, pred_label, count) in enumerate(confusions[:10]):
        print(f"{i+1}. {true_label} → {pred_label}: {count} times")

plt.show()  # Show all plots at the end