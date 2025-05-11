import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Load the trained model we saved earlier.
model = load_model('ingredient_classifier.h5')

# Get the class indices, so we know what our model is predicting.
class_indices = np.load('class_indices.npy', allow_pickle=True).item()
# Flip the dictionary around, so we can go from index to class name.
idx_to_class = {v: k for k, v in class_indices.items()}

# Function to take an image, get it ready for the model, and make a prediction.
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Make pixel values between 0 and 1
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = idx_to_class[predicted_class_idx]
    
    return predicted_class, confidence

# Some test images and their actual labels.
test_data = [
    ("./images/apple/apple_1.jpeg", "apple"),
    ("./images/avocado/Avocado_1.jpeg", "Avocado"),
    ("./images/banana/img1.jpeg", "banana"),
    ("./images/Bread/Bread_1.jpeg", "Bread"),
    ("./images/brinjal/img1.jpeg", "brinjal"),
    ("./images/brocoli/img1.jpeg", "brocoli"),
    ("./images/carrot/img1.jpeg", "carrot"),
    ("./images/Corn/Corn_1.jpeg", "Corn"),
    ("./images/Eggs/Eggs_1.jpeg", "Eggs")
    # Add more test images and their true labels here
]

# Let's see how well our model does on these test images.
correct_predictions = 0
results = []

plt.figure(figsize=(12, 8))
for i, (img_path, true_label) in enumerate(test_data):
    if not os.path.exists(img_path):
        print(f"Warning: Image not found at {img_path}")
        continue
        
    predicted_label, confidence = predict_image(img_path)
    is_correct = predicted_label == true_label
    
    if is_correct:
        correct_predictions += 1
    
    results.append({
        'image': img_path,
        'true_label': true_label,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'correct': is_correct
    })
    
    # Show the images with their predictions.
    if i < 9:  # Show up to 9 images
        plt.subplot(3, 3, i+1)
        img = plt.imread(img_path)
        plt.imshow(img)
        color = 'green' if is_correct else 'red'
        plt.title(f"True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.2f}", 
                  color=color, fontsize=10)
        plt.axis('off')

plt.tight_layout()
plt.savefig('test_results.png')
plt.show()

# Calculate and print the overall accuracy.
accuracy = correct_predictions / len(test_data)
print(f"Test Accuracy: {accuracy:.2f} ({correct_predictions}/{len(test_data)})")

# Print the results for each image.
print("\nDetailed Results:")
for result in results:
    status = "✓" if result['correct'] else "✗"
    print(f"{status} {result['image']} - True: {result['true_label']}, Predicted: {result['predicted_label']} (Confidence: {result['confidence']:.2f})")

# Let's get a confusion matrix and a classification report for more details.
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Get the true and predicted labels for all test images.
y_true = [item[1] for item in test_data]
y_pred = [predict_image(item[0])[0] for item in test_data]

# Get a list of all the unique labels.
unique_labels = list(set(y_true + y_pred))

# Make the confusion matrix.
cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

# Plot the confusion matrix.
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Print the classification report.
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=unique_labels))