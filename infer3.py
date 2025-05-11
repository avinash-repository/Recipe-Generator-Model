import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import os
import re
import pickle
import google.generativeai as genai
import pandas as pd
import random
import tkinter as tk
from tkinter import filedialog

# Figure out how many calories someone needs daily
def calculate_tdee(weight, height, age, gender, activity_level):
    """
  
    activity_level: How active you are
        'sedentary': Almost no exercise
        'light': Exercise 1-3 days/week
        'moderate': Exercise 3-5 days/week
        'active': Hard exercise 6-7 days/week
        'very_active': Very intense exercise or physical job
    """
    # How much to multiply by based on activity
    activity_multipliers = {
        'sedentary': 1.2,      # Little exercise
        'light': 1.375,        # Light exercise
        'moderate': 1.55,      # Moderate exercise
        'active': 1.725,       # Hard exercise
        'very_active': 1.9     # Very hard exercise
    }
    
    # Calculate base metabolism (BMR)
    if gender.lower() == 'male':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:  # female
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    
    # Multiply BMR by activity level to get total daily calories
    multiplier = activity_multipliers.get(activity_level.lower(), 1.2)  
    tdee = bmr * multiplier
    
    return round(tdee)
def test_tkinter():
    root = tk.Tk()
    root.title("Test Window")
    label = tk.Label(root, text="This is a test window")
    label.pack()
    root.mainloop()

# Call this function to test
test_tkinter()
# Ask user for their info
def get_user_info():
    """Ask for height, weight, etc. to calculate calories needed"""
    print("\nTo give you good diet advice, we need some info about you.")
    
    try:
        weight = float(input("Your weight (kg): "))
        height = float(input("Your height (cm): "))
        age = int(input("Your age (years): "))
        
        # Ask for gender
        while True:
            gender = input("Your gender (male/female): ").lower()
            if gender in ['male', 'female']:
                break
            print("Please type 'male' or 'female'.")
        
        # Ask about activity level
        print("\nHow active are you?")
        print("1. Sedentary (little/no exercise)")
        print("2. Light (exercise 1-3 days/week)")
        print("3. Moderate (exercise 3-5 days/week)")
        print("4. Active (hard exercise 6-7 days/week)")
        print("5. Very Active (very intense exercise/physical job)")
        
        activity_map = {
            '1': 'sedentary',
            '2': 'light', 
            '3': 'moderate',
            '4': 'active',
            '5': 'very_active'
        }
        
        while True:
            activity_choice = input("Choose your activity level (1-5): ")
            if activity_choice in activity_map:
                activity_level = activity_map[activity_choice]
                break
            print("Please enter a number between 1 and 5.")
        
        # Calculate daily calories
        tdee = calculate_tdee(weight, height, age, gender, activity_level)
        
        # Ask about fitness goal
        print("\nWhat's your fitness goal?")
        print("1. Lose Weight")
        print("2. Maintain Weight")
        print("3. Gain Muscle/Bulk Up")
        
        goal_map = {
            '1': 'weight_loss',
            '2': 'maintenance',
            '3': 'bulking'
        }
        
        goal = 'maintenance'  # Default
        while True:
            goal_choice = input("Choose your goal (1-3): ")
            if goal_choice in goal_map:
                goal = goal_map[goal_choice]
                break
            print("Please enter a number between 1 and 3.")
        
        # Set target calories based on goal
        target_calories = tdee
        if goal == 'weight_loss':
            target_calories = int(tdee * 0.85)  # 15% less
        elif goal == 'bulking':
            target_calories = int(tdee * 1.15)  # 15% more
        
        return {
            'tdee': tdee,
            'goal': goal,
            'target_calories': target_calories,
            'weight': weight,
            'height': height,
            'age': age,
            'gender': gender,
            'activity_level': activity_level
        }
        
    except ValueError:
        print("Oops! Something went wrong. Using default values.")
        return {
            'tdee': 2000,
            'goal': 'maintenance',
            'target_calories': 2000,
            'weight': 70,
            'height': 170,
            'age': 30,
            'gender': 'male',
            'activity_level': 'moderate'
        }

# Load the calories info from file
def load_calories_db(csv_path='ingredients_calories.csv'):
    """Get calorie info from our CSV file"""
    try:
        calories_df = pd.read_csv(csv_path)
        # Make a simple lookup dictionary
        calories_dict = dict(zip(calories_df['ingredient'].str.lower(), calories_df['calories_per_unit']))
        return calories_df, calories_dict
    except Exception as e:
        print(f"Can't load calories database: {e}")
        return None, {}

# Ask user for ingredient amounts
def get_ingredient_quantities(ingredients_list):
    """Ask how much of each ingredient they have"""
    quantities = {}
    for ingredient in ingredients_list:
        while True:
            try:
                # Ask for units not grams
                qty = float(input(f"How much {ingredient} do you have (units): "))
                if qty >= 0:
                    quantities[ingredient] = qty
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
    return quantities

# Add up all calories
def calculate_total_calories(ingredients_quantities, calories_dict):
    """Figure out total calories for the recipe"""
    total_calories = 0
    calories_breakdown = {}
    
    for ingredient, quantity in ingredients_quantities.items():
        if ingredient.lower() in calories_dict:
            # Multiply quantity by calories per unit
            ingredient_calories = calories_dict[ingredient.lower()] * quantity
            total_calories += ingredient_calories
            calories_breakdown[ingredient] = ingredient_calories
        else:
            print(f"Hmm, we don't know calories for {ingredient}.")
            calories_breakdown[ingredient] = "Unknown"
    
    return total_calories, calories_breakdown

# Check if recipe matches user's diet goals
def determine_diet_type(total_calories, user_profile):
    """See if recipe fits with weight loss, maintenance, or bulking"""
    tdee = user_profile['tdee']
    goal = user_profile['goal']
    target_calories = user_profile['target_calories']
    
    # What percent of daily calories is this recipe
    meal_percentage = (total_calories / tdee) * 100
    daily_percentage = (total_calories / target_calories) * 100
    
    # Match recipe to user's goal
    if goal == 'weight_loss':
        if total_calories < (target_calories / 3):  # Less than 1/3 of daily target
            return "weight_loss", f"This recipe is {meal_percentage:.1f}% of your daily calories ({total_calories:.0f} of {tdee:.0f}) - good for weight loss!"
        else:
            return "caution", f"This recipe is {meal_percentage:.1f}% of your daily calories ({total_calories:.0f} of {tdee:.0f}). Try smaller portions for weight loss."
    
    elif goal == 'bulking':
        if total_calories > (target_calories / 3):  # More than 1/3 of daily target
            return "bulking", f"This recipe is {meal_percentage:.1f}% of your daily calories ({total_calories:.0f} of {tdee:.0f}) - good for muscle gain!"
        else:
            return "caution", f"This recipe is {meal_percentage:.1f}% of your daily calories ({total_calories:.0f} of {tdee:.0f}). Try adding more for bulking."
    
    else:  # maintenance
        if total_calories > (target_calories / 2):
            return "caution", f"This recipe is {meal_percentage:.1f}% of your daily calories ({total_calories:.0f} of {tdee:.0f}). Consider smaller portions to maintain weight."
        elif total_calories < (target_calories / 5):
            return "caution", f"This recipe is only {meal_percentage:.1f}% of your daily calories ({total_calories:.0f} of {tdee:.0f}). This is more of a snack than a meal."
        else:
            return "maintenance", f"This recipe is {meal_percentage:.1f}% of your daily calories ({total_calories:.0f} of {tdee:.0f}) - perfect for maintaining weight!"

# Load our AI model for ingredient recognition
model = load_model('ingredient_classifier.h5')

# Load the mapping of numbers to ingredient names
class_indices = np.load('class_indices.npy', allow_pickle=True).item()
# Flip the dictionary to go from index to name
idx_to_class = {v: k for k, v in class_indices.items()}

# Analyze an image to identify the ingredient
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Make pixel values between 0-1
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = idx_to_class[predicted_class_idx]
    
    return predicted_class, confidence


# Set up the recipe generator AI

# Connect to Gemini AI for recipe improvements
genai.configure(api_key="")
gemini_model = genai.GenerativeModel("")

# Load our recipe generator AI model
rnn_model = tf.keras.models.load_model("./trained_model/recipe_generator_model.h5")
with open("./trained_model/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Set maximum text length
max_sequence_len = 50

# Basic ingredients we can always use
ALLOWED_BASICS = ["salt", "pepper", "spices", "oil", "vinegar", "sauce", "herbs", "garlic", "onion"]

# Clean up ingredient list
def process_ingredients(user_input):
    """Split and clean up ingredients from user text"""
    ingredients_list = [ing.strip().lower() for ing in user_input.split(",")]
    return ingredients_list[:5]  # Use at most 5 ingredients

# Use creative temperature to make AI more random
def sample_with_temperature(preds, temperature=1.0):
    """Add randomness to AI choices"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate recipe text
def generate_text(seed_text, max_gen_words=50, stop_token="<end>", temperature=1.0):
    """Make the AI write a recipe"""
    output_text = seed_text
    for _ in range(max_gen_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted_probs = rnn_model.predict(token_list, verbose=0)[0]
        predicted_index = sample_with_temperature(predicted_probs, temperature)
        # Find word for the predicted number
        output_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        if output_word is None:
            break
        if output_word == stop_token:
            break
        output_text += " " + output_word
    return output_text

# Improve recipe with Gemini AI
def refine_with_gemini(dish_name, procedure, input_ingredients, quantities, total_calories, diet_type, diet_message):
    """Use Gemini AI to make the recipe better with proper quantities"""
    ingredients_with_qty = []
    for ing in input_ingredients:
        qty_str = f"{quantities.get(ing, 1)} units" if ing in quantities else "to taste"
        ingredients_with_qty.append(f"{ing} ({qty_str})")
    
    ingredients_str = ", ".join(ingredients_with_qty)
    allowed_basics_str = ", ".join(ALLOWED_BASICS)
    
    prompt = f"""
Refine this recipe:

Dish Name: {dish_name}
Procedure: {procedure}
Main Ingredients with Quantities: {ingredients_str}
Allowed Basic Ingredients: {allowed_basics_str}
Total Calories: {total_calories:.2f} calories
Diet Type: {diet_type.title()}

Please make this recipe better:
- Make the dish name short and clear
- ONLY use the main ingredients listed and basic ingredients allowed
- NO other ingredients (especially no meat or seafood unless already listed)
- Make it work for a {diet_type} diet ({diet_message})
- If "caution" diet type, suggest how to make it better for the user's goals
- Fix any repetitive or extra words
- Write clear, step-by-step instructions
- Include ingredient quantities in the steps

Format like this:
Refined Dish Name:
<your dish name>
Refined Procedure:
<your step-by-step instructions>
Calorie Information:
Total Calories: {total_calories:.2f}
Diet Type: {diet_type.title()} - {diet_message}
"""
    try:
        response = gemini_model.generate_content(prompt)
        refined = response.text.strip()
        
        # Break down Gemini's response
        refined_sections = {}
        current_section = None
        
        for line in refined.split('\n'):
            if "Refined Dish Name:" in line:
                current_section = "dish_name"
                refined_sections[current_section] = ""
            elif "Refined Procedure:" in line:
                current_section = "procedure"
                refined_sections[current_section] = ""
            elif "Calorie Information:" in line:
                current_section = "calories"
                refined_sections[current_section] = ""
            elif current_section:
                if refined_sections[current_section]:
                    refined_sections[current_section] += "\n"
                refined_sections[current_section] += line
        
        return refined_sections
    except Exception as e:
        print(f"Gemini AI error: {e}")
        # Return basic info if Gemini fails
        return {
            "dish_name": dish_name,
            "procedure": procedure,
            "calories": f"Total Calories: {total_calories:.2f}\nDiet Type: {diet_type.title()} - {diet_message}"
        }

# Make sure recipe doesn't use wrong ingredients
def check_procedure_for_unauthorized_ingredients(procedure, allowed_ingredients):
    """Remove any ingredients we shouldn't be using"""
    # Make everything lowercase
    allowed_ingredients_lower = [ing.lower() for ing in allowed_ingredients]
    
    # Foods we shouldn't include unless specifically allowed
    unauthorized_proteins = ["meat", "chicken", "beef", "pork", "lamb", "fish", "seafood", 
                            "shrimp", "prawn", "turkey", "duck", "salmon", 
                            "tuna", "ham", "bacon", "sausage"]
    
    # Remove proteins that are actually in our allowed list
    unauthorized_proteins = [protein for protein in unauthorized_proteins 
                            if not any(protein in ing for ing in allowed_ingredients_lower)]
    
    # Check if recipe mentions any unauthorized proteins
    found_unauthorized = []
    for protein in unauthorized_proteins:
        if re.search(r'\b' + protein + r'\b', procedure.lower()):
            found_unauthorized.append(protein)
    
    if found_unauthorized:
        # print(f"Warning: Found ingredients we shouldn't use: {', '.join(found_unauthorized)}")
        
        # Create pattern to find these words
        pattern = r'\b(' + '|'.join(found_unauthorized) + r')\b'
        
        # Replace bad ingredients
        procedure = re.sub(pattern, "[ingredient removed]", procedure, flags=re.IGNORECASE)
        
    return procedure

# Put everything together to make a recipe
def generate_recipe_with_nutrition(ingredients_input, quantities, calories_dict, user_profile, temperature=1.0):
    """Create complete recipe with nutrition info"""
    # Process ingredients
    processed_ing_list = process_ingredients(ingredients_input)
    processed_ing = ", ".join(processed_ing_list)
    
    # All ingredients we can use
    all_allowed_ingredients = processed_ing_list + ALLOWED_BASICS
    
    # Vary quantities slightly to get different recipes
    modified_quantities = {}
    for ing, qty in quantities.items():
        # Random variation by ±20%
        variation_factor = random.uniform(0.8, 1.2)  
        modified_quantities[ing] = qty * variation_factor
    
    # Calculate calories
    total_calories, calories_breakdown = calculate_total_calories(modified_quantities, calories_dict)
    diet_type, diet_message = determine_diet_type(total_calories, user_profile)
    
    # Generate dish name
    dish_seed = "<start> Dish:"
    dish_generated = generate_text(dish_seed, max_gen_words=6, stop_token="|", temperature=temperature)
    dish_part = dish_generated.replace("<start>", "").replace("Dish:", "").strip()
    dish_words = dish_part.split()
    dish_name = " ".join(dish_words[:4])
    
    # Generate recipe steps
    seed_for_proc = (f"<start> Dish: {dish_name} | Ingredients: {processed_ing} | "
                     f"Note: Only use the specified ingredients and basic items ({', '.join(ALLOWED_BASICS)}) for the recipe. | Procedure:")
    proc_generated = generate_text(seed_for_proc, max_gen_words=100, stop_token="<end>", temperature=temperature)
    if "| Procedure:" in proc_generated:
        procedure = proc_generated.split("| Procedure:")[-1].strip()
    else:
        procedure = proc_generated.strip()
    
    # Clean up text
    procedure = re.sub(r'\bend\b', '', procedure, flags=re.IGNORECASE)
    procedure = re.sub(r'\s+', ' ', procedure).strip()
    
    # Check for unauthorized ingredients
    procedure = check_procedure_for_unauthorized_ingredients(procedure, all_allowed_ingredients)
    
    # Clean up dish name
    dish_name = re.sub(r'\d+', '', dish_name)
    dish_name = re.sub(r'\bingredients?\b', '', dish_name, flags=re.IGNORECASE).strip()
    
    # Improve recipe with Gemini AI
    refined_recipe = refine_with_gemini(
        dish_name, procedure, processed_ing_list, 
        modified_quantities, total_calories, diet_type, diet_message
    )
    
    # Final check for unauthorized ingredients
    if 'procedure' in refined_recipe:
        refined_recipe['procedure'] = check_procedure_for_unauthorized_ingredients(
            refined_recipe['procedure'], all_allowed_ingredients
        )
    
    return refined_recipe, calories_breakdown, modified_quantities

# Function to select images using file dialog
def select_images(max_images=5):
    """Open directory dialog for user to select a folder containing images"""
    # Create tkinter root window (but hide it)
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    print(f"\nPlease select a directory containing up to {max_images} food ingredient images...")
    
    # Open directory dialog
    dir_path = filedialog.askdirectory(
        title="Select Directory with Ingredient Images"
    )
    
    if not dir_path:
        print("No directory selected. Exiting program.")
        exit()
    
    # Collect image files from the directory
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
    selected_images = []
    for filename in os.listdir(dir_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            img_path = os.path.join(dir_path, filename)
            selected_images.append(img_path)
    
    # Limit to max_images and check if any images found
    selected_images = selected_images[:max_images]
    if not selected_images:
        print("No images found in the directory. Exiting program.")
        exit()
    
    print(f"Found {len(selected_images)} images in the directory.")
    return selected_images


# Function to get manually entered ingredients
def get_manual_ingredients():
    """Ask user to manually enter ingredients"""
    print("\nPlease enter your ingredients separated by commas (e.g., onion, capsicum, cauliflower, eggs):")
    ingredients_input = input("> ")
    
    # Process the input
    ingredients_list = process_ingredients(ingredients_input)
    
    if not ingredients_list:
        print("No valid ingredients entered. Exiting program.")
        exit()
    
    print(f"Detected ingredients: {', '.join(ingredients_list)}")
    return ingredients_list

# Main function to run everything - now with input method choice
# Fix for the main function in the second code
def ingredient_classification_and_recipe_generation_with_nutrition(input_method=None, image_paths=None, manual_ingredients=None, calories_csv='ingredients_calories.csv', num_samples=3, temperature=1.0):
    """Main function: identify ingredients from pictures or text and create recipes"""
    detected_ingredients = []
    
    # Step 1: Load calorie info
    _, calories_dict = load_calories_db(calories_csv)
    
    # Step 2: Get ingredients based on input method
    if input_method is None:
        # Ask user for input method
        print("\nHow would you like to input ingredients?")
        print("1. Upload images of ingredients")
        print("2. Type ingredients manually")
        
        while True:
            choice = input("Enter your choice (1 or 2): ")
            if choice in ['1', '2']:
                input_method = 'images' if choice == '1' else 'manual'
                break
            print("Please enter 1 or 2.")
    
    # Process ingredients based on input method
    if input_method == 'images':
        # If no images provided, ask user to select them
        if not image_paths:
            # This is the critical line that needs to be executed
            image_paths = select_images(max_images=5)
        
        # Identify ingredients from pictures
        for img_path in image_paths:
            try:
                predicted_ingredient, confidence = predict_image(img_path)
                print(f"Image {os.path.basename(img_path)}: Found {predicted_ingredient} (Confidence: {confidence:.2f})")
                detected_ingredients.append(predicted_ingredient)
            except Exception as e:
                print(f"Problem with image {os.path.basename(img_path)}: {e}")
    else:  # manual input
        # If no manual ingredients provided, ask user to enter them
        if not manual_ingredients:
            detected_ingredients = get_manual_ingredients()
        else:
            detected_ingredients = manual_ingredients
    
    # Step 3: Ask for ingredient quantities
    print("\nHow much of each ingredient do you have?")
    quantities = get_ingredient_quantities(detected_ingredients)
    
    # Step 4: Get user info for personalized advice
    user_profile = get_user_info()
    print(f"\nYou need about {user_profile['tdee']} calories per day")
    print(f"For your {user_profile['goal']} goal, aim for {user_profile['target_calories']} calories per day")
    
    # Step 5: Create recipes
    ingredients_input = ", ".join(detected_ingredients)
    print(f"\nCreating {num_samples} recipes using: {ingredients_input}")
    print(f"Quantities: {quantities}")
    
    results = []
    for i in range(num_samples):
        try:
            # Use different randomness for each recipe
            recipe_temperature = temperature * random.uniform(0.8, 1.2)
            
            refined_recipe, calories_breakdown, modified_quantities = generate_recipe_with_nutrition(
                ingredients_input, quantities, calories_dict, user_profile, temperature=recipe_temperature
            )
            
            print(f"\n{'='*50}")
            print(f"Recipe {i+1} – {refined_recipe.get('dish_name', 'Unknown Recipe')}")
            print(f"{'='*50}")
            
            print(f"\nInstructions:")
            print(refined_recipe.get('procedure', 'No instructions available'))
            print(f"\nNutrition Info:")
            print(refined_recipe.get('calories', 'No calorie information available'))
            print(f"This recipe works with your {user_profile['goal']} goal.")
            
            results.append((refined_recipe, calories_breakdown, modified_quantities))
        except Exception as e:
            print(f"Error making recipe: {e}")
    
    return detected_ingredients, quantities, results


# Run the program
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Food Identifier and Recipe Creator")
    print("="*50)
    
    # Run the program with input method selection
    detected_ingredients, quantities, generated_recipes = ingredient_classification_and_recipe_generation_with_nutrition(
        input_method=None,  # Will ask user for input method
        calories_csv='./datasets/ingredients_calories.csv', 
        num_samples=3
    )