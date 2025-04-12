import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import io

# Tensorflow Model Prediction
def model_prediction(test_image):
    # Load the model
    model = load_model(r'E:\miniProject\dog_modelv4.h5')
    
    # Convert the uploaded image to a format suitable for prediction
    test_images = Image.open(test_image)
    test_images = test_images.resize((256, 256))  # Resize image to match model input
    test_images_array = np.array(test_images)
    test_images_array = test_images_array / 255.0  # Normalize pixel values to [0, 1]
    test_images_array = np.expand_dims(test_images_array, axis=0)  # Add batch dimension
    
    # Get model prediction
    result = model.predict(test_images_array)
    return result

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Breed Recognition"])

# Main Page - Home
if app_mode == "Home":
    st.header("DOG BREED RECOGNITION SYSTEM")
    image_path = r"E:\miniProject\home_page.jpeg"
    st.image(image_path, use_column_width=True)

    st.markdown("""
    Welcome to the Dog Breed Recognition System! 🐾📸
                
    Ever wondered about the breed of a dog you spotted or one you’re caring for? Our advanced Dog Breed Recognition System is here to provide quick and accurate answers. Powered by the latest machine learning technologies, our system makes identifying dog breeds a breeze.

    Our Mission
    Dogs are a significant part of our lives, bringing joy, companionship, and love. Our mission is to bridge technology and pet care by empowering users with the ability to instantly identify dog breeds, understand their unique traits, and appreciate the diversity of our furry friends.

    How It Works
    1. Upload Image
    Simply navigate to the Breed Recognition page and upload a clear image of the dog. Whether it’s your pet, a shelter dog, or a stray, we’re here to help.

    2. Advanced Analysis
    Using state-of-the-art Convolutional Neural Networks (CNNs) trained on extensive datasets, our system analyzes the image to detect distinct breed features, ensuring accurate results.

    3. Insightful Results
    Within seconds, you’ll receive:

    Identified Breed Name.
    Key Traits: Size, temperament, and exercise needs.
    Additional Details: Fun facts and care tips for the breed.
    Key Features
    Unmatched Accuracy
    Our system utilizes cutting-edge technology for unmatched precision, recognizing hundreds of breeds and their mixes.

    Comprehensive Database
    A wide range of popular and rare dog breeds are included to ensure inclusivity and depth.

    Educational Value
    Learn about the behavior, needs, and care tips of each breed, helping you or others care for them better.

    Why Choose Us?
    Fast Results
    Experience lightning-speed processing with near-instantaneous results.

    User-Friendly Experience
    Designed with simplicity and accessibility in mind for all age groups.

    Perfect for Dog Enthusiasts and Professionals
    Whether you're an enthusiast, pet owner, vet, or animal shelter worker, this tool adds immense value to your work.

    Get Started
    Don’t wait to explore the canine world. Visit the Breed Recognition page in the sidebar, upload an image, and uncover the story of the dog you’re curious about.

    Future Enhancements
    We’re continually evolving! Upcoming updates include:

    Multiple Dog Detection: Recognize multiple dogs in a single photo.
    Health Recommendations: Tailored health suggestions for each breed.
    Mobile App Integration: Bringing the system to your fingertips.
    About Us
    We are a passionate team of technology and animal lovers, aiming to bring innovation to pet care. From identifying breeds to educating about their unique needs, we believe in a world where humans and animals coexist harmoniously.

    Learn more about our journey, inspiration, and future goals on the About page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 4k RGB images of defferent breed, categorized into 7 different classes. The total dataset is divided into an 69:16:15 ratio of training, validation and testing sets, preserving the directory structure.
                #### Content
                1. train (3027 images)
                2. test (685 images)
                3. validation (755 images)
                """)

# Prediction Page
elif app_mode == "Breed Recognition":
    st.header("Breed Recognition")
    test_image = st.file_uploader("Choose an Image:")

    if test_image:
        st.image(test_image, use_column_width=True)

    if st.button("Predict"):
        if test_image:
            st.snow()
            st.write("Our Prediction")
            result = model_prediction(test_image)

            # List of breeds
            Breed_Name = ['Bull Dog', 'Doberman', 'German Shepherd', 'Golden Retriever', 'Husky', 'Pariah Dog', 'Rottweilers']

            # Get the highest probability and the corresponding breed
            predicted_Breed = np.argmax(result[0])
            predicted_label = Breed_Name[predicted_Breed]
            max_probability = result[0][predicted_Breed]

            # If the max probability is less than 85%, classify as "Unknown"
            if max_probability < 0.85:
                predicted_label = "Unknown"

            # Display the predicted breed label
            st.success(f"Predicted Breed: {predicted_label} (Probability: {max_probability:.4f})")

            # Create two columns side by side: one for breed names and the other for probabilities
            col1, col2 = st.columns(2)

            # Display breed names in the first column and probabilities in the second column
            with col1:
                st.subheader("Breed Names")
                for breed in Breed_Name:
                    st.write(breed)

            with col2:
                st.subheader("Probabilities")
                for probability in result[0]:
                    st.write(f"{probability:.6f}")

            # Optionally, you can also print the probabilities in the console
            print(f"Class Probabilities: {result[0]}")
            print(f"Predicted Class Label: {predicted_label}")
            for i, Probability in enumerate(result[0]):
                print(f"Class {Breed_Name[i]}: {Probability:.6f}")

        else:
            st.warning("Please upload an image for prediction.")
