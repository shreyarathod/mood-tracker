import streamlit as st
import numpy as np
from deepface import DeepFace
import time
import io
import json
import requests
import base64
import cv2

# Function to analyze emotion in an image
def analyze_emotion(image):
    result = DeepFace.analyze(image, actions=['emotion'])
    return result[0]['emotion']

# Function to normalize emotion scores
def normalize_scores(scores):
    total = sum(scores.values())
    normalized_scores = {emotion: (score / total) * 100 for emotion, score in scores.items()}
    return normalized_scores

# Function to save data to a file
def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Function to send data to an API
def send_data_to_api(data, url):
    response = requests.post(url, json=data)
    return response

def main():
    st.title("Mood Tracker App")
    st.write("Click the button to start capturing an image from your webcam.")

    # Button to start capturing an image
    if st.button("Capture Image"):
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Capture one frame from the webcam
        ret, frame = cap.read()

        if ret:
            # Analyze emotion
            result = analyze_emotion(frame)
            normalized_result = normalize_scores(result)

            # Display captured image
            jpeg_image = cv2.imencode('.jpg', frame)[1].tobytes()
            st.image(jpeg_image, caption='Captured Image', use_column_width=True)

            # Display emotion analysis results
            st.subheader("Emotion Analysis")
            for emotion, score in normalized_result.items():
                st.write(f"{emotion.capitalize()}: {score:.2f}%")

            # Save output data to a file
            output_data = {'emotion': normalized_result, 'image': base64.b64encode(jpeg_image).decode('utf-8')}
            save_data(output_data, 'output.json')
            st.success("Output data saved successfully!")
            
            # Print saved output data to console
            print("Saved output emotion data:")
            print(output_data['emotion'])
            
            # # Send data to API
            # api_url = 'http://localhost:8000/api/v1/emotion/post'
            # response = send_data_to_api(output_data, api_url)
            # if response.status_code == 200:
            #     st.success("Data sent successfully to API!")
            # else:
            #     st.error(f"Failed to send data to API. Status code: {response.status_code}")

        # Release the webcam
        cap.release()

if __name__ == "__main__":
    main()
