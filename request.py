import requests

url = 'http://localhost:5000/predict'

# Read the sentence from an HTML file
with open('input_sentence.html', 'r') as file:
    sentence = file.read().strip()

# Ensure the sentence is not empty
if not sentence:
    print("Error: Input sentence is empty.")
else:
    try:
        # Make a POST request to the Flask application
        response = requests.post(url, data={'sentence': sentence})

        # Check the response status code
        if response.status_code == 200:
            print("Prediction Result:", response.text)
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        print(f"Error making request: {e}")

