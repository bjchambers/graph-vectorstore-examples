import os
import requests
import sys

def download_file(url, destination_file):
    # Check if the file already exists
    if not os.path.exists(destination_file):
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)

        # Download the file from the given URL
        print(f"Downloading the file from {url}...")
        response = requests.get(url, stream=True)

        # Save the file to the destination
        with open(destination_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print(f"File downloaded and saved as {destination_file}")
    else:
        print(f"File '{destination_file}' already exists, skipping download.")

