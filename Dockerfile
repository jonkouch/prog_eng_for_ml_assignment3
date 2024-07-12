# Use the official Python 3.10 image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements.txt file from the host to the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script and inference script to the container
COPY train_model.py .
COPY inference.py .

# Run the training script to train the model and save it
RUN python train_model.py

# Define the command to run when the container starts
CMD ["python", "inference.py"]