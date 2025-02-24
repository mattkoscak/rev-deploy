# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# Expose port 8501 (the default for Streamlit)
EXPOSE 8501

# Set environment variables so Streamlit runs in headless mode on Cloud Run
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Command to run your Streamlit app (update the filename if necessary)
CMD ["streamlit", "run", "Rev_current.py"]

