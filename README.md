# California Housing Price Prediction API

A machine learning web API built with FastAPI and Scikit-Learn that predicts California housing prices based on housing and demographic features. The API supports both single-house predictions and batch predictions using CSV uploads.

## Project Overview

This project demonstrates the end-to-end machine learning workflow:

- Data exploration and preprocessing
- Model training using Scikit-Learn
- Model serialization with Joblib
- REST API development using FastAPI
- Input validation using Pydantic
- Batch inference through CSV uploads
- Interactive API documentation with Swagger UI

## Features

- Single property price prediction
- Batch prediction using CSV files
- Automatic request validation
- Interactive Swagger documentation
- Downloadable prediction results
- Production-ready API structure

## Tech Stack

- Python
- FastAPI
- Scikit-Learn
- Pandas
- Joblib
- Pydantic
- Uvicorn

## Project Structure

```text
House_prediction_Api/
│
├── main.py
├── train.py
├── explore.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── screenshots/
│   ├── screenshots\image.png
│   ├── screenshots\Screenshot 2026-06-21 171017.png
│
└── data/
```

## API Documentation

### Swagger UI

![Swagger UI](screenshots\image.png)

### Single Prediction Endpoint

![Single Prediction](screenshots\img1_fast_singlepred.png)

### Batch Prediction Endpoint

![Batch Prediction](screenshots\img2_fast_batchpred.png)

### Folder structure

![folder structure](screenshots\im3_fast_struc.png)

## Installation

Clone the repository:

```bash
git clone https://github.com/Deepak4053/fast_api_project.git
cd fast_api_project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn main:app --reload
```

Open Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Endpoint       | Description                   |
| ------ | -------------- | ----------------------------- |
| GET    | /              | Home page                     |
| GET    | /health        | Health check                  |
| POST   | /predict       | Single house price prediction |
| POST   | /predict/batch | Batch prediction using CSV    |

## Example Input

```json
{
  "MedInc": 8.5,
  "HouseAge": 15,
  "AveRooms": 6.2,
  "AveBedrms": 1.1,
  "Population": 1200,
  "AveOccup": 3.0,
  "Latitude": 34.05,
  "Longitude": -118.25
}
```

## Future Improvements

- Docker containerization
- CI/CD with GitHub Actions
- Cloud deployment (AWS/Azure/GCP)
- Model monitoring
- Database integration
- Authentication and rate limiting

## Author

Deepak
Aspiring AI/ML Engineer | FastAPI | Machine Learning | Python
