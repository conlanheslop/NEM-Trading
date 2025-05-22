# NEM-Trading Python Application

A Python-based web application for predicting electricity prices in Australia's National Electricity Market (NEM). This module was developed as part of a larger energy analytics project. It uses Streamlit to provide a responsive web interface and connects to the OpenNEM API for real-time data. The system integrates with an externally trained XGBoost model to forecast future prices based on live market and weather conditions.

---

## Features

- Retrieve live energy and weather data from the [OpenNEM API](https://opennem.org.au/)
- Predict electricity prices using a trained XGBoost model
- Visualize trends and predictions with interactive charts (Plotly)
- Streamlit-based web dashboard for usability and presentation
- Built using modular, scalable Python components

---

## Requirements

- **Python 3.8+**
- **Streamlit**
- **NumPy, Pandas, Requests, Plotly** (included in `requirements.txt`)
- An IDE such as **Visual Studio Code**, **PyCharm**, or similar
- Internet connection (for fetching live data from OpenNEM)

---

## Getting Started

### 1. Install Python 3.8+

Make sure Python 3.8 or later is installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

### 2. Set up the Project Environment

You can install the dependencies using either a virtual environment or directly into your global Python installation.

#### Option A: Using a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: Global Installation

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Launch the Streamlit app from the terminal:

```bash
streamlit run app.py
```

If your main file has a different name, replace app.py with your actual filename:

```bash
streamlit run your_main_file.py
```

## How It Works

- The application fetches live data from the OpenNEM API, including electricity prices and weather-related metrics.

- This data is passed into a machine learning model trained using XGBoost (the model is trained outside of this repository).

- The output is a predicted electricity price, visualized through dynamic Plotly or Matplotlib charts on a user-friendly Streamlit interface.