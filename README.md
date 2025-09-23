# s3-classifier-project documentation

## Project Overview
This project aims to build a classifier using data stored in an S3 bucket. The data consists of comments and posts from Reddit, partitioned by year and month in parquet format. The classifier will be trained on this data to perform specific tasks, such as sentiment analysis or topic classification.

## Project Structure
```
s3-classifier-project
├── src
│   ├── main.py               # Entry point of the application
│   ├── data
│   │   ├── s3_config.py      # Configuration for S3 access
│   │   ├── loader.py          # Data loading functions
│   │   └── preprocess.py      # Data preprocessing functions
│   ├── model
│   │   ├── train.py           # Model training functions
│   │   └── evaluate.py        # Model evaluation functions
│   └── utils
│       └── helpers.py         # Utility functions
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── config
    └── config.yaml            # Configuration settings
```

## Setup Instructions
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the project, execute the following command:
```
python src/main.py
```
This will initiate the data loading, preprocessing, model training, and evaluation processes.

## Configuration
Configuration settings for S3 access and model parameters can be found in the following files:
- `src/data/s3_config.py`: Contains the S3 bucket details and credentials.
- `config/config.yaml`: Contains model parameters and data paths.
