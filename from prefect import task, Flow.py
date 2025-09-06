from prefect import task, flow
from sklearn.model_selection import train_test_split
from my_project import load_data, preprocess_data, train_model, evaluate_model, save_model

# Task 1: Load data
@task
def load_data_task():
    file_path = '/Users/niveditasaha/Downloads/NEW DATA/Final_structured data.xlsx'  
    df = load_data(file_path)
    return df

# Task 2: Preprocess data
@task
def preprocess_data_task(df):
    X_scaled, y, scaler = preprocess_data(df)
    return X_scaled, y, scaler

# Task 3: Train model
@task
def train_model_task(data):
    X_scaled, y, scaler = data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    model = train_model(X_train, y_train)
    return model, (X_test, y_test), scaler

# Task 4: Evaluate model
@task
def evaluate_model_task(model_data):
    model, test_data, scaler = model_data
    X_test, y_test = test_data
    evaluate_model(model, X_test, y_test)

# Task 5: Save model
@task
def save_model_task(model_data):
    model, test_data, scaler = model_data
    save_model(model, scaler)

# Define the flow
@flow
def ml_pipeline():
    df = load_data_task()
    preprocessed_data = preprocess_data_task(df)
    model_data = train_model_task(preprocessed_data)
    evaluate_model_task(model_data)
    save_model_task(model_data)

# Run the flow
if __name__ == "__main__":
    ml_pipeline()
