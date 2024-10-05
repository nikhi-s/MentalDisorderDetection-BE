# Test for functions
import pytest
from main import check_depression, depr_fn_new

def test_check_depression():
    assert check_depression("Yes, there are signs of depression.", "Input text") == 1
    assert check_depression("No signs of depression.", "Input text") == 0
    assert check_depression("Uncertain response", "Input text") == -1
    assert check_depression("", "Input text") == -2

@pytest.mark.asyncio
async def test_depr_fn_new(mocker):
    # Mock the replicate.run function
    mock_run = mocker.patch('replicate.run')
    mock_run.return_value = iter(["Yes, there are signs of depression."])
    
    result, prediction = await depr_fn_new("model_name", "Test post", "Test image description")
    assert result == "Yes, there are signs of depression."
    assert prediction == 1

# Test for API endpoints
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item(mocker):
    # Mock the depr_fn_new function
    mocker.patch('main.depr_fn_new', return_value=("Test response", 1))
    
    response = client.post("/items/1", data={"text": "Test post"})
    assert response.status_code == 200
    assert response.json() == {
        "item_id": 1,
        "response_text": "Test response",
        "predictions": 1
    }

def test_read_item(mocker):
    # Mock the depr_fn_new function
    mocker.patch('main.depr_fn_new', return_value=("Test response", 0))
    
    response = client.get("/items/1?text=Test%20post")
    assert response.status_code == 200
    assert response.json() == {
        "item_id": 1,
        "response_text": "Test response",
        "predictions": 0
    }

def test_create_item_empty_text():
    response = client.post("/items/1", data={"text": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Text input cannot be empty"}

def test_read_item_missing_text():
    response = client.get("/items/1")
    assert response.status_code == 400
    assert response.json() == {"detail": "Text input is required"}

# Test for error handling
def test_create_item_error(mocker):
    # Mock depr_fn_new to raise an exception
    mocker.patch('main.depr_fn_new', side_effect=Exception("Test error"))
    
    response = client.post("/items/1", data={"text": "Test post"})
    assert response.status_code == 500
    assert response.json() == {"detail": "An error occurred while processing the request"}

def test_read_item_error(mocker):
    # Mock depr_fn_new to raise an exception
    mocker.patch('main.depr_fn_new', side_effect=Exception("Test error"))
    
    response = client.get("/items/1?text=Test%20post")
    assert response.status_code == 500
    assert response.json() == {"detail": "An error occurred while processing the request"}

import os

def test_environment_variables():
    assert "REPLICATE_API_TOKEN" in os.environ
    assert os.environ["REPLICATE_API_TOKEN"] != ""

def test_constants():
    from main import MAX_NEW_TOKENS, MODEL_NAME
    assert MAX_NEW_TOKENS == 250
    assert MODEL_NAME.startswith("mistralai/mixtral-8x7b-instruct-v0.1")
