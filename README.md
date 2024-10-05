MentalDisorderDetetion is licensed under the GNU General Public License v3.0.
Copyright (C) 2024 Nikila Swaminathan

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#Instructions for Testing this code

Here's a step-by-step guide to set up your testing environment:
1. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
2. Install required packages:
    pip install fastapi uvicorn python-dotenv replicate pytest pytest-asyncio httpx
    #fastapi
    #uvicorn (for running the FastAPI server)
    #python-dotenv (for loading environment variables)
    #replicate (for interacting with the Replicate API)
    #pytest (for running tests)
    #pytest-asyncio (for testing async functions)
    #httpx (for testing FastAPI applications)

3. Create a .env file in your project root (if testing with a token):
    REPLICATE_API_TOKEN=your_actual_token_here
4. Create a test_main.py file with your test cases.
5. Run your tests:
    pytest test_main.py

For manual testing:
6. Run your FastAPI application:
    uvicorn main:app --reload
7. Use a tool like Postman or curl to send requests to your endpoints:
    curl -X POST "http://localhost:8000/items/1" -H "Content-Type: application/x-www-form-urlencoded" -d "text=Your test text here"

Remember to test both with and without the Replicate API token to ensure your application handles both scenarios correctly. If you're writing the tests I suggested earlier, you'll also need to install pytest-mock:
    pip install pytest-mock

This setup will allow you to thoroughly test your FastAPI application, including unit tests for individual functions and integration tests for the API endpoints.


#Instruction to deploy this repo in the Render
## Deploy FastAPI on Render
Use this repo as a template to deploy a Python [FastAPI](https://fastapi.tiangolo.com) service on Render.

See https://render.com/docs/deploy-fastapi or follow the steps below:
### Manual Steps
1. You may use this repository directly or [create your own repository from this template](https://github.com/render-examples/fastapi/generate) if you'd like to customize the code.
2. Create a new Web Service on Render.
3. Specify the URL to your new repository or this repository.
4. Render will automatically detect that you are deploying a Python service and use `pip` to download the dependencies.
5. Specify the following as the Start Command.

    ```shell
    uvicorn main:app --host 0.0.0.0 --port $PORT
    ```

6. Click Create Web Service.

Or simply click:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/render-examples/fastapi)
