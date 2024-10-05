import os
from fastapi import FastAPI, Form, Depends, HTTPException
import replicate
import logging
from dotenv import load_dotenv
import re
from typing import Optional

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 250
MODEL_NAME = "mistralai/mixtral-8x7b-instruct-v0.1:7b3212fbaf88310cfef07a061ce94224e82efc8403c26fc67e8f6c065de51f21"

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

app = FastAPI()

def get_replicate_client():
    if not REPLICATE_API_TOKEN:
        logger.warning("REPLICATE_API_TOKEN is not set. Replicate functionality will be limited.")
        return None
    return replicate

def check_depression(local_generated_text: str, local_text_message: str) -> int:
  if not REPLICATE_API_TOKEN:
      logger.warning("REPLICATE_API_TOKEN is not set. Returning mock response.")
      return "Mock response: Unable to process without Replicate API token", -1
  # Remove the input text from the generated text
  if local_generated_text.startswith(local_text_message):
      processed_text = local_generated_text[len(local_text_message):].strip()
  else:
      processed_text = local_generated_text

  # Remove leading whitespace and special characters; also if there is a word called "answer"
  trimmed_text = re.sub(r'^\s*answer:\s*|^\s*\W+', '', processed_text, flags=re.IGNORECASE)

  #print(f"Processed text: {trimmed_text}")  

  if(not bool(trimmed_text)):
    is_depressed = -2
    #print(f"Is the poster depressed? {is_depressed}")  
    return is_depressed

  # Remove specific words/phrases
  trimmed_text = re.sub(r'Answer: |Response: ', '', trimmed_text)
  # Trim the word (removing leading and trailing whitespace and non-word characters)
  trimmed_text = re.sub(r'^[\s\W]+|[\s\W]+$', '', trimmed_text)  
  # Extract the first word
  first_word = trimmed_text.split()[0].lower()  # Convert to lowercase for case-insensitive comparison
  #print(f"First word: {first_word}")
  first_word_trimmed = re.sub(r'^[\s\W]+|[\s\W]+$', '', first_word)
  
  # Check if the first word is 'yes' or 'no'
  is_yes = first_word_trimmed == "yes"
  is_no = first_word_trimmed == "no"

  if (is_yes or is_no):
    is_depressed = 1 if is_yes else 0
  else:
    # Convert to lowercase for case-insensitive matching
    processed_text_lower = trimmed_text.lower()    
    is_depressed = -1  # default as 0

  #print(f"Is the poster depressed? {is_depressed}")  
  return is_depressed

def depr_fn_new(modelname: str, post_text: str, image2text: str) -> tuple[str, int]:

  '''prompt_text = "Given a text post from depression-related subreddits on Reddit, provide a probability score indicating the likelihood that the poster is experiencing depression. \
                Additionally, include any relevant keywords or phrases from the post that contribute to this assessment. Please also provide an explanation for the prediction. \
                Post : " +truncated_text+"\
                Image Description:\
                ++
                Analysis Request:\
                Question: Based on the content of the above social media post and the associated image description, \
                are there discernible signs or indicators of depression?\
                Required Response Format: Please provide a 'yes' or 'no' answer, followed by a brief explanation of the \
                reasoning behind this conclusion."'''

  prompt_text = """
Social Media Post:

{0}

Image Description:

{1}

Analysis Request:
Question: Based on the content of the above social media post and the associated image description, are there discernible signs or indicators of depression?
Required Response Format: Please provide a 'yes' or 'no' answer, followed by a brief explanation of the reasoning behind this conclusion.
""".format(post_text,image2text)

  output = replicate.run(
      modelname,
      input={
          "prompt": prompt_text,
          "max_new_tokens": MAX_NEW_TOKENS
          }
  )

  generated_text = ""
  for item in output:
    generated_text += item
  
  print(f"generated_text: {generated_text}")

  '''response = OAI_client.chat.completions.create(
      messages=[
          {
              "role": "user",
              "content": "Please analyze the following assessment made by a large language model regarding a social media post containing text and images. Determine if the language model has identified signs of depression. Based on its analysis, respond with a single word: 'Yes', 'No', or 'Uncertain'.  {0}".format(generated_text),
              }
          ],
      model="gpt-3.5-turbo",
      )
  # Accessing the message
  message_content = response.choices[0].message.content
  
  # Print the message content
  print(message_content)'''

  isdepressed = check_depression(generated_text,prompt_text)
  return generated_text, isdepressed

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

def get_replicate_client():
    return replicate

@app.post("/items/{item_id}", response_model=dict)
async def create_item(item_id: int, text: str = Form(...), replicate_client: replicate = Depends(get_replicate_client)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    try:
        if not replicate_client:
            return {
                "item_id": item_id,
                "response_text": "Unable to process: Replicate API token not set",
                "predictions": -1
            }
        model_generated_text, prediction = depr_fn_new(MODEL_NAME, text, "No image file provided")
        logger.info(f"Prediction: {prediction}")
        logger.info(f"Model generated response: {model_generated_text}")
        return {
            "item_id": item_id,
            "response_text": model_generated_text,
            "predictions": prediction
        }
    except Exception as e:
        logger.error(f"Error processing item {item_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")    
    
@app.get("/items/{item_id}", response_model=dict)
def read_item(item_id: int, text: Optional[str] = None):
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required")
    try:
        model_generated_text, prediction = depr_fn_new(MODEL_NAME, text, "No image file provided")
        logger.info(f"Prediction: {prediction}")
        logger.info(f"Model generated response: {model_generated_text}")
        return {
            "item_id": item_id,
            "response_text": model_generated_text,
            "predictions": prediction
        }
    except Exception as e:
        logger.error(f"Error processing item {item_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")
