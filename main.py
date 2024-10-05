import os
from fastapi import FastAPI, Form, Depends, HTTPException, UploadFile, File
import replicate
from replicate.exceptions import ReplicateError 
import logging
from dotenv import load_dotenv
import re
from typing import Optional
import base64 

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 250
LLAVA_MODEL_NAME = "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb"  # Replace with the LLAVA model name
MIXTRAL_MODEL_NAME = "mistralai/mixtral-8x7b-instruct-v0.1:7b3212fbaf88310cfef07a061ce94224e82efc8403c26fc67e8f6c065de51f21"
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

app = FastAPI()

def get_replicate_client():
    if not REPLICATE_API_TOKEN:
        logger.warning("REPLICATE_API_TOKEN is not set. Replicate functionality will be limited.")
        return None
    return replicate

def image_to_text(image: bytes) -> str:
    """Function to convert image to text using the LLAVA model."""
    try:
        image_base64 = base64.b64encode(image).decode('utf-8')
        prompt = """
        Analyze this image in detail, focusing on elements that might be relevant to mental health assessment. Consider the following:

        1. Image type: Is this a selfie, a photo of a person, or another type of image?

        2. If people are present:
           - Facial expressions and emotions conveyed
           - Body language and posture
           - Overall appearance and self-presentation
           - Any signs of stress, fatigue, or mood indicators

        3. If it's not a photo of a person:
           - Describe the main elements of the image
           - Analyze the mood or atmosphere of the image
           - Identify any symbols or themes that might relate to mental state

        4. Environmental context:
           - Describe the setting (if visible)
           - Note any unusual or significant elements in the environment

        5. Color palette and lighting:
           - Describe the overall color scheme and lighting
           - Consider how these might relate to mood or emotional state

        6. Any text or writing in the image:
           - Transcribe and interpret any visible text

        7. Overall impression:
           - Provide your interpretation of what this image might convey about the mental or emotional state of the person who chose to share it

        Provide a comprehensive description that could be used by mental health professionals to gain insights into the individual's emotional state and potential mental health concerns. Be objective and avoid making definitive diagnoses.
        """
        output = replicate.run(
            LLAVA_MODEL_NAME,
            input={
                "image": f"data:image/jpeg;base64,{image_base64}",
                "prompt": prompt
            }
        )
        logger.info(f"LLAVA model response: {output}")
        return output
    except ReplicateError as e:
        if e.status == 402:
            logger.error("Billing required for Replicate API. Please set up billing at https://replicate.com/account/billing#billing")
            raise HTTPException(status_code=500, detail="Server configuration error: Replicate API billing not set up")
        elif e.status == 422:
            logger.error(f"Input validation failed: {e.detail}")
            raise HTTPException(status_code=400, detail=f"Invalid input: {e.detail}")
        else:
            logger.error(f"Replicate API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing image with Replicate API")

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
  
    #print(f"generated_text: {generated_text}")

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
async def create_item(item_id: int, text: str = Form(...), image: UploadFile = File(...), replicate_client: Optional[replicate] = Depends(get_replicate_client)):
    
    logger.info(f"Received request for item_id: {item_id}")
    logger.info(f"Text content: {text}")
    logger.info(f"Image filename: {image.filename}")

    if not text.strip():
        logger.warning("Empty text received")
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    if not image.filename:
        logger.warning("No image file received")
        raise HTTPException(status_code=400, detail="Image file is required")

    try:
        if not replicate_client:
            return {
                "item_id": item_id,
                "response_text": "Unable to process: Replicate API token not set",
                "predictions": -1
            }
        
        # Process the image
        image_content = await image.read()  # Read the uploaded file content

        image_description = image_to_text(image_content)  # Get the text description of the image

        model_generated_text, prediction = depr_fn_new(MIXTRAL_MODEL_NAME, text, image_description)        
        logger.info(f"Prediction: {prediction}")
        logger.info(f"Model generated response: {model_generated_text}")
        return {
            "item_id": item_id,
            "response_text": model_generated_text,
            "predictions": prediction
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing item {item_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the request: {str(e)}")
    
@app.get("/items/{item_id}", response_model=dict)
async def get_item(item_id: int, text: str = Form(...), image: UploadFile = File(...), replicate_client: Optional[replicate] = Depends(get_replicate_client)):

    logger.info(f"Received request for item_id: {item_id}")
    logger.info(f"Text content: {text}")
    logger.info(f"Image filename: {image.filename}")

    if not text.strip():
        logger.warning("Empty text received")
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    if not image.filename:
        logger.warning("No image file received")
        raise HTTPException(status_code=400, detail="Image file is required")

    try:
        if not replicate_client:
            return {
                "item_id": item_id,
                "response_text": "Unable to process: Replicate API token not set",
                "predictions": -1
            }
        
        # Process the image
        image_content = await image.read()  # Read the uploaded file content

        image_description = image_to_text(image_content)  # Get the text description of the image

        model_generated_text, prediction = depr_fn_new(MIXTRAL_MODEL_NAME, text, image_description)        
        logger.info(f"Prediction: {prediction}")
        logger.info(f"Model generated response: {model_generated_text}")
        return {
            "item_id": item_id,
            "response_text": model_generated_text,
            "predictions": prediction
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing item {item_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the request: {str(e)}")