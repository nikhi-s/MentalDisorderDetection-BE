from typing import Optional

from fastapi import FastAPI

import requests

from huggingface_hub import notebook_login

api_token = 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
# Login to the Hugging Face Model Hub with the provided token
notebook_login(api_token=api_token)

from datasets import load_dataset
import datasets

#raw_datasets = load_dataset("nikilas/reddit_depression", use_auth_token=True)
raw_datasets = load_dataset("nikilas/DepressionIncludingImageText", use_auth_token=True)
#Split the dataset into training, validation abd test datasets
train_testvalid = raw_datasets['train'].train_test_split(train_size=0.7, seed=42)
# Split the 30% (test + valid) in half test, half valid
test_valid = train_testvalid['test'].train_test_split(train_size=0.5, seed=42)
# gather everyone if you want to have a single DatasetDict
raw_datasets = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'val': test_valid['train']})
print("Splits available in the DatasetDict:", raw_datasets.keys())

import os
os.environ["REPLICATE_API_TOKEN"] = r8_IIHG7hbkTN0No13c5QcwxqKPIQYidDu1kZitY

import re

def check_depression(local_generated_text,local_text_message):

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
  log_file.write(f"\nFirst word: {first_word}")  
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

def depr_fn_new(modelname,post_text,image2text):

  max_words = 450  

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
          "max_new_tokens": 250
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

  isdepressed = check_depression(generated_text,prompt_text,log_file)
  return generated_text, isdepressed

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, text: Optional[str] = None):
    models = ["mistralai/mixtral-8x7b-instruct-v0.1:7b3212fbaf88310cfef07a061ce94224e82efc8403c26fc67e8f6c065de51f21"]
    for index, modelname in enumerate(models):
        # Process model_output to get the prediction of depression
        model_generated_text,prediction = depr_fn_new(modelname,text,"No image file provided")
        print("Prediction",prediction)
        print("Model_generated_response",model_generated_text)
    return {"item_id": item_id, "response text": model_generated_text,"Prediction",}
