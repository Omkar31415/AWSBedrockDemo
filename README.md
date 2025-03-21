# AWSBedrockDemo

1. Login to amazon console 

2. Create IAM user with `AdminAccess` and `FullBedRockAccess` and create a AccessToken for that user and save it.

3. Go to Amazon Bedrock, in region select `us-east-1 (N.Virginia)` and have access to desired models for this application you need access to `Mistral 7B Instruct` & `Llama 3 8B Instruct`

4. Now in your pwd:
```
aws configure
```
- Enter IAM accesskey from IAM user
- Enter Secret access key from IAM user
- Region: us-east-1
- Output format: json

5. 
```
conda create -p venv python==3.10 -y
```

6. 
```
conda activate venv\
```
7. 
```
pip install -r requirements.txt
```
8. 
```
streamlit run app.py
```

Make sure to follow all the above steps.