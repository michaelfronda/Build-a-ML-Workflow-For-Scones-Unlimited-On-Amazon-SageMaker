
### Test Step Function ###
{
  "image_data": "",
  "s3_bucket": "sagemaker-us-east-1-175778184020",
  "s3_key": "test/bicycle_s_000030.png"
}


"""
 serializeImageData:  
    Serialize the image data
"""
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]
    
    # Download the data from s3 to /tmp/image.png
    file_name = "/tmp/image.png"
    s3.download_file(bucket, key, file_name)
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }



"""
imageInference:
    Classify Image
"""
import boto3
import json
import base64

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2022-08-16-06-04-48-450'
# We will be using the AWS's lightweight runtime solution to invoke an endpoint.
runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])

    # Instantiate a Predictor
    predictor = runtime.invoke_endpoint(EndpointName=ENDPOINT,
                                       ContentType='image/png',
                                       Body=image)

    # We return the data back to the Step Function    
    event["body"]["inferences"] = json.loads(predictor['Body'].read().decode('utf-8'))
    return {
        'statusCode': 200,
        'body': {
            'image_data': event["body"]['image_data'],
            's3_bucket': event["body"]['s3_bucket'],
            's3_key': event["body"]['s3_key'],
            'inferences': event["body"]['inferences']
        }
        
    }


"""
 filterLowConfidenceInferences:  
    Filter Low Confidence Inferences
"""
import json


THRESHOLD = .7


def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = event['body']["inferences"]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = (max(inferences) > THRESHOLD)

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': {
            'image_data': event['body']['image_data'],
            's3_bucket': event['body']['s3_bucket'],
            's3_key': event['body']['s3_key'],
            'inferences': event['body']['inferences']
        }
    }



