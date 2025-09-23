import json
import boto3
import os
from typing import Dict, Any

# Initialize SageMaker client
sagemaker = boto3.client('sagemaker-runtime')
endpoint_name = os.environ['SAGEMAKER_ENDPOINT']

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for immigration text classification
    """
    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        # Extract text to classify
        text = body.get('text', '')
        if not text:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Text input is required'
                })
            }
        
        # Prepare payload for SageMaker
        payload = {
            'instances': [{'text': text}]
        }
        
        # Call SageMaker endpoint
        response = sagemaker.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response['Body'].read())
        
        # Format response
        classification_result = {
            'text': text,
            'category': result.get('predictions', [{}])[0].get('category', 'unknown'),
            'confidence': result.get('predictions', [{}])[0].get('confidence', 0.0),
            'timestamp': context.aws_request_id
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(classification_result)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            })
        }
