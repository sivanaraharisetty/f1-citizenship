import boto3
import traceback
from typing import Optional


def send_email_alert(subject: str, body: str, *, sender: str, recipient: str, region: str = "us-east-1") -> bool:
    """
    Send a simple email using AWS SES. Returns True if accepted by SES, False otherwise.
    Requires SES to be verified for sender/recipient in the configured region.
    """
    try:
        ses = boto3.client("ses", region_name=region)
        resp = ses.send_email(
            Source=sender,
            Destination={"ToAddresses": [recipient]},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
            },
        )
        return bool(resp.get("MessageId"))
    except Exception:
        # Fallback to stdout so failures are visible in logs
        print("[ALERT][FAILED TO SEND EMAIL]\n" + body)
        return False


def format_exception_alert(context: str, err: BaseException) -> str:
    return f"Context: {context}\n\n" + "\n".join(traceback.format_exception(type(err), err, err.__traceback__))
