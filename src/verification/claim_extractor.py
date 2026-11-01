import re

def extract_claims(answer):

    claims = re.split(
        r'[.!?]',
        answer
    )

    claims = [
        c.strip()
        for c in claims
        if c.strip()
    ]

    return claims