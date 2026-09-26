import laya
from laya import Router

# Initialize router with preloading (avoids swap delay)
router = Router(preload=True)

# Define complex state
ticket = {
    "ticket_id": "TCK-8821",
    "customer": "enterprise_user",
    "subject": "System downtime and billing dispute",
    "body": "Our production API has been failing since 6 AM. We lost critical transactions. We demand an immediate SLA refund.",
}

# Define multiple questions of different primitives
questions = {
    "queue": {
        "type": "choice",
        "instructions": "Which engineering queue owns this ticket?",
        "criteria": {
            "infrastructure": "server outages, network downtime, database failures",
            "billing": "refunds, SLA credits, invoice disputes",
            "security": "breaches, vulnerability reports",
            "support": "general customer inquiries",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["low priority", "medium", "high priority", "critical blocker"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the customer threaten to cancel or express severe churn intent?",
    },
}

# Single forward pass: evaluates all questions simultaneously
res = router.predict(ticket, questions)

print("Routing Decision :", res["routing"]["model"])
# -> english

print("Assigned Queue   :", res["answers"]["queue"]["choice"])
# -> infrastructure (confidence: 0.96)

print("Urgency Score    :", res["answers"]["urgency"]["score"])
# -> 2.87 / 3.0

print("Churn Risk       :", f"{res['answers']['churn_risk']['noul']:.1%}")
# -> 91.4%
