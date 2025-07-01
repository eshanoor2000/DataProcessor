import os
import json
import time
import sys
import logging
from datetime import datetime
from datetime import timedelta
import re
from rapidfuzz import fuzz
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc
import psutil
import pymongo
from pymongo.errors import BulkWriteError
import smtplib
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import spacy
from collections import Counter

# --- Configuration from environment variables ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "brand_monitoring")
RAW_COLLECTION = os.getenv("RAW_COLLECTION", "raw_articles")
PROCESSED_COLLECTION = os.getenv("PROCESSED_COLLECTION", "processed_articles")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

# Logging Configuration
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# Initialize logging
configure_logging()

UNICODE_REPLACEMENTS = {
    "\u2013": "-",
    "\u2014": "—",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u201a": ",",
    "\u201e": '"',
    "\u2032": "'",
    "\u2033": '"',
}

AD_MESSAGES = [
    "We and select advertising partners collect some of your data in order to give you a better experience through personalized content and advertising",
    "To limit the collection of your data, please modify your privacy preferences",
    "You can learn more in our Privacy Notice (new window)",
    "Searching for your content... Phone 877-269-7890 from 8 AM - 10 PM ET Contact Cision 877-269-7890 from 8 AM - 10 PM ET Nov 01, 2024, 12:30 ET Share this article"
]

EMAIL_CONFIG = {
    "EMAIL_SENDER": EMAIL_SENDER,
    "EMAIL_PASSWORD": EMAIL_PASSWORD,
    "EMAIL_RECEIVER": EMAIL_RECEIVER,
    "SMTP_SERVER": SMTP_SERVER,
    "SMTP_PORT": SMTP_PORT,
}

def get_collection(collection_name):
    if not hasattr(get_collection, "client"):
        get_collection.client = pymongo.MongoClient(MONGO_URI)
    return get_collection.client[MONGO_DB][collection_name]

def clear_memory():
    """Clear memory caches and perform garbage collection"""
    torch.cuda.empty_cache()
    gc.collect()
    logging.info("Memory cache cleared")

def check_memory():
    """Check current memory usage and log it"""
    mem = psutil.virtual_memory()
    logging.info(f"Memory usage: {mem.percent}% ({mem.used/1024/1024:.1f}MB used of {mem.total/1024/1024:.1f}MB)")
    if mem.percent > 70:
        clear_memory()
    return mem.percent

# Sentiment analysis models with lazy loading and cleanup
_sentiment_models = {}

def get_roberta_model():
    if "roberta" not in _sentiment_models:
        try:
            _sentiment_models["roberta"] = {
                "tokenizer": AutoTokenizer.from_pretrained(
                    "cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            }
            
            if check_memory() > 60:
                clear_memory()
            
            # Try loading with safetensors first to avoid torch.load security issue
            try:
                _sentiment_models["roberta"]["model"] = AutoModelForSequenceClassification.from_pretrained(
                    "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    use_safetensors=True
                )
            except Exception as safetensors_error:
                logging.warning(f"Failed to load with safetensors: {safetensors_error}")
                # Fallback to regular loading
                _sentiment_models["roberta"]["model"] = AutoModelForSequenceClassification.from_pretrained(
                    "cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
        except Exception as e:
            logging.error(f"Failed to load RoBERTa model: {e}")
            if "roberta" in _sentiment_models:
                del _sentiment_models["roberta"]
            raise
    return _sentiment_models["roberta"]

def get_topic_model():
    if not hasattr(get_topic_model, "model"):
        get_topic_model.model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    return get_topic_model.model

# --- New Topic Tagging Logic ---
# Rich topic anchors with description, keywords, and examples
TOPIC_ANCHORS = {
    "Records": {
        "description": "Issues with requesting or receiving Condo documents and information, including financials, meeting minutes, and records of decisions about the condo. Condominium corporations create, maintain, and provide records to owners, purchasers, and mortgagees on request.",
        "keywords": ["record", "document", "financial", "access", "AGM", "annual general meeting", "status certificate", "lists and addresses of owners", "contract", "budget", "reserve fund", "copy of returns", "notice of change", "ballot"],
        "examples": [
            "How do I access condo records and financial documents?",
            "The board is refusing to provide AGM minutes.",
            "Condo owners have a right to access some records, but not all."
        ]
    },
    "Noise": {
        "description": "Concerns about persistent or disruptive sounds coming from neighbours or common areas in condominium. Common sources are another resident, a utility, an amenity, piping, ventilation, or external source such as construction.",
        "keywords": ["noise", "loud", "music", "banging", "stomping", "disruption", "loud footsteps", "loud talking or arguing", "slamming of doors", "pet making noise", "loud music and tv", "drilling", "hammering"],
        "examples": [
            "Loud music from my neighbour is constant.",
            "There is banging and stomping above me every night.",
            "When sound becomes persistent or excessively loud."
        ]
    },
    "Vibration": {
        "description": "Issues related to physical vibrations or tremors identifiable by feel that affect comfort or safety in a unit.",
        "keywords": ["vibration", "shaking", "tremors", "construction", "movement"],
        "examples": [
            "My floor shakes when someone walks upstairs.",
            "Tremors from nearby construction are disturbing residents.",
            "Tremors through the walls or floors can disrupt quality of life."
        ]
    },
    "Condo managers": {
        "description": "Issues with condo management including delays, no responses, or unprofessional behavior.",
        "keywords": ["management", "condo manager", "manager", "no response", "ignored", "inaction", "abuse of power", "mismanagement", "bad leadership"],
        "examples": [
            "The condo manager is ignoring my complaints.",
            "I submitted a request and got no response from management.",
            "Inaction by management can affect day-to-day operations."
        ]
    },
    "Pets & animals": {
        "description": "Complaints related to pets, including noise, leash rules, and disturbances.",
        "keywords": ["pet", "animal", "bark", "off-leash", "pet causing disturbance", "dangerous pet", "illegal pet"],
        "examples": [
            "My neighbour's dog barks all day.",
            "There are complaints about pets being off-leash in hallways.",
            "Even the most lovable companions can disrupt peace and quiet."
        ]
    },
    "Parking & storage": {
        "description": "Issues involving parking spaces, storage lockers, and misuse or theft.",
        "keywords": ["parking spot", "storage", "locker", "assigned spot", "theft", "park a car"],
        "examples": [
            "Someone keeps parking in my assigned spot.",
            "My storage locker was broken into.",
            "Issues arise when you're dealing with limited space."
        ]
    },
    "Smoke & vapour": {
        "description": "Concerns about the transmission of smoke or vapour between units.",
        "keywords": ["smoke", "cannabis", "tobacco", "vapour", "vent", "seep"],
        "examples": [
            "Cannabis smoke is entering my unit from the hallway.",
            "The smell of tobacco is coming through the vents.",
            "When tobacco, cannabis or other vapours seep into other units."
        ]
    },
    "Odours": {
        "description": "Unpleasant or persistent smells in common areas or units.",
        "keywords": ["odour", "smell", "spice", "sewage", "unpleasant", "odours caused by cannabis", "smell caused by pet"],
        "examples": [
            "The hallway constantly smells of strong spices.",
            "There's a sewage-like odour near my unit.",
            "Unpleasant smells can be a nuisance for other occupants."
        ]
    },
    "Light": {
        "description": "Disruptions caused by excessive or inadequate lighting in or around the condo building.",
        "keywords": ["light", "bright", "flickering", "security light", "disruptive"],
        "examples": [
            "Bright security lights are shining into my bedroom.",
            "Common area lighting is flickering and disruptive.",
            "Too bright? Too dim? There may be a simple solution."
        ]
    },
    "Vehicles": {
        "description": "Concerns about vehicle operation in garages or near the condo, including safety and idling.",
        "keywords": ["vehicle", "speeding", "idling", "garage", "dangerous driving", "entrance"],
        "examples": [
            "Residents are speeding in the underground garage.",
            "Cars are idling too long near building entrances.",
            "From idling cars to dangerous driving, see what rules apply."
        ]
    },
    "Settlement Agreements": {
        "description": "Non-compliance with decisions or agreements made through the Condo Authority Tribunal (CAT).",
        "keywords": ["settlement", "agreement", "CAT", "compliance", "ignored", "tribunal", "resolve issue"],
        "examples": [
            "The other party isn't complying with our CAT settlement.",
            "We reached an agreement but it's being ignored.",
            "Steps to take when someone doesn't comply with your Condo Tribunal settlement agreement."
        ]
    },
    "Harassment": {
        "description": "Experiencing threatening, discriminatory, or inappropriate behavior from others in the condo.",
        "keywords": ["harassment", "threatening", "discrimination", "unwelcome", "remark", "bullying", "verbal confrontation", "physical confrontation"],
        "examples": [
            "A neighbour is making discriminatory remarks toward me.",
            "I feel threatened by repeated interactions with another resident.",
            "Unwelcome, threatening or discriminatory interactions with other residents."
        ]
    },
    "Infestation": {
        "description": "Presence of mold or unwanted animal pests such as insects, rodents, or birds within the condo building.",
        "keywords": ["infestation", "cockroach", "mice", "mould", "bedbug", "pest", "bacteria", "fungus"],
        "examples": [
            "There are cockroaches in the stairwell.",
            "My unit has mice and the board won't help.",
            "From mice and mould to birds and bedbugs, infestation is a serious issue."
        ]
    },
    "Short-term rentals": {
        "description": "Concerns about temporary rentals like Airbnb affecting condo security and atmosphere.",
        "keywords": ["Airbnb", "rent", "short term stay", "sublease", "tenant"],
        "examples": [
            "My neighbour is running an Airbnb in their unit.",
            "There are constant strangers entering and leaving the unit next door.",
            "What to do when short-term rentals do not follow the condo rules."
        ]
    },
    "Meetings": {
        "description": "Issues surrounding the scheduling, cancellation, or transparency of condo meetings.",
        "keywords": ["meeting", "AGM", "minutes", "notice", "cancelled"],
        "examples": [
            "The AGM was cancelled without notice.",
            "Meeting minutes were never distributed.",
            "Meetings are an important way for condo communities to come together transparently."
        ]
    },
    "Other": {
        "description": "Any concern that doesn't clearly fall under one of the listed categories.",
        "keywords": ["other"],
        "examples": [
            "My issue doesn't fall into any of the standard categories.",
            "This problem doesn't seem to be listed anywhere.",
            "Any issue that does not fit the predefined categories."
        ]
    },
    "Fee Increases & Special Assessments": {
        "description": "Concerns about unexpected or unaffordable increases to monthly condo fees or lump-sum special assessments, involving lack of transparency, limited payment options, or poor communication about financial planning.",
        "keywords": ["fee", "special assessment", "maintenance fee", "increase", "unexpected cost", "financial burden", "payment plan", "assessment notice", "cost of repairs", "budget shortfall"],
        "examples": [
            "The board imposed a $40,000 special assessment without warning.",
            "Our condo fees went up 25% in a single year.",
            "We weren't given any payment plan options for the special assessment."
        ]
    },
    "Building Safety & Maintenance Neglect": {
        "description": "Issues arising from unsafe or deteriorating building conditions, often due to delayed or avoided maintenance. ",
        "keywords": ["safety", "structural", "leak", "mould", "exposed wire", "damaged", "repair", "maintenance backlog", "hazard", "crack", "disrepair", "unsafe condition"],
        "examples": [
            "There are cracks in the walls and the ceiling leaks every time it rains.",
            "Our stairwells have exposed wires and broken lights.",
            "Maintenance has been ignored for years and the building is falling apart."
        ]
    },
    "Board Election Abuse & Proxy Fraud": {
        "description": "Problems related to how condo board elections are conducted, such as forged proxies, ineligible board members, lack of transparency in vote counting, or abuse of electronic voting systems.",
        "keywords": ["election", "proxy", "fraud", "vote", "ballot", "forgery", "electronic voting", "board takeover", "illegitimate director", "voting irregularity"],
        "examples": [
            "Board members were elected using forged proxy forms.",
            "One director doesn't even live in the building but controls all votes.",
            "Our board has been hijacked through unfair election practices."
        ]
    },
    "Financial Mismanagement & Transparency": {
        "description": "Concerns about how condo funds are used or reported, including undisclosed loans, lack of access to financial records, misuse of budgets, and decisions made without owner knowledge or approval.",
        "keywords": ["loan", "financial abuse", "budget misuse", "unauthorized spending", "transparency", "financial decision", "hidden expense", "no disclosure", "audit", "oversight", "mismanagement"],
        "examples": [
            "The board took out a $3 million loan without telling the owners.",
            "We suspect financial abuse but the board won't share budget details.",
            "Funds are being used for unclear or unapproved projects."
        ]
    },
    "Security & Surveillance Misuse": {
        "description": "Issues where surveillance or security measures are perceived as excessive, misused, or invasive.",
        "keywords": ["security", "surveillance", "camera", "monitored", "WhatsApp group", "privacy", "patrol", "CCTV", "guard", "feeling watched", "security abuse"],
        "examples": [
            "Residents are being watched constantly through hallway cameras.",
            "Security uses a WhatsApp group to track residents' behavior.",
            "Our building feels more like a prison than a home."
        ]
    }
}

# Load spaCy's English language model
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def lemmatize_text(text):
    nlp = get_nlp()
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space])

def lemmatize_list(text_list):
    return [lemmatize_text(text) for text in text_list]

def lemmatize_topic_anchors(data):
    for topic, content in data.items():
        if "keywords" in content:
            content["keywords"] = lemmatize_list(content["keywords"])
        if "examples" in content:
            content["examples"] = lemmatize_list(content["examples"])
    return data

# Precompute lemmatized topic anchors and topic embeddings
LEMMATIZED_TOPIC_ANCHORS = lemmatize_topic_anchors(json.loads(json.dumps(TOPIC_ANCHORS)))
_TOPIC_EMBEDDINGS = None
def get_topic_embeddings():
    global _TOPIC_EMBEDDINGS
    if _TOPIC_EMBEDDINGS is None:
        model = get_topic_model()
        _TOPIC_EMBEDDINGS = {
            topic: model.encode(
                " ".join(data["keywords"] + data["examples"] + [data["description"]]),
                convert_to_tensor=True
            )
            for topic, data in LEMMATIZED_TOPIC_ANCHORS.items()
        }
    return _TOPIC_EMBEDDINGS

def assign_topic(text, threshold=0.5, max_topics=3):
    """Assign up to max_topics above the similarity threshold (multiple topics allowed, always include 'Other' if none)"""
    if not text:
        return ["Other"]
    model = get_topic_model()
    topic_embeddings = get_topic_embeddings()
    lemmatized_text = lemmatize_text(text)
    embedding = model.encode(lemmatized_text, convert_to_tensor=True)
    matched_topics = []
    for topic, topic_embedding in topic_embeddings.items():
        sim_score = util.cos_sim(embedding, topic_embedding).item()
        if sim_score >= threshold:
            matched_topics.append((topic, sim_score))
    matched_topics.sort(key=lambda x: x[1], reverse=True)
    if not matched_topics:
        return ["Other"]
    return [item[0] for item in matched_topics[:max_topics]]

def clean_text(text):
    """Clean text by removing unwanted characters and ads"""
    if not text:
        return ""
    
    for bad_char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(bad_char, replacement)
    
    for ad in AD_MESSAGES:
        text = text.replace(ad, "")
    
    return text.strip()

def extract_sentences(text, tags):
    """Extract sentences containing tags or fuzzy matches"""
    doc = get_nlp()(text)
    exact = [s.text for s in doc.sents if any(t.lower() in s.text.lower() for t in tags)]
    if exact:
        return exact
    fuzzy = []
    for s in doc.sents:
        if any(fuzz.partial_ratio(t.lower(), s.text.lower()) >= 90 for t in tags):
            fuzzy.append(s.text)
    return fuzzy or [text]

def classify_sentiment(text):
    """Classify sentiment using Twitter-RoBERTa"""
    try:
        model = get_roberta_model()
        enc = model["tokenizer"](text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model["model"](**enc)
        scores = torch.nn.functional.softmax(out.logits, dim=-1)[0]
        idx = scores.argmax().item()
        labels = ["negative", "neutral", "positive"]
        return labels[idx], float(scores[idx])
    except Exception as e:
        logging.error(f"Sentiment classification failed: {e}")
        return "neutral", 0.0



def analyze_sentiment(text, tags):
    """Analyze sentiment using Twitter-RoBERTa with sentence extraction and robust aggregation. Returns array format for compatibility."""
    try:
        # Extract relevant sentences
        sentences = extract_sentences(text, tags)
        # Analyze each sentence
        sentiment_analyses = []
        for sentence in sentences:
            label, score = classify_sentiment(sentence)
            sentiment_analyses.append({
                "text": sentence,
                "label": label,
                "score": round(score, 4)
            })
        # If no sentences, analyze the whole text
        if not sentiment_analyses:
            label, score = classify_sentiment(text)
            sentiment_analyses.append({
                "text": text,
                "label": label,
                "score": round(score, 4)
            })
        
        # Aggregate multiple sentiment analyses into a single result (like sort_json script)
        if len(sentiment_analyses) > 1:
            label_counts = Counter()
            label_score_sums = Counter()
            total_score = 0.0

            for sa in sentiment_analyses:
                lbl = sa['label']
                score = sa['score']
                label_counts[lbl] += 1
                label_score_sums[lbl] += score
                total_score += score

            def key_for(lbl):
                return (label_counts[lbl], label_score_sums[lbl] / label_counts[lbl])

            dominant_label = max(label_counts, key=key_for)
            avg_score = total_score / len(sentiment_analyses)
            combined_text = " | ".join(sa['text'].strip() for sa in sentiment_analyses)

            return [{
                'text': combined_text,
                'label': dominant_label,
                'score': round(avg_score, 4)
            }]
        else:
            return sentiment_analyses
            
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {e}")
        return [{"text": text, "label": "neutral", "score": 0.0}]

# Add mapping from topic to grouping
ISSUE_GROUPING_MAP = {
    "Condo managers": "Governance & Management",
    "Meetings": "Governance & Management",
    "Records": "Governance & Management",
    "Board Election Abuse & Proxy Fraud": "Governance & Management",
    "Settlement Agreements": "Governance & Management",
    "Fee Increases & Special Assessments": "Financial Issues",
    "Financial Mismanagement & Transparency": "Financial Issues",
    "Harassment": "Resident Behaviour & Conflict",
    "Short-term rentals": "Resident Behaviour & Conflict",
    "Pets & animals": "Resident Behaviour & Conflict",
    "Vehicles": "Resident Behaviour & Conflict",
    "Noise": "Nuisance & Disruption",
    "Odours": "Nuisance & Disruption",
    "Smoke & vapour": "Nuisance & Disruption",
    "Vibration": "Nuisance & Disruption",
    "Light": "Nuisance & Disruption",
    "Infestation": "Health, Safety & Infrastructure",
    "Building Safety & Maintenance Neglect": "Health, Safety & Infrastructure",
    "Parking & storage": "Health, Safety & Infrastructure",
    "Security & Surveillance Misuse": "Health, Safety & Infrastructure",
    "Other": "Other"
}

def assign_issue_grouping(topics):
    # topics is a list of topic names
    groupings = set()
    for topic in topics:
        group = ISSUE_GROUPING_MAP.get(topic, "Other")
        groupings.add(group)
    return list(groupings)

def normalize_article(article, processed_fields):
    # Required fields must be present and non-empty
    required_fields = [
        "title", "link", "content", "published_date", "scraped_date", "tags", "source", "sentiment_analysis", "assigned_issue", "issue_grouping"
    ]
    for field in required_fields:
        if not article.get(field):
            return None
    # subreddit, upvotes, comments can be null
    doc = {
        "title": article["title"],
        "link": article["link"],
        "content": article["content"],
        "published_date": article["published_date"],
        "tags": article["tags"],
        "source": article["source"],
        "subreddit": article.get("subreddit", None),
        "upvotes": article.get("upvotes", None),
        "comments": article.get("comments", None),
        "scraped_date": article["scraped_date"],
        "sentiment_analysis": article["sentiment_analysis"],
        "assigned_issue": article["assigned_issue"],
        "issue_grouping": article["issue_grouping"]
    }
    return doc

# Article Processing Function
def process_articles():
    """Process all articles scraped from March 6, 2025 up to today (inclusive)"""
    raw_collection = get_collection(RAW_COLLECTION)
    processed_collection = get_collection(PROCESSED_COLLECTION)

    start_date = datetime(2025, 3, 6)
    today = datetime.utcnow()
    query = {
        "processing_status": "pending",
        "scraped_date": {"$gte": start_date.isoformat(), "$lte": today.isoformat()},
        "tags": {"$exists": True, "$ne": []}
    }

    logging.info(f"Processing articles scraped from {start_date.date()} to {today.date()}")

    pending_articles = list(raw_collection.find(query))

    if not pending_articles:
        logging.info("No pending articles found for the specified date range.")
        return 0

    processed_count = 0

    for article in pending_articles:
        try:
            title = clean_text(article.get("title", ""))
            content = clean_text(article.get("content", ""))
            text_to_analyze = f"{title} {content}"
            tags = article.get("tags", [])
            source = article.get("source", "")
            # Normalize source
            if source.strip().lower() in ["bing", "web scraping"]:
                source = "Other"
            elif source.strip().lower() == "tocondo":
                source = "Toronto Condo News"
            subreddit = article.get("subreddit", None)
            upvotes = article.get("upvotes", None)
            comments = article.get("comments", None)
            published_date = article.get("published_date", "")
            scraped_date = article.get("scraped_date", "")

            # --- RAW COLLECTION DUPLICATE CHECK (new) ---
            if article.get("source", "").lower() == "tocondo":
                is_duplicate_raw = raw_collection.find_one({
                    "link": article.get("link"),
                    "published_date": article.get("published_date"),
                    "_id": {"$ne": article["_id"]}
                })
            else:
                is_duplicate_raw = raw_collection.find_one({
                    "link": article.get("link"),
                    "_id": {"$ne": article["_id"]}
                })

            if is_duplicate_raw:
                logging.info(f"Duplicate found in raw collection: {article.get('link')}")
                raw_collection.update_one(
                    {"_id": article["_id"]},
                    {"$set": {"processing_status": "duplicate"}}
                )
                continue
            # --- END RAW COLLECTION DUPLICATE CHECK ---

            # --- PROCESSED COLLECTION DUPLICATE CHECK ---
            if article.get("source", "").lower() == "tocondo":
                is_duplicate = processed_collection.find_one({
                    "link": article.get("link"),
                    "published_date": article.get("published_date")
                })
            else:
                is_duplicate = processed_collection.find_one({
                    "link": article.get("link")
                })
            
            if is_duplicate:
                logging.info(f"Skipping duplicate in processed: {article.get('link')}")
                raw_collection.update_one(
                    {"_id": article["_id"]},
                    {"$set": {"processing_status": "duplicate"}}
                )
                continue
                
            # --- END PROCESSED COLLECTION DUPLICATE CHECK ---
            
            # Analyze sentiment and topics
            sentiment_result = analyze_sentiment(text_to_analyze, tags)
            topics = assign_topic(text_to_analyze)
            issue_groupings = assign_issue_grouping(topics)

            base_doc = {
                "title": title,
                "link": article.get("link", ""),
                "content": content,
                "published_date": published_date,
                "tags": tags,
                "source": source,
                "subreddit": subreddit,
                "upvotes": upvotes,
                "comments": comments,
                "scraped_date": scraped_date,
                "sentiment_analysis": sentiment_result,
                "assigned_issue": topics,
                "issue_grouping": issue_groupings
            }

            normalized_doc = normalize_article(base_doc, [
                "title", "link", "content", "published_date", "tags", "source", "subreddit", "upvotes", "comments", "scraped_date", "sentiment_analysis", "assigned_issue", "issue_grouping"
            ])
            if not normalized_doc:
                logging.warning(f"Skipping article due to missing required fields: {article.get('link')}")
                continue

            processed_collection.insert_one(normalized_doc)

            raw_collection.update_one(
                {"_id": article["_id"]},
                {"$set": {"processing_status": "processed"}}
            )

            processed_count += 1

            if processed_count % 10 == 0:
                clear_memory()

        except Exception as e:
            logging.error(f"Error processing article {article.get('title')}: {e}")
            raw_collection.update_one(
                {"_id": article["_id"]},
                {"$set": {"processing_status": "error"}}
            )
            continue

    logging.info(f"Processed {processed_count} articles")
    return processed_count

def send_email(subject, body):
    try:
        missing_keys = [k for k, v in EMAIL_CONFIG.items() if not v]
        if missing_keys:
            logging.warning(f"Missing email config values: {missing_keys}. Skipping email.")
            return

        logging.info(f"Sending email to {EMAIL_CONFIG['EMAIL_RECEIVER']}")
        logging.debug(f"Subject: {subject}")
        logging.debug(f"Body: {body}")
        logging.debug(f"SMTP: {EMAIL_CONFIG['SMTP_SERVER']}:{EMAIL_CONFIG['SMTP_PORT']}")

        msg = MIMEMultipart()
        msg["From"] = EMAIL_CONFIG["EMAIL_SENDER"]
        msg["To"] = EMAIL_CONFIG["EMAIL_RECEIVER"]
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(EMAIL_CONFIG["SMTP_SERVER"], EMAIL_CONFIG["SMTP_PORT"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["EMAIL_SENDER"], EMAIL_CONFIG["EMAIL_PASSWORD"])
            server.sendmail(EMAIL_CONFIG["EMAIL_SENDER"], EMAIL_CONFIG["EMAIL_RECEIVER"], msg.as_string())

        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def send_success_email(processed_count):
    send_email(
        subject=f"Processing Complete - {processed_count} Articles Processed",
        body=f"Data processing completed successfully.\n\nProcessed {processed_count} articles."
    )

def send_error_email(error_message):
    send_email("Processing Script Encountered an Error", f"The processing script encountered an error:\n\n{error_message}")

def send_no_articles_email():
    subject = "No Articles Processed Today"
    body = (
        "The processor ran successfully,\n"
        "but no new articles were found to process today.\n\n"
        "This might mean the scraper found duplicates only, or the scrape job didn't run."
    )
    send_email(subject, body)

def send_monthly_volume_alert():
    collection = get_collection(PROCESSED_COLLECTION)
    today = datetime.utcnow()
    first_day = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    count = collection.count_documents({
        "scraped_date": {
            "$gte": first_day.isoformat()
        }
    })

    if count >= 5:
        subject = f"Monthly Article Threshold Met - {count} Articles"
        body = f"{count} articles have been processed this month so far.\n\nKeep an eye on content trends!"
        send_email(subject, body)

def validate_db_connection():
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Force connection
        logging.info("MongoDB connection validated.")
    except Exception as e:
        raise RuntimeError(f"MongoDB connection failed: {e}")

# Main function
if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()  # Ensures secrets are loaded

        configure_logging()
        validate_db_connection()

        logging.info("Starting GitHub-scheduled article processing job...")
        processed_count = process_articles()

        if processed_count > 0:
            send_success_email(processed_count)
        else:
            logging.info("No articles to process today.")
            send_no_articles_email()

        send_monthly_volume_alert()

    except Exception as e:
        logging.error(f"Processing failed: {e}")
        send_error_email(str(e))
        sys.exit(1)
