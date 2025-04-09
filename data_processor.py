import os
import json
import time
import sys
import logging
from datetime import datetime
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

# Logging Configuration
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# Initialize logging
configure_logging()

# Database configuration
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "brand_monitoring")
RAW_COLLECTION = os.getenv("RAW_COLLECTION", "raw_articles")
PROCESSED_COLLECTION = os.getenv("PROCESSED_COLLECTION", "processed_articles")

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
    "sender_email": os.getenv("SENDER_EMAIL"),
    "sender_password": os.getenv("SENDER_PASSWORD"),
    "receiver_email": os.getenv("RECEIVER_EMAIL"),
    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.getenv("SMTP_PORT", 587)),
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

def get_bertweet_model():
    if "bertweet" not in _sentiment_models:
        try:
            _sentiment_models["bertweet"] = {
                "tokenizer": AutoTokenizer.from_pretrained(
                    "finiteautomata/bertweet-base-sentiment-analysis",
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            }
            
            if check_memory() > 60:
                clear_memory()
            
            _sentiment_models["bertweet"]["model"] = AutoModelForSequenceClassification.from_pretrained(
                "finiteautomata/bertweet-base-sentiment-analysis",
                device_map="auto",
                torch_dtype=torch.float16
            )
        except Exception as e:
            logging.error(f"Failed to load BERTweet model: {e}")
            if "bertweet" in _sentiment_models:
                del _sentiment_models["bertweet"]
            raise
    return _sentiment_models["bertweet"]

def get_vader_analyzer():
    if not hasattr(get_vader_analyzer, "analyzer"):
        get_vader_analyzer.analyzer = SentimentIntensityAnalyzer()
    return get_vader_analyzer.analyzer

def clear_sentiment_models():
    if "bertweet" in _sentiment_models:
        del _sentiment_models["bertweet"]["model"]
        del _sentiment_models["bertweet"]["tokenizer"]
        _sentiment_models.pop("bertweet", None)
    torch.cuda.empty_cache()

def get_topic_model():
    if not hasattr(get_topic_model, "model"):
        get_topic_model.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return get_topic_model.model

def clean_text(text):
    """Clean text by removing unwanted characters and ads"""
    if not text:
        return ""
    
    for bad_char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(bad_char, replacement)
    
    for ad in AD_MESSAGES:
        text = text.replace(ad, "")
    
    return text.strip()

def analyze_sentiment(text):
    """Analyze sentiment using both VADER and BERTweet"""
    try:
        # First try with VADER (lightweight)
        vader_scores = get_vader_analyzer().polarity_scores(text)
        
        # Only use BERTweet if text is complex enough
        if len(text.split()) > 10:
            try:
                model = get_bertweet_model()
                # Remove the character limit and just use truncation
                inputs = model["tokenizer"](text, 
                                         return_tensors="pt", 
                                         truncation=True)
                outputs = model["model"](**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                bertweet_scores = {
                    "positive": probs[0][2].item(),
                    "neutral": probs[0][1].item(),
                    "negative": probs[0][0].item(),
                    "compound": (probs[0][2].item() - probs[0][0].item())
                }
            except Exception as e:
                logging.warning(f"BERTweet failed: {e}")
                bertweet_scores = None
        else:
            bertweet_scores = None
            
        return {
            "vader": vader_scores,
            "bertweet": bertweet_scores,
            "final_sentiment": determine_final_sentiment(vader_scores, bertweet_scores)
        }
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {e}")
        return None

def determine_final_sentiment(vader_scores, bertweet_scores):
    """Determine final sentiment based on both models"""
    if bertweet_scores:
        # Use BERTweet if available and confident
        if max(bertweet_scores.values()) > 0.7:
            if bertweet_scores["positive"] > bertweet_scores["negative"]:
                return "positive"
            else:
                return "negative"
    
    # Fall back to VADER
    if vader_scores["compound"] >= 0.05:
        return "positive"
    elif vader_scores["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"

def assign_topic(text, threshold=0.45):
    if not text:
        return "Other"

    model = get_topic_model()
    topic_anchors = {
        "Records": [
            "How do I access condo records and financial documents?",
            "The board is refusing to provide AGM minutes.",
            "Condo owners have a right to access some records, but not all."
        ],
        "Noise": [
            "Loud music from my neighbour is constant.",
            "There's banging and stomping above me every night.",
            "When sound becomes persistent or excessively loud."
        ],
        "Vibration": [
            "My floor shakes when someone walks upstairs.",
            "Tremors from nearby construction are disturbing residents.",
            "Tremors through the walls or floors can disrupt quality of life."
        ],
        "Condo managers": [
            "The condo manager is ignoring my complaints.",
            "I submitted a request and got no response from management.",
            "Inaction by management can affect day-to-day operations."
        ],
        "Pets & animals": [
            "My neighbour's dog barks all day.",
            "There are complaints about pets being off-leash in hallways.",
            "Even the most lovable companions can disrupt peace and quiet."
        ],
        "Parking & storage": [
            "Someone keeps parking in my assigned spot.",
            "My storage locker was broken into.",
            "Issues arise when you're dealing with limited space."
        ],
        "Smoke & vapour": [
            "Cannabis smoke is entering my unit from the hallway.",
            "The smell of tobacco is coming through the vents.",
            "When tobacco, cannabis or other vapours seep into other units."
        ],
        "Odours": [
            "The hallway constantly smells of strong spices.",
            "There's a sewage-like odour near my unit.",
            "Unpleasant smells can be a nuisance for other occupants."
        ],
        "Light": [
            "Bright security lights are shining into my bedroom.",
            "Common area lighting is flickering and disruptive.",
            "Too bright? Too dim? There may be a simple solution."
        ],
        "Vehicles": [
            "Residents are speeding in the underground garage.",
            "Cars are idling too long near building entrances.",
            "From idling cars to dangerous driving, see what rules apply."
        ],
        "Settlement Agreements": [
            "The other party isn't complying with our CAT settlement.",
            "We reached an agreement but it's being ignored.",
            "Steps to take when someone doesn't comply with your Condo Tribunal settlement agreement."
        ],
        "Harassment": [
            "A neighbour is making discriminatory remarks toward me.",
            "I feel threatened by repeated interactions with another resident.",
            "Unwelcome, threatening or discriminatory interactions with other residents."
        ],
        "Infestation": [
            "There are cockroaches in the stairwell.",
            "My unit has mice and the board won't help.",
            "From mice and mould to birds and bedbugs, infestation is a serious issue."
        ],
        "Short-term rentals": [
            "My neighbour is running an Airbnb in their unit.",
            "There are constant strangers entering and leaving the unit next door.",
            "What to do when short-term rentals don't follow the condo rules."
        ],
        "Meetings": [
            "The AGM was cancelled without notice.",
            "Meeting minutes were never distributed.",
            "Meetings are an important way for condo communities to come together transparently."
        ],
        "Other": [
            "My issue doesn't fall into any of the standard categories.",
            "This problem doesn't seem to be listed anywhere.",
            "Any issue that does not fit the predefined categories."
        ]
    }
    
    topic_embeddings = {
        topic: model.encode(anchors, convert_to_tensor=True).mean(dim=0)
        for topic, anchors in topic_anchors.items()
    }
    
    embedding = model.encode(text, convert_to_tensor=True)
    best_score = -1
    best_topic = "Other"

    for topic, anchor_embedding in topic_embeddings.items():
        score = util.cos_sim(embedding, anchor_embedding).item()
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic if best_score >= threshold else "Other"

# Article Processing Function
def process_articles():
    """Process only articles scraped today (based on scraped_date)"""
    raw_collection = get_collection(RAW_COLLECTION)
    processed_collection = get_collection(PROCESSED_COLLECTION)

    today_str = datetime.utcnow().date().isoformat()
    query = {
        "processing_status": "pending",
        "scraped_date": {"$regex": f"^{today_str}"}
    }

    logging.info(f"Processing articles scraped today: {today_str}")

    pending_articles = list(raw_collection.find(query))

    if not pending_articles:
        logging.info("No pending articles found for today.")
        return 0

    processed_count = 0

    for article in pending_articles:
        try:
            title = clean_text(article.get("title", ""))
            content = clean_text(article.get("content", ""))
            text_to_analyze = f"{title} {content}"

            # Check for duplicate in raw collection (excluding self)
            if raw_collection.find_one({
                "link": article.get("link"),
                "_id": {"$ne": article["_id"]}
            }):
                logging.info(f"Duplicate found in raw collection: {article.get('link')}")
                raw_collection.update_one(
                    {"_id": article["_id"]},
                    {"$set": {"processing_status": "duplicate"}}
                )
                continue

            # Check for duplicate in processed collection
            if processed_collection.find_one({"link": article.get("link")}):
                logging.info(f"Skipping duplicate in processed: {article.get('link')}")
                raw_collection.update_one(
                    {"_id": article["_id"]},
                    {"$set": {"processing_status": "duplicate"}}
                )
                continue

            sentiment = analyze_sentiment(text_to_analyze)
            topic = assign_topic(text_to_analyze)

            base_doc = {
                "title": title,
                "link": article.get("link", ""),
                "published_date": article.get("published_date", ""),
                "content": content,
                "tags": article.get("tags", []),
                "source": "RSS Feeds" if article.get("source", "").strip().lower() == "rss" else article.get("source", ""),
                "subreddit": article.get("subreddit", None),
                "upvotes": article.get("upvotes", None),
                "comments": article.get("comments", None),
                "scraped_date": article.get("scraped_date", datetime.utcnow().isoformat()),
                "assigned_issue": topic,
                "sentiment_analysis": [sentiment],
                "original_id": article["_id"]
            }

            processed_collection.insert_one(base_doc)

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
        msg = MIMEMultipart()
        msg["From"] = EMAIL_CONFIG["sender_email"]
        msg["To"] = EMAIL_CONFIG["receiver_email"]
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
            server.sendmail(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["receiver_email"], msg.as_string())
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
        configure_logging()
        validate_db_connection()

        logging.info("Starting GitHub-scheduled article processing job...")
        processed_count = process_articles()

        if processed_count > 0:
            send_success_email(processed_count)
        else:
            logging.info("No articles to process today.")
            send_email(
                subject="No Articles Processed Today",
                body="The processor ran successfully, but no new articles were found to process today."
            )
            
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        send_error_email(str(e))
        sys.exit(1)
