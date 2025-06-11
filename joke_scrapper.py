import praw
import os

# --- Reddit API credentials (from sillybillyscraper app) ---
REDDIT_CLIENT_ID = "SuiwSK8nJugIlSUaM3PX3w"
REDDIT_CLIENT_SECRET = "Ivvg3TxXhurH29ot1hgwzoq0VPyQ6Q"
REDDIT_USER_AGENT = "sillybillyjokegenerator/0.1 by u/YOUR_REDDIT_USERNAME"

# --- Connect to Reddit (no login needed for read-only scraping) ---
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

# --- Scrape jokes from a subreddit ---
def get_jokes(subreddit_name, limit=300):
    jokes = []
    print(f"🔍 Scraping r/{subreddit_name}...")
    for submission in reddit.subreddit(subreddit_name).hot(limit=limit):
        if not submission.stickied and submission.selftext:
            joke = submission.title.strip() + " " + submission.selftext.strip()
            joke = joke.replace("\n", " ").replace("\r", "").strip()
            if len(joke) > 20:
                jokes.append(joke)
    return jokes

# --- Subreddits to scrape ---
subreddits = ["cleanjokes", "3amjokes", "dadjokes"]
all_jokes = []

for sub in subreddits:
    all_jokes.extend(get_jokes(sub))

# --- Save jokes to file ---
os.makedirs("data", exist_ok=True)
output_path = "data/reddit_jokes.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for joke in all_jokes:
        f.write(joke + "\n")

print(f"✅ Scraped {len(all_jokes)} jokes total.")
print(f"📁 Saved to: {output_path}")
