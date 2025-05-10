# utils/content_analyzer.py
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentAnalyzer:
    def __init__(self):
        # Download NLTK resources if not already available
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
        
        # Fix: Use string 'english' instead of set of stopwords
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def extract_keywords(self, text, top_n=5):
        """Extract the most important keywords from text"""
        try:
            # Vectorize the text
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get the TF-IDF scores
            dense = tfidf_matrix.todense()
            scores = dense.tolist()[0]
            
            # Create word-score pairs and sort
            word_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            return word_scores[:top_n]
        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}")
            return []
    
    def analyze_topics(self, text):
        """Identify potential topics in the text"""
        topic_indicators = {
            "work": {'job', 'work', 'boss', 'career', 'office', 'manager', 'colleague', 'coworker', 'promotion', 'salary', 'raise'},
            "relationships": {'friend', 'partner', 'spouse', 'wife', 'husband', 'boyfriend', 'girlfriend', 'relationship', 'family'},
            "health": {'health', 'sick', 'illness', 'doctor', 'hospital', 'pain', 'symptom', 'disease', 'medication'},
            "mental_health": {'anxiety', 'depression', 'stress', 'therapy', 'mental', 'emotional', 'feeling', 'mood'},
            "future": {'future', 'plan', 'goal', 'dream', 'aspiration', 'hope', 'expect', 'anticipate'},
            "achievement": {'promotion', 'achieved', 'success', 'win', 'award', 'recognition', 'praise', 'compliment', 'bonus', 'achievement'}
        }
        
        words = set(text.lower().split())
        detected_topics = []
        
        for topic, indicators in topic_indicators.items():
            if any(word in indicators for word in words):
                detected_topics.append(topic)
        
        return detected_topics