import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import json
import logging
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def enhance_emotion_detection(text, top_emotion, confidence, all_emotions=None):
    """
    A comprehensive approach to enhance emotion detection for any user input.
    
    Args:
        text: The user's input text
        top_emotion: The initially detected emotion
        confidence: The confidence score for the detected emotion
        all_emotions: Optional list of (emotion, score) tuples for all emotions
        
    Returns:
        Tuple of (enhanced_emotion, enhanced_confidence)
    """
    # Convert text to lowercase for analysis
    text_lower = text.lower()
    
    # Define emotion categories for better classification
    positive_emotions = [
        "joy", "admiration", "amusement", "excitement", "gratitude", 
        "love", "optimism", "pride", "approval", "caring", "desire", "relief"
    ]
    
    negative_emotions = [
        "sadness", "anger", "annoyance", "disapproval", "disgust", 
        "embarrassment", "fear", "grief", "nervousness", "disappointment", 
        "remorse", "confusion"
    ]
    
    neutral_emotions = ["surprise", "realization", "curiosity", "neutral"]
    
    # Step 1: Analyze linguistic context more deeply
    # Look for emotional indicators in linguistic patterns
    
    # Sentiment modifier words that intensify emotions
    intensifiers = ["very", "really", "extremely", "incredibly", "so", "too", "absolutely", "completely"]
    
    # Words indicating emotional states
    emotional_state_words = {
        # Positive states
        "happy": "joy",
        "joyful": "joy",
        "excited": "excitement",
        "thrilled": "excitement",
        "glad": "joy",
        "pleased": "joy",
        "grateful": "gratitude",
        "thankful": "gratitude",
        "proud": "pride",
        "confident": "pride",
        "hopeful": "optimism",
        "optimistic": "optimism",
        "content": "joy",
        "satisfied": "joy",
        "loving": "love",
        "amused": "amusement",
        
        # Negative states
        "sad": "sadness",
        "unhappy": "sadness",
        "depressed": "sadness",
        "miserable": "sadness",
        "heartbroken": "grief",
        "grieving": "grief",
        "disappointed": "disappointment",
        "let down": "disappointment",
        "angry": "anger",
        "furious": "anger",
        "mad": "anger",
        "irritated": "annoyance",
        "annoyed": "annoyance",
        "frustrated": "annoyance",
        "disgusted": "disgust",
        "repulsed": "disgust",
        "afraid": "fear",
        "scared": "fear",
        "terrified": "fear",
        "anxious": "fear",
        "nervous": "nervousness",
        "worried": "fear",
        "ashamed": "embarrassment",
        "embarrassed": "embarrassment",
        "humiliated": "embarrassment",
        "guilty": "remorse",
        "regretful": "remorse",
        "confused": "confusion",
        "puzzled": "confusion",
        "bewildered": "confusion",
        
        # Common phrases indicating emotional states
        "feeling down": "sadness",
        "feeling blue": "sadness",
        "feeling low": "sadness",
        "in a bad mood": "sadness",
        "fed up": "annoyance",
        "had enough": "annoyance",
        "can't stand": "anger",
        "hate": "anger",
        "love": "love",
        "adore": "love",
        "appreciate": "gratitude"
    }
    
    # Personal expression patterns
    personal_expressions = [
        # Positive expressions
        (r"i (?:am|feel|'m) (?:so |really |very |extremely )?(happy|glad|excited|thrilled|pleased|grateful|thankful|proud|confident|hopeful|content)", "positive"),
        (r"i (?:am|feel|'m) (?:so |really |very |extremely )?(satisfied|loving|amused|optimistic)", "positive"),
        (r"i love", "positive"),
        (r"makes me happy", "positive"),
        (r"this is (?:so |really |very |extremely )?(good|great|awesome|amazing|excellent|wonderful|fantastic)", "positive"),
        
        # Negative expressions
        (r"i (?:am|feel|'m) (?:so |really |very |extremely )?(sad|unhappy|depressed|miserable|disappointed|angry|furious|mad|irritated)", "negative"),
        (r"i (?:am|feel|'m) (?:so |really |very |extremely )?(annoyed|frustrated|disgusted|repulsed|afraid|scared|terrified|anxious|nervous)", "negative"),
        (r"i (?:am|feel|'m) (?:so |really |very |extremely )?(worried|ashamed|embarrassed|humiliated|guilty|regretful|confused|puzzled)", "negative"),
        (r"i hate", "negative"),
        (r"makes me (?:sad|angry|upset|frustrated)", "negative"),
        (r"this is (?:so |really |very |extremely )?(bad|terrible|awful|horrible|dreadful|unbearable)", "negative")
    ]
    
    # Step 2: Analyze explicit emotion statements
    # Check if user explicitly states their emotion
    for state_word, emotion in emotional_state_words.items():
        pattern = r'\b' + re.escape(state_word) + r'\b'
        if re.search(pattern, text_lower):
            # If explicit emotion statement found, increase confidence
            # Check if intensifiers are present to boost confidence further
            intensified = any(intensifier in text_lower.split() for intensifier in intensifiers)
            new_confidence = min(confidence + 0.25 + (0.1 if intensified else 0), 0.95)
            return emotion, new_confidence
    
    # Step 3: Check for personal expression patterns
    for pattern, sentiment in personal_expressions:
        if re.search(pattern, text_lower):
            # If there's a personal expression, align with the sentiment
            if sentiment == "positive" and top_emotion not in positive_emotions:
                # Find the highest scoring positive emotion in all_emotions if available
                if all_emotions:
                    positive_scores = [(e, s) for e, s in all_emotions if e in positive_emotions]
                    if positive_scores:
                        best_positive = max(positive_scores, key=lambda x: x[1])
                        return best_positive[0], max(confidence, best_positive[1] + 0.15)
                # Default to joy if no other positive emotions available
                return "joy", max(confidence, 0.75)
            elif sentiment == "negative" and top_emotion not in negative_emotions:
                # Find the highest scoring negative emotion in all_emotions if available
                if all_emotions:
                    negative_scores = [(e, s) for e, s in all_emotions if e in negative_emotions]
                    if negative_scores:
                        best_negative = max(negative_scores, key=lambda x: x[1])
                        return best_negative[0], max(confidence, best_negative[1] + 0.15)
                # Based on text content, try to determine a specific negative emotion
                if "boss" in text_lower or "work" in text_lower or "job" in text_lower:
                    return "disappointment", max(confidence, 0.75)
                elif "angry" in text_lower or "mad" in text_lower or "hate" in text_lower:
                    return "anger", max(confidence, 0.75)
                elif "sad" in text_lower or "unhappy" in text_lower:
                    return "sadness", max(confidence, 0.75)
                elif "afraid" in text_lower or "scared" in text_lower or "worry" in text_lower:
                    return "fear", max(confidence, 0.75)
                # Default to sadness if no specific negative emotion can be determined
                return "sadness", max(confidence, 0.75)
    
    # Step 4: Adjust neutral predictions with low confidence
    # If the model predicts neutral with low confidence, try to find a better emotion
    if top_emotion in neutral_emotions and confidence < 0.7:
        # Check for emotional content indicators
        emotional_words = [word for word in emotional_state_words.keys() 
                          if any(w in text_lower.split() for w in word.split())]
        
        if emotional_words:
            # Get the most likely emotion based on emotional words
            emotions = [emotional_state_words[word] for word in emotional_words]
            most_common_emotion = max(set(emotions), key=emotions.count)
            return most_common_emotion, max(confidence + 0.15, 0.7)
        
        # Check for workplace content
        if any(word in text_lower for word in ["boss", "work", "job", "manager", "coworker", "fired"]):
            # Check tone for workplace content
            if any(neg in text_lower for neg in ["hate", "tired", "sick", "bad", "low", "difficult"]):
                return "disappointment", max(confidence + 0.15, 0.7)
    
    # Step 5: Semantic context analysis
    # Analyze context for specific domains and adjust accordingly
    
    # Family context
    family_terms = ["family", "mom", "dad", "parent", "child", "son", "daughter", "husband", "wife", "marriage"]
    if any(term in text_lower for term in family_terms):
        if top_emotion == "neutral" and confidence < 0.75:
            # Check for positive/negative indicators
            if any(pos in text_lower for pos in ["love", "happy", "glad", "joy"]):
                return "love", max(confidence + 0.15, 0.7)
            elif any(neg in text_lower for neg in ["argument", "fight", "conflict", "upset"]):
                return "sadness", max(confidence + 0.15, 0.7)
    
    # Health context
    health_terms = ["health", "sick", "illness", "disease", "pain", "hospital", "doctor", "diagnosis", "treatment"]
    if any(term in text_lower for term in health_terms):
        if top_emotion == "neutral" and confidence < 0.75:
            # Most health concerns indicate fear or anxiety
            return "fear", max(confidence + 0.15, 0.7)
    
    # Step 6: If no adjustments were necessary, return the original prediction
    return top_emotion, confidence

class EmotionDetector:
    def __init__(self, model_path=None, device=None):
        """
        Initialize the emotion detector with a pre-trained or fine-tuned model
        
        Args:
            model_path (str, optional): Path to the fine-tuned model directory
            device (str, optional): Device to run the model on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading fine-tuned model from {model_path}")
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            
            # Load label mapping if available
            label_map_path = os.path.join(model_path, 'label_map.json')
            if os.path.exists(label_map_path):
                with open(label_map_path, 'r') as f:
                    self.label_map = json.load(f)
                logger.info(f"Loaded label mapping with {len(self.label_map)} labels")
                # Invert the mapping for prediction
                self.idx_to_label = {v: k for k, v in self.label_map.items()}
            else:
                # Use default GoEmotions labels if no mapping is found
                self.idx_to_label = None
        else:
            if model_path:
                logger.warning(f"Model path {model_path} not found. Falling back to pre-trained model.")
            else:
                logger.info("No model path provided. Using pre-trained model.")
            
            # Fall back to pre-trained model
            self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.idx_to_label = None
        
        # GoEmotions dataset has 28 emotion labels (including neutral)
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
            "confusion", "curiosity", "desire", "disappointment", "disapproval", 
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
            "joy", "love", "nervousness", "optimism", "pride", "realization", 
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Emotion detector initialized successfully")
    
    def predict(self, text):
        """
        Predict the emotions in the input text with enhanced accuracy
        
        Args:
            text (str): Input text to analyze
                
        Returns:
            tuple: (emotion, confidence) where emotion is the predicted emotion label
                and confidence is the model's confidence score (0-1)
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            
        # Get top emotion and confidence
        top_emotion_idx = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][top_emotion_idx].item()
        
        # Get all emotions and their scores for more comprehensive analysis
        all_emotions = []
        for i in range(len(predictions[0])):
            idx = i
            score = predictions[0][idx].item()
            
            # Map index to emotion label
            if self.idx_to_label:
                emotion = self.idx_to_label[idx]
            else:
                if idx < len(self.emotion_labels):
                    emotion = self.emotion_labels[idx]
                else:
                    continue
            
            all_emotions.append((emotion, score))
        
        # Sort emotions by confidence score
        all_emotions.sort(key=lambda x: x[1], reverse=True)
        
        # Map top index to emotion label
        if self.idx_to_label:
            top_emotion = self.idx_to_label[top_emotion_idx]
        else:
            if top_emotion_idx < len(self.emotion_labels):
                top_emotion = self.emotion_labels[top_emotion_idx]
            else:
                top_emotion = "unknown"
                logger.warning(f"Predicted index {top_emotion_idx} out of bounds for emotion_labels")
        
        # Log the original prediction
        logger.info(f"Original prediction: {top_emotion} with confidence {confidence:.4f}")
        
        # Enhance emotion detection
        enhanced_emotion, enhanced_confidence = enhance_emotion_detection(
            text, top_emotion, confidence, all_emotions
        )
        
        # Log the enhanced prediction
        logger.info(f"Enhanced prediction: {enhanced_emotion} with confidence {enhanced_confidence:.4f}")
        
        return enhanced_emotion, enhanced_confidence
    
    def get_all_emotions(self, text, top_k=3):
        """
        Get the top k emotions and their confidence scores for the input text
        
        Args:
            text (str): Input text to analyze
            top_k (int): Number of top emotions to return
            
        Returns:
            list: List of (emotion, confidence) tuples for the top k emotions
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
        
        # Get top k emotions
        top_k_values, top_k_indices = torch.topk(predictions, k=min(top_k, predictions.size(1)))
        
        # Convert to list of (emotion, confidence) tuples
        top_emotions = []
        for i in range(top_k_values.size(1)):
            idx = top_k_indices[0][i].item()
            confidence = top_k_values[0][i].item()
            
            # Map index to emotion label
            if self.idx_to_label:
                # Use custom label mapping if available
                emotion = self.idx_to_label[idx]
            else:
                # Use default GoEmotions labels
                if idx < len(self.emotion_labels):
                    emotion = self.emotion_labels[idx]
                else:
                    emotion = "unknown"
            
            top_emotions.append((emotion, confidence))
        
        return top_emotions


# Example usage
if __name__ == "__main__":
    # Example text
    text = "I'm feeling really happy today after receiving good news!"
    
    # Initialize detector
    detector = EmotionDetector()
    
    # Get prediction
    emotion, confidence = detector.predict(text)
    print(f"Detected emotion: {emotion} (confidence: {confidence:.4f})")
    
    # Get top 3 emotions
    top_emotions = detector.get_all_emotions(text, top_k=3)
    print("Top emotions:")
    for emotion, confidence in top_emotions:
        print(f"  {emotion}: {confidence:.4f}")





# import torch
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# import os

# class EmotionDetector:
#     def __init__(self, model_path=None):
#         # Default to pretrained model if no fine-tuned model path is provided
#         if model_path and os.path.exists(model_path):
#             self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
#             self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
#         else:
#             # Fall back to pretrained model
#             self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
#             self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
#         # GoEmotions dataset has 27 emotion labels, plus neutral
#         self.emotion_labels = [
#             "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
#             "confusion", "curiosity", "desire", "disappointment", "disapproval", 
#             "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
#             "joy", "love", "nervousness", "optimism", "pride", "realization", 
#             "relief", "remorse", "sadness", "surprise", "neutral"
#         ]
        
#         # Move model to GPU if available
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.model.eval()
    
#     def predict(self, text):
#         """
#         Predict the emotions in the input text
#         """
#         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             predictions = torch.softmax(outputs.logits, dim=1)
            
#         # Get top emotion
#         top_emotion_idx = torch.argmax(predictions, dim=1).item()
#         top_emotion = self.emotion_labels[top_emotion_idx]
#         confidence = predictions[0][top_emotion_idx].item()
        
#         return top_emotion, confidence



# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
# import torch
# import torch.nn.functional as F

# model = DistilBertForSequenceClassification.from_pretrained('./emotion_detector/model')
# tokenizer = DistilBertTokenizerFast.from_pretrained('./emotion_detector/model')

# EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
#             "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
#             "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise"]

# def predict_emotion(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     probs = F.softmax(logits, dim=-1)
#     top_idx = torch.argmax(probs, dim=-1).item()
#     return EMOTIONS[top_idx], probs[0][top_idx].item()

# if __name__ == "__main__":
#     text = "I feel like nothing is going right today."
#     emotion, confidence = predict_emotion(text)
#     print(f"Detected emotion: {emotion} (Confidence: {confidence:.2f})")
