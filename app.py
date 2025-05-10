import os
import time
import json
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Import our custom modules
from emotion_detector.predict import EmotionDetector
from response_generator.generate import ResponseGenerator
from utils.emotion_tracker import EmotionTracker
from utils.content_analyzer import ContentAnalyzer
from utils.ethical_safeguards import EthicalSafeguards

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🧠 AI Mental Health Journal",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem !important;
        color: #4F8BF9;
        margin-bottom: 0.5rem;
    }
    .emotion-label {
        font-weight: bold;
        color: #4CAF50;
    }
    .therapist-header {
        font-weight: bold;
        color: #3F51B5;
    }
    .user-message {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .therapist-message {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .crisis-message {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #EF5350;
        margin-bottom: 1rem;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'emotion_tracker' not in st.session_state:
    st.session_state.emotion_tracker = EmotionTracker()
if 'content_analyzer' not in st.session_state:
    st.session_state.content_analyzer = ContentAnalyzer()
if 'ethical_safeguards' not in st.session_state:
    st.session_state.ethical_safeguards = EthicalSafeguards()

# Function to load models with caching
@st.cache_resource
def load_models():
    """Load the emotion detector and response generator models"""
    # Path to the fine-tuned model
    model_path = "models/emotion_detector"
    
    # Use the default model if fine-tuned model doesn't exist
    if not os.path.exists(model_path):
        st.warning("⚠️ Fine-tuned model not found. Using default pre-trained model.")
        model_path = None
    
    # Initialize the emotion detector
    emotion_detector = EmotionDetector(model_path=model_path)
    
    # Get API key from environment or Streamlit secrets
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    
    # Initialize the response generator
    response_generator = ResponseGenerator(api_key=api_key)
    
    return emotion_detector, response_generator

# Function to load templates
@st.cache_resource
def load_templates():
    """Load prompt templates for display in the sidebar"""
    template_path = "response_generator/prompt_templates.json"
    try:
        with open(template_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Function to enhance keywords based on identified emotional context
def enhance_keywords(keywords, user_input, emotion):
    """Add additional keywords based on emotional context"""
    text_lower = user_input.lower()
    
    # Add emotional keywords if not already present
    if emotion not in [k[0] for k in keywords]:
        keywords.append((emotion, 0.9))
    
    # Add work-related keywords for workplace content
    work_terms = ["boss", "job", "work", "manager", "office", "workplace", "colleague", "coworker"]
    if any(term in text_lower for term in work_terms):
        for term in work_terms:
            if term in text_lower and not any(k[0] == term for k in keywords):
                keywords.append((term, 0.8))
    
    # Add family-related keywords
    family_terms = ["family", "mom", "dad", "parent", "child", "son", "daughter", "husband", "wife"]
    if any(term in text_lower for term in family_terms):
        for term in family_terms:
            if term in text_lower and not any(k[0] == term for k in keywords):
                keywords.append((term, 0.8))
    
    # Add health-related keywords
    health_terms = ["health", "sick", "illness", "pain", "doctor", "hospital"]
    if any(term in text_lower for term in health_terms):
        for term in health_terms:
            if term in text_lower and not any(k[0] == term for k in keywords):
                keywords.append((term, 0.8))
                
    # Add emotional state keywords
    emotion_state_terms = {
        "happy": "joy", "sad": "sadness", "angry": "anger", "afraid": "fear",
        "anxious": "fear", "worried": "fear", "stressed": "anxiety",
        "frustrated": "anger", "disappointed": "disappointment"
    }
    
    for term, related_emotion in emotion_state_terms.items():
        if term in text_lower and not any(k[0] == term for k in keywords):
            keywords.append((term, 0.75))
            
    return keywords

# Main app layout
def main():
    """Main app function that defines the UI and app flow."""
    # Display title and description
    st.markdown('<p class="main-title">🧠 AI Mental Health Journal</p>', unsafe_allow_html=True)
    st.markdown("""
    Share how you're feeling today, and the AI will detect your emotions
    and respond with empathetic, therapeutic insights.
    """)
    
    # Load models
    emotion_detector, response_generator = load_models()
    
    # User input
    user_input = st.text_area("How are you feeling today?", height=150, 
                             placeholder="Type your thoughts here...")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        submit_button = st.button("Analyze and Respond")
    with col2:
        clear_history = st.button("Clear History")
    with col3:
        show_insights = st.button("Show Insights")
        
    # Clear history if requested
    if clear_history:
        st.session_state.messages = []
        st.session_state.emotion_tracker = EmotionTracker()
        st.experimental_rerun()
    
    # Handle the submit button
    if submit_button:
        if not user_input.strip():
            st.warning("Please enter some text to proceed.")
        else:
            # Check for crisis language
            if st.session_state.ethical_safeguards.check_for_crisis(user_input):
                crisis_response = st.session_state.ethical_safeguards.get_crisis_response()
                st.markdown(f"<div class='crisis-message'>{crisis_response}</div>", 
                           unsafe_allow_html=True)
            
            # Show a spinner while processing
            with st.spinner("Analyzing your entry..."):
                # Extract keywords and topics
                keywords = st.session_state.content_analyzer.extract_keywords(user_input)
                topics = st.session_state.content_analyzer.analyze_topics(user_input)
                
                # Get emotion predictions using enhanced emotion detection
                # The enhancement is now handled directly in the EmotionDetector class
                emotion, confidence = emotion_detector.predict(user_input)
                
                # Get all possible emotions for this input
                top_emotions = emotion_detector.get_all_emotions(user_input, top_k=5)
                
                # Enhance keywords with emotional context
                keywords = enhance_keywords(keywords, user_input, emotion)
                
                # Slight delay for better UX
                time.sleep(0.5)
                
                # Get the conversation history to provide context for the response
                conversation_history = st.session_state.messages.copy() if st.session_state.messages else []
                
                try:
                    # Generate response with conversation history for context-aware responses
                    # Using positional arguments to avoid keyword argument issues
                    response = response_generator.generate_response(user_input, emotion, conversation_history)
                    
                    # Error checking - if response contains instruction text from T5, fix it
                    if ", a kind AI therapist. Patient says:" in response:
                        # Extract just the response part
                        if "Reply in 2-3 sentences with empathy and an open question." in response:
                            response_parts = response.split("Reply in 2-3 sentences with empathy and an open question.")
                            if len(response_parts) > 1:
                                response = response_parts[1].strip()
                        # Fallback repair for other instruction patterns
                        elif ". Reply in" in response:
                            response_parts = response.split(". Reply in")
                            if len(response_parts) > 0:
                                response = response_parts[0].strip()
                    
                    # Remove any leading punctuation
                    response = response.lstrip(',.;:-— ')
                    
                    # If response is still problematic, use a simple default
                    if not response or len(response) < 10:
                        if emotion in response_generator.negative_emotions:
                            response = f"I can hear that you're feeling {emotion} today. That's completely understandable given what you're going through. Would you like to share more about what's been happening?"
                        else:
                            response = f"Thank you for sharing how you're feeling. I'd like to understand more about your experience. Could you tell me more about what's on your mind?"
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    response = f"I can hear that you're experiencing {emotion}. Would you like to share more about what's going on for you?"
                
                # Add to emotion tracker
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.emotion_tracker.add_entry(
                    user_input, emotion, confidence, timestamp
                )
                
                # Add to conversation history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": user_input, 
                    "timestamp": timestamp
                })
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "emotion": emotion,
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "keywords": keywords,
                    "topics": topics,
                    "all_emotions": top_emotions  # Store all detected emotions
                })
                
                # Display detected emotion with confidence
                st.markdown(f"<p class='emotion-label'>Detected Emotion: {emotion.title()} ({confidence:.2f})</p>", 
                           unsafe_allow_html=True)
                
                # Display other top emotions
                if top_emotions and len(top_emotions) > 1:
                    other_emotions = [(e, c) for e, c in top_emotions if e != emotion][:3]
                    if other_emotions:
                        other_emotions_str = ", ".join([f"{e.title()} ({c:.2f})" for e, c in other_emotions])
                        st.caption(f"**Other possible emotions**: {other_emotions_str}")
                
                # Display keywords and topics if available
                if keywords:
                    keyword_str = ", ".join([f"{k[0]} ({k[1]:.2f})" for k in keywords])
                    st.caption(f"**Keywords**: {keyword_str}")
                
                if topics:
                    st.caption(f"**Topics**: {', '.join(topics)}")
                
                # Display the AI therapist response
                st.markdown(f"<p class='therapist-header'>AI Therapist Says:</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='therapist-message'>{response}</div>", unsafe_allow_html=True)
    
    # Display conversation history if any
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### Conversation History")
        
        for i in range(0, len(st.session_state.messages), 2):
            if i + 1 < len(st.session_state.messages):
                user_msg = st.session_state.messages[i]
                ai_msg = st.session_state.messages[i + 1]
                
                # Format the timestamp
                timestamp = user_msg.get("timestamp", "")
                if timestamp:
                    st.caption(f"**{timestamp}**")
                
                # Display user message
                st.markdown(f"<div class='user-message'>{user_msg['content']}</div>", 
                           unsafe_allow_html=True)
                
                # Display AI response with emotion
                emotion = ai_msg.get('emotion', 'unknown')
                confidence = ai_msg.get('confidence', 0.0)
                st.markdown(f"<p class='emotion-label'>Detected: {emotion.title()} ({confidence:.2f})</p>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div class='therapist-message'>{ai_msg['content']}</div>", 
                           unsafe_allow_html=True)
                
                st.markdown("---")
    
    # Show insights if requested
    if show_insights and len(st.session_state.messages) >= 4:
        st.markdown("### Your Emotional Journey")
        
        # Get emotion counts and chart
        emotion_counts = st.session_state.emotion_tracker.get_emotion_counts()
        emotion_chart = st.session_state.emotion_tracker.generate_emotion_chart()
        
        # Display emotion counts
        st.bar_chart(emotion_counts)
        
        # Display emotion chart if available
        if emotion_chart:
            st.image(f"data:image/png;base64,{emotion_chart}", use_column_width=True)
        
        # Show most common emotion
        most_common = st.session_state.emotion_tracker.get_most_common_emotion()
        if most_common:
            st.info(f"Your most frequently detected emotion is: **{most_common.title()}**")
        
        import pandas as pd
        
        # Show keyword analysis
        all_keywords = []
        for msg in st.session_state.messages:
            if msg.get("role") == "assistant" and "keywords" in msg:
                all_keywords.extend([k[0] for k in msg["keywords"]])
        
        if all_keywords:
            keyword_counts = dict(pd.Series(all_keywords).value_counts())
            st.markdown("### Common Themes in Your Journal")
            st.bar_chart(keyword_counts)
            
        # Show emotional context analysis
        st.markdown("### Emotional Context Analysis")
        st.markdown("""
        The AI analyzes not just your primary emotion, but also contextual factors like:
        
        - **Linguistic patterns**: How you express yourself
        - **Emotional intensity**: How strongly you feel
        - **Contextual domains**: Work, family, health, etc.
        - **Emotional trajectory**: How your emotions change over time
        
        This helps provide more personalized responses tailored to your specific situation.
        """)

# Sidebar content
def sidebar():
    """Defines the sidebar with additional information."""
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This AI Mental Health Journal uses:
    
    - **Enhanced Emotion Detection**: Advanced contextual analysis of emotions
    - **Response Generation**: AI-generated therapeutic responses
    - **Content Analysis**: Keyword extraction and topic detection
    - **Emotional Tracking**: Visualization of your emotional journey
    - **Contextual Memory**: Responses evolve based on conversation history
    
    The app can detect 28 different emotions and provides supportive responses that mimic a real therapist.
    """)
    
    st.sidebar.header("How It Works")
    st.sidebar.markdown("""
    1. **Type your thoughts** in the text area
    2. Click **Analyze and Respond**
    3. The AI will **detect your emotions** using contextual analysis
    4. You'll receive a **personalized therapeutic response**
    5. Your conversation history is saved and used to create a more personal therapeutic experience
    6. Click **Show Insights** to see analysis of your emotional patterns
    """)
    
    st.sidebar.header("Enhanced Emotion Detection")
    st.sidebar.markdown("""
    Our system uses sophisticated analysis to detect emotions:
    
    - **Linguistic Analysis**: Understanding emotional expressions
    - **Context Recognition**: Work, family, health contexts
    - **Personal Expressions**: How you uniquely express feelings
    - **Emotional Intensity**: Detecting subtle vs. strong emotions
    
    This creates a more accurate understanding of your emotional state.
    """)
    
    st.sidebar.header("Therapeutic Approaches")
    st.sidebar.markdown("""
    The AI therapist uses several evidence-based therapeutic techniques:
    
    - **Validation**: Acknowledging feelings as valid and understandable
    - **Reflection**: Mirroring back what you've shared to demonstrate understanding
    - **Open-ended Questions**: Encouraging deeper exploration
    - **Perspective Offering**: Providing gentle insights without being prescriptive
    - **Continuity**: Referencing past conversations to create a cohesive therapeutic journey
    """)
    
    st.sidebar.header("Emotion Categories")
    with st.sidebar.expander("View All Emotions"):
        emotions = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
            "confusion", "curiosity", "desire", "disappointment", "disapproval", 
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
            "joy", "love", "nervousness", "optimism", "pride", "realization", 
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        st.markdown("- " + "\n- ".join(emotions).title())
    
    # Display template examples in the sidebar
    templates = load_templates()
    if templates:
        st.sidebar.header("Response Templates")
        template_category = st.sidebar.selectbox(
            "View templates for emotion:",
            options=list(templates.keys()),
            index=0
        )
        if template_category in templates:
            with st.sidebar.expander("View Templates"):
                for i, template in enumerate(templates[template_category]):
                    st.markdown(f"**Template {i+1}:**\n\n{template}")
    
    st.sidebar.markdown("""
    ---
    **Note**: This is not a replacement for professional mental health support.
    If you're in crisis, please contact a mental health professional or crisis hotline.
    """)

# Run the app
if __name__ == "__main__":
    sidebar()
    main()



# # app.py
# import os
# import time
# import json
# import streamlit as st
# from datetime import datetime
# from dotenv import load_dotenv

# # Import our custom modules
# from emotion_detector.predict import EmotionDetector
# from response_generator.generate import ResponseGenerator

# # Load environment variables
# load_dotenv()

# # Page configuration
# st.set_page_config(
#     page_title="🧠 AI Mental Health Journal",
#     page_icon="🧠",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

# # Add custom CSS
# st.markdown("""
# <style>
#     .main-title {
#         font-size: 2.5rem !important;
#         color: #4F8BF9;
#         margin-bottom: 0.5rem;
#     }
#     .emotion-label {
#         font-weight: bold;
#         color: #4CAF50;
#     }
#     .therapist-header {
#         font-weight: bold;
#         color: #3F51B5;
#     }
#     .user-message {
#         background-color: #E3F2FD;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 0.5rem;
#     }
#     .therapist-message {
#         background-color: #F1F8E9;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#     }
#     hr {
#         margin-top: 2rem;
#         margin-bottom: 2rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state for conversation history
# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# # Function to load models with caching
# @st.cache_resource
# def load_models():
#     """Load the emotion detector and response generator models"""
#     # Path to the fine-tuned model
#     model_path = "models/emotion_detector"
    
#     # Use the default model if fine-tuned model doesn't exist
#     if not os.path.exists(model_path):
#         st.warning("⚠️ Fine-tuned model not found. Using default pre-trained model.")
#         model_path = None
    
#     # Initialize the emotion detector
#     emotion_detector = EmotionDetector(model_path=model_path)
    
#     # API key is not used due to rate limits, so we pass None
#     response_generator = ResponseGenerator(api_key=None)
    
#     return emotion_detector, response_generator

# # Function to load templates
# @st.cache_resource
# def load_templates():
#     """Load prompt templates for display in sidebar"""
#     template_path = "response_generator/prompt_templates.json"
#     try:
#         with open(template_path, "r") as f:
#             return json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError):
#         return {}

# # Main app layout
# def main():
#     """Main app function"""
#     # Display title and description
#     st.markdown('<p class="main-title">🧠 AI Mental Health Journal</p>', unsafe_allow_html=True)
#     st.markdown("""
#     Share how you're feeling today, and the AI will detect your emotions
#     and respond with empathetic, therapeutic insights.
#     """)
    
#     # Load models
#     emotion_detector, response_generator = load_models()
    
#     # User input
#     user_input = st.text_area("How are you feeling today?", height=150,
#                               placeholder="Type your thoughts here...")
    
#     col1, col2 = st.columns([1, 1])
#     with col1:
#         submit_button = st.button("Analyze and Respond")
#     with col2:
#         clear_history = st.button("Clear History")
        
#     # Clear history if requested
#     if clear_history:
#         st.session_state.messages = []
#         st.experimental_rerun()
    
#     # Handle the submit button
#     if submit_button:
#         if not user_input.strip():
#             st.warning("Please enter some text to proceed.")
#         else:
#             # Show a spinner while processing
#             with st.spinner("Analyzing your entry..."):
#                 # Detect emotion
#                 emotion, confidence = emotion_detector.predict(user_input)
                
#                 # Add slight delay for better UX
#                 time.sleep(0.5)
                
#                 # Generate response using our enhanced local generator
#                 response = response_generator.generate_response(user_input, emotion)
                
#                 # Add to conversation history
#                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 st.session_state.messages.append({
#                     "role": "user",
#                     "content": user_input,
#                     "timestamp": timestamp
#                 })
#                 st.session_state.messages.append({
#                     "role": "assistant",
#                     "content": response,
#                     "emotion": emotion,
#                     "confidence": confidence,
#                     "timestamp": timestamp
#                 })
                
#                 # Display detected emotion
#                 st.markdown(f"<p class='emotion-label'>Detected Emotion: {emotion.title()} ({confidence:.2f})</p>",
#                            unsafe_allow_html=True)
                
#                 # Display the AI response
#                 st.markdown(f"<p class='therapist-header'>AI Therapist Says:</p>", unsafe_allow_html=True)
#                 st.markdown(f"<div class='therapist-message'>{response}</div>", unsafe_allow_html=True)
    
#     # Display conversation history
#     if st.session_state.messages:
#         st.markdown("---")
#         st.markdown("### Conversation History")
        
#         for i in range(0, len(st.session_state.messages), 2):
#             if i + 1 < len(st.session_state.messages):
#                 user_msg = st.session_state.messages[i]
#                 ai_msg = st.session_state.messages[i + 1]
                
#                 # Format timestamp
#                 timestamp = user_msg.get("timestamp", "")
#                 if timestamp:
#                     st.caption(f"**{timestamp}**")
                
#                 # Display user message
#                 st.markdown(f"<div class='user-message'>{user_msg['content']}</div>",
#                            unsafe_allow_html=True)
                
#                 # Display AI response with emotion
#                 emotion = ai_msg.get('emotion', 'unknown')
#                 confidence = ai_msg.get('confidence', 0.0)
#                 st.markdown(f"<p class='emotion-label'>Detected: {emotion.title()} ({confidence:.2f})</p>",
#                            unsafe_allow_html=True)
#                 st.markdown(f"<div class='therapist-message'>{ai_msg['content']}</div>",
#                            unsafe_allow_html=True)
                
#                 st.markdown("---")
    
#     # Show emotion distribution if enough entries
#     if len(st.session_state.messages) >= 4:
#         emotions = [msg.get('emotion') for msg in st.session_state.messages
#                    if msg.get('role') == 'assistant' and msg.get('emotion')]
        
#         if emotions:
#             st.markdown("### Your Emotional Journey")
#             st.bar_chart({emotion: emotions.count(emotion) for emotion in set(emotions)})

# # Sidebar content
# def sidebar():
#     """Define sidebar with info"""
#     st.sidebar.header("About")
#     st.sidebar.markdown("""
#     This AI Mental Health Journal uses:
    
#     - **Emotion Detection**: DistilBERT fine-tuned on GoEmotions
#     - **Response Generation**: AI-generated therapeutic responses
    
#     The app can detect 28 different emotions and provides supportive responses.
#     """)
    
#     st.sidebar.header("How It Works")
#     st.sidebar.markdown("""
#     1. **Type your thoughts** in the text area
#     2. Click **Analyze and Respond**
#     3. The AI will **detect your emotions** and **respond supportively**
#     4. Your conversation history is saved in the current session
#     """)
    
#     st.sidebar.header("Emotion Categories")
#     with st.sidebar.expander("View All Emotions"):
#         emotions = [
#             "admiration", "amusement", "anger", "annoyance", "approval", "caring",
#             "confusion", "curiosity", "desire", "disappointment", "disapproval",
#             "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
#             "joy", "love", "nervousness", "optimism", "pride", "realization",
#             "relief", "remorse", "sadness", "surprise", "neutral"
#         ]
#         st.markdown("- " + "\n- ".join(emotions).title())
    
#     # Display template examples
#     templates = load_templates()
#     if templates:
#         st.sidebar.header("Response Templates")
#         template_category = st.sidebar.selectbox(
#             "View templates for emotion:",
#             options=list(templates.keys()),
#             index=0
#         )
#         if template_category in templates:
#             with st.sidebar.expander("View Templates"):
#                 for i, template in enumerate(templates[template_category]):
#                     st.markdown(f"**Template {i+1}:**\n\n{template}")
    
#     st.sidebar.markdown("""
#     ---
#     **Note**: This is not a replacement for professional mental health support.
#     If you're in crisis, please contact a mental health professional or crisis hotline.
#     """)

# # Run the app
# if __name__ == "__main__":
#     sidebar()
#     main()



# # import streamlit as st
# # from emotion_detector.predict import EmotionDetector
# # from response_generator.generate import ResponseGenerator
# # import os

# # # Initialize the emotion detector and response generator
# # @st.cache_resource
# # def load_models():
# #     # Path to the fine-tuned model
# #     model_path = "models/emotion_detector"
    
# #     # Use the default model if fine-tuned model doesn't exist
# #     if not os.path.exists(model_path):
# #         model_path = None
    
# #     emotion_detector = EmotionDetector(model_path=model_path)
    
# #     # Get API key from environment or Streamlit secrets
# #     api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
# #     response_generator = ResponseGenerator(api_key=api_key)
    
# #     return emotion_detector, response_generator

# # # Get models
# # emotion_detector, response_generator = load_models()

# # # Set up the Streamlit app
# # st.title("🧠 AI Mental Health Journal")

# # # User input
# # user_input = st.text_area("How are you feeling today?")

# # # Handle the submit button
# # if st.button("Analyze and Respond"):
# #     if user_input:
# #         # Show a spinner while processing
# #         with st.spinner("Analyzing your entry..."):
# #             # Detect emotion
# #             emotion, confidence = emotion_detector.predict(user_input)
            
# #             # Generate response
# #             response = response_generator.generate_response(user_input, emotion)
            
# #             # Display detected emotion with confidence
# #             st.success(f"**Detected Emotion**: {emotion.title()} ({confidence:.2f})")
            
# #             # Display the AI therapist response
# #             st.info(f"**AI Therapist Says**: {response}")
# #     else:
# #         st.warning("Please enter some text to proceed.")




# # # import streamlit as st
# # # from response_generator.generate import generate_response

# # # st.title("🧠 AI Mental Health Journal")

# # # user_input = st.text_area("How are you feeling today?")
# # # if st.button("Analyze and Respond"):
# # #     if user_input:
# # #         emotion, response = generate_response(user_input)
# # #         st.success(f"**Detected Emotion**: {emotion}")
# # #         st.info(f"**AI Therapist Says**: {response}")
# # #     else:
# # #         st.warning("Please enter some text to proceed.")
