# mental_health_journal/utils/emotion_tracker.py
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64

class EmotionTracker:
    def __init__(self):
        self.entries = []
    
    def add_entry(self, text, emotion, confidence, timestamp=None):
        """Add a new journal entry with emotion data"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        self.entries.append({
            'text': text,
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': timestamp
        })
    
    def get_emotion_counts(self):
        """Get counts of each emotion in the entries"""
        if not self.entries:
            return {}
            
        emotions = [entry['emotion'] for entry in self.entries]
        return dict(pd.Series(emotions).value_counts())
    
    def generate_emotion_chart(self):
        """Generate a chart of emotions over time"""
        if not self.entries or len(self.entries) < 2:
            return None
            
        df = pd.DataFrame(self.entries)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Group emotions by day
        df['date'] = df['timestamp'].dt.date
        emotion_counts = df.groupby(['date', 'emotion']).size().unstack().fillna(0)
        
        # Create the chart
        plt.figure(figsize=(10, 6))
        emotion_counts.plot(kind='line', ax=plt.gca())
        plt.title('Your Emotional Journey')
        plt.xlabel('Date')
        plt.ylabel('Number of Entries')
        plt.tight_layout()
        
        # Convert plot to base64 for displaying in Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return img_str
        
    def get_most_common_emotion(self):
        """Get the most common emotion in the entries"""
        if not self.entries:
            return None
            
        emotions = [entry['emotion'] for entry in self.entries]
        return max(set(emotions), key=emotions.count)