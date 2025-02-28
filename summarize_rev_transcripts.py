import os
import json
import re
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TranscriptAnalyzer:
    def __init__(self, directory_path):
        """Initialize the analyzer with the directory containing transcript files."""
        self.directory_path = directory_path
        self.transcripts = {}
        self.summaries = {}
        self.metadata = {}
        self.topics = defaultdict(list)
        
    def load_transcripts(self, extensions=['.txt']):
        """Load transcript files from the directory.
        
        Args:
            extensions: List of file extensions to consider as transcripts. Default is ['.txt']
        """
        file_count = 0
        if not os.path.exists(self.directory_path):
            print(f"Error: The directory '{self.directory_path}' does not exist.")
            return file_count
            
        print(f"Searching for files with extensions: {extensions}")
        
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                # Check if the file has any of the specified extensions
                if any(file.lower().endswith(ext.lower()) for ext in extensions):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.directory_path)
                    
                    try:
                        # Try different encodings if utf-8 fails
                        encodings = ['utf-8', 'latin-1', 'cp1252']
                        content = None
                        
                        for encoding in encodings:
                            try:
                                with open(file_path, 'r', encoding=encoding) as f:
                                    content = f.read()
                                break  # If successful, exit the encoding loop
                            except UnicodeDecodeError:
                                continue  # Try the next encoding
                                
                        if content is not None:
                            self.transcripts[relative_path] = content
                            file_count += 1
                            print(f"Loaded: {relative_path}")
                        else:
                            print(f"Could not decode {relative_path} with any of the attempted encodings.")
                    except Exception as e:
                        print(f"Error loading {relative_path}: {e}")
        
        print(f"Total transcript files loaded: {file_count}")
        return file_count
        
    def generate_summary(self, transcript_text, max_sentences=5):
        """Generate a summary for a transcript by extracting key sentences."""
        # Split into sentences
        sentences = sent_tokenize(transcript_text)
        
        if not sentences:
            return "Empty transcript or unable to parse sentences."
            
        # If there are very few sentences, return them all
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
            
        # Simple extractive summarization based on word frequency
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(transcript_text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Get word frequency
        word_freq = FreqDist(filtered_words)
        
        # Score sentences based on word frequency
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence.lower())
            score = sum(word_freq[word] for word in sentence_words if word in word_freq)
            # Normalize by sentence length to avoid bias toward longer sentences
            if len(sentence_words) > 0:  # Avoid division by zero
                sentence_scores[i] = score / len(sentence_words)
            else:
                sentence_scores[i] = 0
                
        # Get top N sentences with highest scores
        top_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
        top_sentence_indices.sort()  # Sort by original order
        
        summary = " ".join([sentences[i] for i in top_sentence_indices])
        return summary
        
    def generate_all_summaries(self):
        """Generate summaries for all loaded transcripts."""
        for path, content in self.transcripts.items():
            self.summaries[path] = self.generate_summary(content)
            print(f"Generated summary for: {path}")
        return self.summaries
        
    def extract_metadata(self, transcript_text):
        """Extract metadata like speakers, dates, meeting type."""
        metadata = {
            "speakers": [],
            "date": None,
            "type": None,
            "keywords": []
        }
        
        # Extract potential speakers (names followed by colon)
        speaker_pattern = re.compile(r'([A-Z][a-z]+ [A-Z][a-z]+):', re.MULTILINE)
        speakers = speaker_pattern.findall(transcript_text)
        metadata["speakers"] = list(set(speakers))  # Remove duplicates
        
        # Try to determine transcript type
        if "meeting" in transcript_text.lower():
            metadata["type"] = "meeting"
        elif "interview" in transcript_text.lower():
            metadata["type"] = "interview"
        elif "deposition" in transcript_text.lower() or "court" in transcript_text.lower():
            metadata["type"] = "legal"
        else:
            metadata["type"] = "unknown"
            
        # Extract date (simple pattern, might need refinement)
        date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} [A-Z][a-z]+ \d{2,4}|[A-Z][a-z]+ \d{1,2},? \d{2,4}')
        dates = date_pattern.findall(transcript_text)
        if dates:
            metadata["date"] = dates[0]
            
        # Extract keywords (top frequency words excluding stopwords)
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(transcript_text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 3]
        word_freq = FreqDist(filtered_words)
        metadata["keywords"] = [word for word, _ in word_freq.most_common(10)]
        
        return metadata
        
    def extract_all_metadata(self):
        """Extract metadata for all loaded transcripts."""
        for path, content in self.transcripts.items():
            self.metadata[path] = self.extract_metadata(content)
            print(f"Extracted metadata for: {path}")
        return self.metadata
        
    def identify_topics(self, num_topics=10):
        """Group transcripts by common topics."""
        # A simple approach using keyword overlap
        all_keywords = []
        for metadata in self.metadata.values():
            all_keywords.extend(metadata["keywords"])
            
        # Find most common keywords across all transcripts
        keyword_freq = FreqDist(all_keywords)
        common_topics = [word for word, _ in keyword_freq.most_common(num_topics)]
        
        # Categorize transcripts by these topics
        for path, metadata in self.metadata.items():
            for topic in common_topics:
                # If any keyword matches the topic or topic is in the transcript
                if topic in metadata["keywords"] or topic in self.transcripts[path].lower():
                    self.topics[topic].append(path)
                    
        return self.topics
        
    def save_results(self, output_dir):
        """Save all analysis results to JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'summaries.json'), 'w') as f:
            json.dump(self.summaries, f, indent=2)
            
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        with open(os.path.join(output_dir, 'topics.json'), 'w') as f:
            json.dump(self.topics, f, indent=2)
            
        print(f"Results saved to {output_dir}")
        
    def build_searchable_index(self, output_dir):
        """Create a searchable index of all transcripts."""
        index = {}
        
        for path, content in self.transcripts.items():
            # Create a dictionary with all the important information
            index[path] = {
                "summary": self.summaries.get(path, ""),
                "metadata": self.metadata.get(path, {}),
                "content": content,  # Full content for searching
                "path": path  # Original path
            }
            
        with open(os.path.join(output_dir, 'transcript_index.json'), 'w') as f:
            json.dump(index, f, indent=2)
            
        print(f"Searchable index created at {output_dir}/transcript_index.json")
        return index
        
    def search_transcripts(self, query, index_file):
        """Search transcripts using a query string."""
        # Load the index if it's a file path
        if isinstance(index_file, str):
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = index_file
            
        results = []
        query = query.lower()
        
        for path, data in index.items():
            score = 0
            
            # Check content
            if query in data["content"].lower():
                score += 3
                
            # Check summary (higher weight)
            if query in data["summary"].lower():
                score += 5
                
            # Check metadata
            for key, value in data["metadata"].items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and query in item.lower():
                            score += 2
                elif isinstance(value, str) and query in value.lower():
                    score += 2
                    
            if score > 0:
                results.append({
                    "path": path,
                    "score": score,
                    "summary": data["summary"],
                    "metadata": data["metadata"]
                })
                
        # Sort results by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

# Usage example
if __name__ == "__main__":
    import sys
    import os
    
    # Check if directory path was provided as command line argument
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = input("Enter the full path to your transcript files directory: ")
    
    # Expand the tilde to the home directory if present
    directory_path = os.path.expanduser(directory_path)
    
    print(f"Searching for transcript files in: {directory_path}")
    
    # Initialize analyzer with user-provided path
    analyzer = TranscriptAnalyzer(directory_path)
    
    # Try loading with different file extensions
    file_extensions = ['.txt', '.json', '.vtt', '.srt', '.doc', '.docx', '.md', '.rtf']
    print(f"Looking for transcript files with extensions: {file_extensions}")
    file_count = analyzer.load_transcripts(extensions=file_extensions)
    
    if file_count == 0:
        print("No transcript files found. Please check:")
        print("1. Is the directory path correct?")
        print("2. Do the files have .txt extension?")
        print("3. Do you have permission to access these files?")
        print("\nYou can try again with the correct path.")
        sys.exit(1)
    
    # Continue with processing if files were found
    analyzer.generate_all_summaries()
    analyzer.extract_all_metadata()
    analyzer.identify_topics()
    analyzer.save_results("transcript_analysis")
    index = analyzer.build_searchable_index("transcript_analysis")
    
    print("\nAnalysis complete! You can now search your transcripts.")
    
    # Interactive search mode
    while True:
        query = input("\nEnter a search term (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        results = analyzer.search_transcripts(query, index)
        if not results:
            print("No matching transcripts found.")
        else:
            print(f"\nFound {len(results)} matching transcripts:")
            for i, result in enumerate(results[:5]):  # Show top 5 results
                print(f"\n{i+1}. File: {result['path']}")
                print(f"   Score: {result['score']}")
                print(f"   Summary: {result['summary'][:200]}...")
                
            if len(results) > 5:
                print(f"\n...and {len(results) - 5} more results.")