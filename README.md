# ArguLex - Legal Assistant

A professional web-based legal assistant that combines general legal knowledge with advanced PDF document analysis capabilities. The system provides a modern, user-friendly interface with dark mode support and secure authentication.

## Features

### 1. Web Interface
- **Modern UI/UX**
  - Responsive design for all devices
  - Dark mode support
  - Smooth animations
  - User-friendly forms

- **Authentication System**
  - Secure user registration
  - Login functionality
  - Session management
  - Protected routes

### 2. PDF Document Analysis
- **Case Report Generation**
  - Extracts case name, date, and judges
  - Generates comprehensive case summary
  - Identifies key conclusions
  - Creates downloadable reports

- **Document Querying**
  - Answers specific questions about uploaded documents
  - Provides context-aware responses
  - Quotes relevant sections from the document
  - Maintains accuracy and relevance

### 3. Chatbot System
- **Professional Chatbot**
  - Specialized in legal document analysis
  - Handles complex legal queries
  - Provides detailed explanations
  - Maintains professional tone

- **General Chatbot**
  - Answers general legal questions
  - Provides legal information and guidance
  - Maintains professional tone
  - Suggests professional consultation when needed

- **Conversation Management**
  - Maintains chat history
  - Provides well-structured responses
  - Handles multiple topics
  - Professional formatting

## Technical Details

### Dependencies
```python
# Core Dependencies
flask>=2.0.0
openai>=1.0.0
langchain>=0.1.0
PyMuPDF (fitz)>=1.22.0
sentence-transformers>=2.2.0
nltk>=3.8.1
faiss-cpu>=1.7.4
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
pymongo>=4.0.0
```

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with:
OPENAI_API_KEY=your_api_key_here
FLASK_SECRET_KEY=your_secret_key_here
MONGODB_URI=your_mongodb_uri_here
```

4. Run the application:
```bash
python app.py
```

### Usage

1. **Web Interface**
   - Visit `http://localhost:5000` in your browser
   - Create an account or log in
   - Choose between Professional or General chatbot
   - Start chatting or upload documents

2. **PDF Document Analysis**
   - Upload a PDF document
   - Get automatic analysis and summary
   - Ask specific questions about the document
   - Download comprehensive reports

3. **Chatbot Interaction**
   - Ask general legal questions
   - Get professional responses
   - Maintain conversation history
   - Switch between chatbots as needed

## Project Structure

```
project/
├── app.py
├── chatbot/
│   ├── __init__.py
│   ├── professional_chatbot.py
│   ├── general_chatbot.py
│   └── chat_manager.py
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── signup.html
│   ├── selection.html
│   ├── professional_chat.html
│   └── chat.html
├── requirements.txt
├── .env
└── README.md
```

## Key Components

### Web Application
- Flask-based web server
- MongoDB for data storage
- Session management
- Route protection

### Chatbot System
- Professional and General chatbots
- Document analysis capabilities
- Chat history management
- User session handling

### User Interface
- Bootstrap-based responsive design
- Dark mode support
- Form validation
- Interactive elements

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models
- Flask for web framework
- Bootstrap for UI components
- MongoDB for data storage

## Support

For support, please open an issue in the repository or contact the maintainers. 