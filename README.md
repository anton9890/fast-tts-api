# Fast TTS API with Llama Integration

Text-to-Speech API server that combines Llama language model with XTTS v2 for high-quality speech synthesis.

## Features

- 🤖 Llama Language Model Integration
- 🗣️ High-quality Text-to-Speech using XTTS v2
- 🌐 FastAPI-based Web Interface
- 🌍 Multi-language Support
- 🚀 Real-time Speech Generation
- 💻 Easy-to-use Web UI

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ Disk Space (for models)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fast-tts-api.git
cd fast-tts-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

The server will automatically download required models on first run.

## Usage

1. Access the web interface:
```
http://localhost:8001
```

2. Enter your text and select the desired language
3. Click "Generate Speech" to create audio
4. The system will:
   - Process your text through Llama
   - Generate a response
   - Convert the response to speech
   - Play the audio automatically

## API Endpoints

- `GET /`: Web interface
- `GET /health`: Server health check
- `POST /tts`: Text-to-speech conversion
  - Parameters:
    - `text`: Input text
    - `language`: Target language code (default: "en")

## Supported Languages

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Polish (pl)
- Turkish (tr)
- Russian (ru)
- Dutch (nl)
- Czech (cs)
- Arabic (ar)
- Chinese (zh-cn)
- Japanese (ja)
- Korean (ko)
- Hindi (hi)

## Directory Structure

```
fast-tts-api/
├── main.py           # Main server implementation
├── requirements.txt  # Python dependencies
├── templates/        # Web interface templates
└── models/          # Downloaded model files (auto-generated)
```

## Notes

- Models are downloaded automatically on first run
- The server uses port 8001 by default
- CUDA GPU is recommended for better performance
- Initial model download may take several minutes

## License

MIT License 