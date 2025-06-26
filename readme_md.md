# Document Classification System

A Streamlit-based prototype for automated document classification, specifically designed for processing order documents including Purchase Orders, Proforma Invoices, and Email Orders.

## Features

- **Real-time Document Classification**: Upload PDFs, text files, or paste content directly
- **Machine Learning Pipeline**: Enhanced Random Forest classifier with TF-IDF and manual feature extraction
- **Interactive Dashboard**: Training data management, performance monitoring, and analytics
- **Model Persistence**: Save and load trained models with versioning
- **User Feedback System**: Continuous learning through user corrections
- **Advanced Analytics**: Feature importance analysis and performance tracking

## Demo

![Document Processing Demo](docs/demo-screenshot.png)

## Document Types Supported

1. **Structured Proforma**: Formal proforma invoices with standardized layouts
2. **Simple Purchase Order**: Basic purchase orders with minimal formatting
3. **Complex Purchase Order**: Multi-line, multi-phase orders with special requirements
4. **Email Body Order**: Informal orders placed via email text

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-classification-system.git
cd document-classification-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run streamlit_python_app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the application:
   - **Document Processing**: Upload files or paste text for classification
   - **Training Dashboard**: Monitor model performance and dataset statistics
   - **Live Monitoring**: View real-time processing metrics
   - **System Settings**: Configure processing parameters

## Project Structure

```
document-classification-system/
├── streamlit_python_app.py    # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore patterns
├── models/                  # Saved ML models (created automatically)
├── data/                    # Training data storage (created automatically)
├── docs/                    # Documentation and assets
└── tests/                   # Unit tests
```

## Machine Learning Pipeline

### Feature Extraction
- **Text Features**: TF-IDF vectorization with n-grams
- **Keyword Features**: Order-specific terminology detection
- **Structural Features**: Document layout and formatting analysis
- **Numerical Features**: Currency, dates, and quantity patterns

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Features**: 200+ TF-IDF features + 50+ manual features
- **Training**: Cross-validation with stratified splits
- **Evaluation**: Accuracy, precision, recall, and F1-score

### Training Data
- Initial synthetic dataset with realistic examples
- User feedback integration for continuous improvement
- Balanced representation across document categories

## Performance

- **Classification Accuracy**: 89-94% on test data
- **Processing Speed**: ~18 seconds average per document
- **Feature Count**: 250+ combined features
- **Model Size**: Lightweight for production deployment

## Development Roadmap

### Phase 1: Traditional ML (Current)
- [x] Random Forest classifier with manual features
- [x] TF-IDF text processing
- [x] Streamlit prototype interface
- [x] Model persistence and versioning

### Phase 2: Enhanced Features
- [ ] Advanced PDF text extraction with layout preservation
- [ ] Image-based document analysis
- [ ] Email API integration (Gmail, Outlook)
- [ ] Batch processing capabilities

### Phase 3: Neural Networks
- [ ] Deep learning models for complex layouts
- [ ] Transfer learning from pre-trained document models
- [ ] Multi-modal processing (text + images)
- [ ] Real-time model updates

### Phase 4: Production Deployment
- [ ] REST API development
- [ ] Microservices architecture
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Monitoring and alerting systems

## Configuration

The application supports various configuration options through the System Settings page:

- **Processing Mode**: Real-time vs. batch processing
- **Confidence Thresholds**: Automatic vs. manual review triggers
- **Email Integration**: Provider and folder configuration
- **Model Parameters**: Retraining frequency and performance targets

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Deployment

### Local Development
```bash
streamlit run streamlit_python_app.py
```

### Docker Deployment
```bash
docker build -t doc-classifier .
docker run -p 8501:8501 doc-classifier
```

### Cloud Deployment
- **Streamlit Cloud**: Connect GitHub repo for automatic deployment
- **Heroku**: Use provided `Procfile` for deployment
- **AWS/GCP**: Container deployment with load balancing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Your Name** - Initial work - [YourGitHub](https://github.com/yourusername)

## Acknowledgments

- Streamlit for the excellent web framework
- scikit-learn for machine learning capabilities
- The open-source community for inspiration and tools

## Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Contact: your.email@example.com
- Documentation: [Project Wiki](https://github.com/yourusername/document-classification-system/wiki)

## Changelog

### v1.0.0 (2024-06-26)
- Initial release with basic classification
- Streamlit interface
- Model persistence
- User feedback system

---

**Note**: This is a prototype system designed for demonstration purposes. For production use, additional security, scalability, and performance optimizations are recommended.