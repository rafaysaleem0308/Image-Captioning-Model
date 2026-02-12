# Image-Captioning-Model
🚀 Built an End-to-End Image Captioning System Using Deep Learning
I recently developed a complete Image Caption Generator that converts images into natural language descriptions using a Seq2Seq deep learning architecture.
This project integrates computer vision and NLP into a full pipeline — from feature extraction to deployment.
🔎 Project Highlights
🔹 Feature Extraction
Pretrained ResNet50 (ImageNet weights)
Removed final classification layer
Generated 2048-dimensional image embeddings
GPU-accelerated batch processing with PyTorch
🔹 Text Processing & Vocabulary Engineering
Caption cleaning and tokenization
Custom vocabulary with frequency thresholding
Special tokens: <start>, <end>, <pad>, <unk>
Proper train/validation/test split by image to avoid data leakage
🔹 Model Architecture
Encoder: Linear → BatchNorm → ReLU → Dropout
Decoder: Embedding → Multi-layer LSTM → Linear output layer
Teacher Forcing during training
Gradient clipping for stability
🔹 Training Setup
CrossEntropyLoss (ignoring padding tokens)
Adam optimizer
Validation monitoring + best model checkpointing
GPU training
🔹 Inference Methods
Greedy Search
Beam Search (for improved caption quality)
🔹 Evaluation
BLEU-4 score
Token-level Precision, Recall, F1-score
Visual comparison between generated and ground-truth captions
🔹 Deployment
Built an interactive Gradio web app
Users can upload an image and receive AI-generated captions in real time
🛠 Tech Stack
Python | PyTorch | torchvision | NumPy | Pandas | NLTK | Scikit-learn | Matplotlib | Gradio | Kaggle GPU
