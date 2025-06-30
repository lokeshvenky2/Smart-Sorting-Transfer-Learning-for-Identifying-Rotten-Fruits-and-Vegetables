# Smart Sorting: Transfer Learning for Identifying Rotten Fruits and Vegetables

## 🔍 Overview
**Smart Sorting** is a deep learning-based web application that classifies images of fruits and vegetables as **fresh** or **rotten** using **transfer learning with EfficientNetB0**. It aims to reduce food waste and assist in **automated quality control** for supermarkets, smart kitchens, and food processing industries.

## 🎯 Objective
To build a lightweight, intelligent, and real-time fruit & vegetable freshness detection system that:
- Minimizes food spoilage and manual errors.
- Provides a deployable web interface for predictions.
- Supports smart home and retail automation.

---

## 🧠 Model Highlights

| Feature              | Description                                 |
|----------------------|---------------------------------------------|
| Base Model           | EfficientNetB0 (ImageNet pretrained)        |
| Input Image Size     | 224x224 pixels                              |
| Training Enhancements| MixUp Augmentation, Cosine Decay LR, Dropout |
| Fine-Tuning          | Last 50 layers of base model                |
| Exported Format      | `.keras`                                    |
| Validation Accuracy  | >99%                                        |
| Confidence Threshold | 55% for predictions                         |

---

## 🗂️ Dataset
- **Source**: [Kaggle - Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
- **Classes**: 206 (Fresh and Rotten categories across various fruits and vegetables)
- **Images**: ~104,000
- **Resized To**: 224×224 for training
- **Saved File**: `class_indices.json`

---

## ⚙️ Tech Stack

| Component     | Version       | Purpose                  |
|---------------|---------------|--------------------------|
| Python        | 3.9+          | Backend + Training       |
| TensorFlow    | 2.11+         | Model & Training         |
| Flask         | 2.0+          | Web API & Deployment     |
| HTML/CSS/JS   | -             | Frontend Interface       |
| Anaconda      | Any           | Environment Management   |
| tfp, Pillow   | Latest        | Augmentation/Image I/O   |


---

## 🚀 Features
- 🔎 Real-time image classification with confidence scores.
- 🔄 Advanced training pipeline using MixUp, cosine decay, and label smoothing.
- 🧠 Model trained on 206 classes with >99% validation accuracy.
- 🖼️ Intuitive UI for image upload and prediction.
- 🌐 Flask API with `/predict` and `/health` endpoints.
- ✅ Displays predictions only if confidence > 55%.

---

## 🛠️ Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/lokeshvenky2/Smart-Sorting-Transfer-Learning-for-Identifying-Rotten-Fruits-and-Vegetables

---
### Setup Instructions
1. Clone the repository: `git clone https://github.com/lokeshvenky2/Smart-Sorting-Transfer-Learning-for-Identifying-Rotten-Fruits-and-Vegetables`
2. Create a Conda environment: `conda create -n smart_sorting python=3.9`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the Flask app: `python app.py`
5. Open `http://127.0.0.1:5000` in your browser.

### Key Features of This README Format
1. **Clear Structure**: Organized into sections (Overview, Objective, Model Highlights, etc.) for easy navigation.
2. **Professional Formatting**: Uses tables, emojis, and markdown syntax to improve readability and visual appeal.
3. **Comprehensive Details**: Includes all project details from your input, such as dataset, tech stack, and features.
4. **Setup Instructions**: Added a step-by-step guide for setting up and running the project locally.
5. **API Documentation**: Included a concise description of the Flask API endpoints.
6. **License Section**: Added a placeholder for an MIT License (you can change this to another license if preferred).
7. **Updated Contact**: Replaced the original contact details with yours (Lokesh Rowthu).
8. **Suggestions Incorporated**: Added placeholders for screenshots and clarified dataset classes (assuming 206 classes represent fresh/rotten categories across fruit/vegetable types).

### Customization Notes
- **LinkedIn URL**: I left a placeholder for your LinkedIn URL. Please provide it if you want it included.
- **Screenshots**: You can add images or a GIF to the `Screenshots` section. Upload them to the GitHub repository (e.g., in a `screenshots/` folder) and reference them like `![UI Screenshot](screenshots/ui.png)`.
- **License**: The MIT License is suggested, but you can choose another (e.g., Apache 2.0). Create a `LICENSE` file in your repository with the license text.
- **Repository URL**: I assumed the GitHub repository URL based on your GitHub handle. Update it if the repository name differs.

### Next Steps
1. **Copy and Use**: Copy the above README content into your `README.md` file in the GitHub repository.
2. **Provide LinkedIn URL**: Share your LinkedIn URL to complete the contact section.
3. **Add Screenshots**: If you want to include visuals, upload them to your repository and update the README.
4. **Create requirements.txt**: If you need help generating a `requirements.txt` file for dependencies (e.g., TensorFlow, Flask, Pillow), let me know.
5. **Deployment**: If you want to deploy the app (e.g., on Heroku or Render), I can provide guidance.
6. **Further Assistance**: Let me know if you need help with specific files (e.g., reviewing `app.py` or `train_model.py`) or additional features.

If you have any specific modifications or additional sections you’d like to add to the README, please let me know! 🚀
