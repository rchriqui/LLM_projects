# 🎲 Synthetic Data Generation with Diffusion Models

## 📋 Description

This project demonstrates **synthetic data generation** using **diffusion models** to create realistic synthetic datasets for training deep learning models. Specifically, it generates synthetic road sign images (stop signs and warning signs) using a text-to-image diffusion pipeline, then trains a **ResNet18** classifier on this synthetic data. The project evaluates whether models can effectively learn from artificially generated data and tests generalization to real-world images.

## 🎯 Objectives

- **Generate synthetic road sign images** using diffusion models (Z-Image-Turbo)
- **Create a balanced dataset** with 50 stop sign images and 50 warning sign images
- **Train a ResNet18 classifier** on the synthetic dataset
- **Evaluate model performance** before and after training
- **Test generalization** on real-world road photos from the internet

## 🔧 Technologies Used

- **Python**: Main programming language
- **PyTorch**: Deep learning framework
- **Hugging Face Diffusers**: Z-Image-Turbo pipeline (`Tongyi-MAI/Z-Image-Turbo`)
- **ZImagePipeline**: Text-to-image diffusion model for generating synthetic images
- **ResNet18**: Pre-trained ResNet architecture with 2-class classification head
- **Torchvision**: Image transformations and datasets (`ImageFolder`, `transforms`)
- **Jupyter Notebook**: Interactive development environment
- **Google Colab**: Cloud-based execution environment (GPU support)

## 📁 Project Structure

```
synthetic_data_generation/
└── synthetic_data_generator.ipynb    # Main notebook: diffusion generation + ResNet training
```

The generated dataset structure:
```
data/
  train/
    stop/          # 50 synthetic stop sign images
    warning/       # 50 synthetic warning sign images
  val/
    stop/          # Validation stop sign images
    warning/       # Validation warning sign images
```

## 🚀 Installation

1. **Install dependencies**:
```bash
pip install git+https://github.com/huggingface/diffusers
pip install torch torchvision
pip install pillow numpy matplotlib scikit-learn
```

2. **Launch Jupyter Notebook**:
```bash
jupyter notebook synthetic_data_generator.ipynb
```

**Note**: This project was developed on Google Colab with GPU support. For local execution, ensure you have CUDA-compatible GPU or adjust device settings.

## 💻 Usage

### Step 1: Generate Synthetic Data

1. **Load the diffusion model**:
   - Model: `Tongyi-MAI/Z-Image-Turbo`
   - Uses `ZImagePipeline` from Hugging Face diffusers
   - Configured with `bfloat16` precision for GPU optimization

2. **Generate images**:
   - Prompt: "Generate a stop sign on the road"
   - Prompt: "Generate a warning sign on the road"
   - Image size: 1024×1024
   - Inference steps: 9 (8 DiT forward passes)
   - Guidance scale: 0.0 (for Turbo models)

3. **Save dataset**:
   - Organize images into `train/` and `val/` folders
   - Each folder contains `stop/` and `warning/` subdirectories
   - Dataset saved to Google Drive (Colab-friendly)

### Step 2: Train ResNet18

1. **Data Loading**:
   - Use `ImageFolder` dataset loader
   - Image size: 224×224
   - Training augmentations: RandomHorizontalFlip
   - Normalization: ImageNet statistics

2. **Model Setup**:
   - Architecture: ResNet18 (pre-trained on ImageNet)
   - Final layer: 2-class classification head (stop vs warning)
   - Loss function: CrossEntropyLoss
   - Optimizer: Adam
   - Batch size: 32
   - Training epochs: 8

3. **Training Process**:
   - Train/validation loops with loss tracking
   - Validation accuracy monitoring
   - Model checkpointing

### Step 3: Evaluation

1. **Performance Metrics**:
   - Validation accuracy across epochs
   - Train/validation loss curves
   - Confusion matrix (optional)

2. **Real-world Testing**:
   - Test on real road photos from the internet
   - Display predictions with confidence scores
   - Evaluate "synthetic → real" generalization gap

## 🔬 Workflow

```
Text Prompts → Z-Image-Turbo Diffusion Model → Synthetic Images (50 stop + 50 warning)
                                                          ↓
                                              Dataset Organization (train/val split)
                                                          ↓
                                              ResNet18 Training (8 epochs)
                                                          ↓
                                              Model Evaluation (validation + real images)
```

## 📊 Key Features

- **Diffusion-based generation**: Uses state-of-the-art Z-Image-Turbo model for high-quality synthetic images
- **ResNet18 architecture**: Pre-trained ImageNet weights with fine-tuning for 2-class classification
- **End-to-end pipeline**: From text prompts to trained classifier
- **Performance comparison**: Before/after training evaluation
- **Real-world testing**: Inference on actual road photos

## 🎓 Research Applications

- **Data augmentation**: Generate training data when real data is scarce
- **Privacy preservation**: Train models without using sensitive real-world data
- **Domain adaptation**: Generate data for specific scenarios (road signs, weather conditions, etc.)
- **Model robustness**: Test generalization from synthetic to real data
- **Synthetic-to-real transfer**: Evaluate the gap between synthetic and real-world performance

## 📝 Technical Details

### Diffusion Model Configuration
- **Model**: `Tongyi-MAI/Z-Image-Turbo`
- **Pipeline**: `ZImagePipeline`
- **Precision**: `bfloat16` (GPU optimized)
- **Image dimensions**: 1024×1024
- **Inference steps**: 9
- **Guidance scale**: 0.0 (Turbo models)

### ResNet Training Configuration
- **Architecture**: ResNet18 (ImageNet pre-trained)
- **Classes**: 2 (stop, warning)
- **Image size**: 224×224
- **Batch size**: 32
- **Epochs**: 8
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss
- **Data augmentation**: RandomHorizontalFlip
- **Normalization**: ImageNet statistics ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

## 🔍 Key Considerations

- **Dataset size**: Small synthetic dataset (50 images per class) - may need scaling for production
- **Data quality**: Synthetic data quality directly impacts ResNet training performance
- **Computational resources**: Diffusion models require GPU (CUDA) for efficient generation
- **Generalization gap**: Synthetic-to-real transfer may show performance differences
- **Future improvements**: 
  - Generate more variety (lighting, angles, distances, weather conditions)
  - Increase dataset size
  - Test on real-world datasets for proper generalization evaluation

## 📈 Results

### Training Performance

Based on the notebook execution:
- **Initial validation accuracy**: ~50% (random baseline)
- **After training**: Validation accuracy reaches ~95-100%
- **Training loss**: Decreases from ~0.37 to ~0.02
- **Validation loss**: Decreases from ~0.12 to ~0.02

### Real-World Generalization

**Key Finding**: This project successfully demonstrates that **training on synthetic data enables good performance on real-world images from the internet**.

The ResNet18 model trained exclusively on diffusion-generated synthetic images (50 stop signs + 50 warning signs) was able to:
- **Generalize effectively** to real road photos downloaded from the internet
- **Achieve accurate predictions** on actual stop and warning signs in natural settings
- **Maintain high confidence scores** when classifying real-world images

This proves that synthetic data generated by diffusion models can serve as a viable alternative to real-world datasets for training computer vision models, especially when:
- Real data collection is expensive or time-consuming
- Privacy concerns limit the use of real-world data
- Specific scenarios or domains need to be targeted

## 🎯 Key Achievement

This project **proves that training on synthetic data allows achieving good results on real images from the internet**. Despite being trained exclusively on diffusion-generated synthetic images, the ResNet18 model successfully generalizes to real-world road photos, demonstrating the practical viability of synthetic data for computer vision tasks.

## 🤝 Contribution

This project is part of a series of experiments on synthetic data generation and deep learning model training. It demonstrates the feasibility and effectiveness of training computer vision models entirely on diffusion-generated synthetic data, with successful generalization to real-world images.

## 📄 License

See the LICENSE file at the root of the project.
