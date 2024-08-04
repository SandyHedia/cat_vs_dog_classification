# Cat vs. Dog Classification Project

## Project Overview
This project aims to classify images of cats and dogs using various machine learning models, including convolutional neural networks (CNN) and transfer learning techniques. The goal is to build, train, and fine-tune models to achieve high accuracy in distinguishing between images of cats and dogs.

## Project Structure
The project is organized into several steps:
1. **Data Preprocessing**: Loading and preprocessing the image data.
2. **Model Building**: Constructing a CNN for image classification and utilizing pre-trained models.
3. **Model Training**: Training the CNN and fine-tuning pre-trained models on the dataset.
4. **Model Evaluation**: Evaluating the model's performance.
5. **Results Visualization**: Visualizing the training process and results.

## Installation
To run this project, ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- TensorFlow/Keras or PyTorch
- OpenCV (optional for image preprocessing)
- Transformers (Hugging Face)

You can install the required packages using pip:
```bash
pip install numpy pandas matplotlib tensorflow torch opencv-python transformers
```

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/SandyHedia/cat_vs_dog_classification.git
   cd cat_vs_dog_classification
   ```

2. **Run the Jupyter Notebook**:
   Open the `cat_vs_dog_classification.ipynb` file in Jupyter Notebook or Jupyter Lab and run the cells to execute the project.

## Dataset
The dataset used in this project is the [Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765). It consists of 25,000 images of cats and dogs (12,500 images each for cats and dogs).

## Model Architectures
- **CNN**: Custom convolutional neural network with layers such as convolution, pooling, and fully connected layers.
- **MobileNet**: Pre-trained MobileNet model fine-tuned with different configurations:
  - Without freezing any layers
  - Freezing all layers
  - Freezing only convolutional layers
  - Freezing the first two layers
- **Google ViT**: Fine-tuning the Vision Transformer (ViT) model from Hugging Face transformers library.

## Training and Evaluation
- The models are trained using the Adam optimizer and categorical cross-entropy loss function.
- The dataset is split into training and validation sets.
- The performance of the models is evaluated using accuracy and loss metrics.

## Results
- **CNN**: Achieved baseline accuracy with a custom CNN.
- **MobileNet**:
  - Without freezing any layers: Improved accuracy by fine-tuning all layers.
  - Freezing all layers: Used MobileNet as a feature extractor.
  - Freezing only convolutional layers: Fine-tuned the fully connected layers for better performance.
  - Freezing the first two layers: A balanced approach to fine-tuning.
- **Google ViT**: Achieved the best performance with an accuracy of 99.68%.

## Conclusion
This project demonstrates the effectiveness of convolutional neural networks and transfer learning techniques in image classification tasks. The fine-tuning of pre-trained models, such as MobileNet and Google ViT, significantly enhances the performance of the classification model. The ViT model provided the best performance with an accuracy of 99.68%.

## Future Work
- Experiment with different CNN architectures and hyperparameters.
- Implement additional data augmentation techniques to enhance the dataset.
- Explore other pre-trained models for transfer learning.

## Acknowledgements
- The dataset used in this project is provided by Microsoft.
- TensorFlow/Keras or PyTorch libraries are used for building and training the models.
- The ViT model is provided by the Hugging Face library.
