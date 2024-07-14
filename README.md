# AI-image-captioning-system
Image captioning system that can automatically generate descriptive captions for images uploading. This involves integrating Convolutional Neural Networks (CNNs) for feature extraction from images and Transformer architectures for generating coherent and contextually accurate textual descriptions.

You can clone this repo from: https://github.com/Naeemahmd/AI-image-captioning-system.git

# Project Structure
- Data Collection
- Data Preprocessing
- Feature Extraction
- Model Building
- Evaluation
- Visualization

The dataset used is Flickr8k, which are images that are each paired with five different captions which provide clear descriptions of the salient entities and events: https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download

# virtual environment setup
To build and install the module in a virtual environment, execute the following commands in the project's root directory

```ruby
# create a virtual enviroment     
python3 -m venv myenv

# active the environment
source myenv/bin/activate       # for Linux, Mac
myenv/Scripts/activate          # for Windows

#install required packages
pip install -r requirements.txt
# to update required packages
pip freeze > requirements.txt

```