# Micro-Bach

This project is my attempt at using a neural network to compose musical counterparts in the style of J.S. Bach. The basic idea is to give the model a single melody line (the soprano part) and have it generate the three accompanying harmony parts (alto, tenor, and bass).

My goal was to see if a relatively simple machine learning model could learn the fundamental rules of four-part harmony and voice leading that are characteristic of Bach's chorales. The entire process involves parsing musical data, converting it into a format the model can understand, and then training the model to predict the harmonies.

### How It Works

The process can be broken down into a few main steps:

1.  **Data Processing**: I used the `music21` library to load Bach's chorales directly from its corpus. I wrote a script to scan through them and pull out four-measure segments. To create a larger dataset for training, I also included transposed versions of each segment.

2.  **Piano Roll Encoding**: Music needs to be in a numerical format for a neural network. I converted each musical part into a "piano roll," which is essentially a matrix representing which notes are played at specific, quantized moments in time. Each part is represented as an array of size (21, 64), while bass line is of size (28, 64). The 64 represents 4 measures * 16th note quantization. For the upper three parts, 21 represents the range in midi notes used, while the bass has 28. I analyzed bachs corpus to find the best possible ranges to include as many songs as possible and the range I used represented 85 percent of the chorales. Note, the ranges of the parts overlap, however each part's notes are labeled 1-21/28 respectively.

3.  **The Model**: The core of the project is a simple feedforward neural network built with PyTorch.

      * **Input**: It takes the flattened piano roll of the soprano part.
      * **Architecture**: It has one hidden layer with a ReLU activation function.
      * **Output**: It predicts the flattened piano rolls for the alto, tenor, and bass parts concatenated together, using a sigmoid activation to determine the probability of each note, then uses argmax to select most likely note.

### Built With

  * Python
  * PyTorch
  * music21
  * NumPy
  * Matplotlib

### Getting Started

To get this project running on your own machine, you'll need to follow these steps.

**Prerequisites**
You'll need Python 3 installed, along with the pip package manager.

**Installation**

1.  First, clone the repository to your local machine:

    ```sh
    git clone https://github.com/slamduncaan747/Micro-Bach.git
    cd your-repository
    ```

2.  Install the necessary libraries:

    ```sh
    pip install torch music21 numpy matplotlib
    ```

### How to Run the Project

There are three main scripts you'll use:

1.  **Process the Data**: First, you need to build the dataset from the raw music files.

    ```sh
    python encoding.py
    ```

    This will create a `dataset.pkl` file containing all the piano roll segments.

2.  **Train the Model**: With the dataset ready, you can start training the network.

    ```sh
    python train.py
    ```

    This script will save model checkpoints periodically into the `checkpoints/` folder.

3.  **Generate Music**: Once you have a trained model, you can use it to generate harmonies.

    ```sh
    python sample.py
    ```

    This script will pick a random soprano line from the dataset, feed it to the model, and then generate the full four-part harmony. It will save the output as a MIDI file and show you a visual representation of the music, and open in your configurated midi player if possible.

### Project Status and Future Work

As it stands, the model does a pretty bad job of learning basic harmonic relationships, and it has some clear limitations. It tends to overfit heavily, and by flattening the 2D piano roll into a 1D vector, a lot of important musical structure is lost.

The next steps in this process are:

  * **Change the Architecture**: I'd move to a Convolutional Neural Network (CNN) similar to U-Net, which are much better at handling 2D data like piano rolls and could better preserve the musical structure.
  * **Explore Advanced Models**: I'm also interested in experimenting with Transformers or Diffusion models, which have shown incredible results in other generative tasks and could better capture the long-range dependencies in music.
  * **Improve Musicality**: I'd want to focus on generating more interesting and independent melodic lines in the harmony parts, moving beyond simple chordal accompaniment.
  * **Random Sampling**: I'm gonna try using random sampling instead of argmax to make more unique and interesting generations.
