# Tetris Gameplay with Gesture Detection
Our project introduces a novel interface for Tetris, leveraging gesture detection powered by a deep learning model trained on facial landmarks. This innovation transforms conventional gameplay, enabling controls through intuitive head movements. Developed using the Pygame library, our interface facilitates controls like piece movement and rotation via corresponding head motions, underscoring the potential of gesture-based interfaces in gaming.

## 1. Introduction & Background
### 1.1 Objective
By intertwining the nostalgic charm of Tetris with contemporary gesture detection technologies, we aim to reshape the gaming landscape. Our project's motivation stems from both innovation and inclusivity. Traditional gaming controls, often hand-based, limit gameplay interaction. Our gesture-based controls contribute to an evolving gaming space that enhances user engagement. Moreover, our controls provide an alternative for those who may find typical gaming controls challenging, promoting more accessible gaming experiences.

### 1.2 Intended Outcome
Our project serves dual purposes: entertainment and potential therapeutic applications. While offering an engaging way to play Tetris, it also suggests potential uses in motor skill rehabilitation or coordination improvement exercises.

## 2. Technical Overview
### 2.1 Workflow
Face Detector: Employs a real-time face detector optimized for CPUs.
PIPNet: Calculates facial landmarks.
Gesture Detector: Determines the onset of a gesture.
Gesture Classifier: Classifies the gesture based on facial landmarks.
We utilize the Resnet18 CNN architecture for its accuracy and swift inference. The facial landmarks are determined using the PIPNet method.

## 3. File Structure Overview
### 3.1 Face Landmarks Folder
Contains the trained model and scripts to detect facial landmarks, which are crucial for the gesture detection mechanism.

### 3.2 My Pipnet Folder
Houses the implementation of the PIPNet method, which is used to calculate facial landmarks. This method is integral to the gesture detection process.

### 3.3 tetris_pygame Folder
main.py: Main game logic with a GUI interface.
tetris.py: Contains the Tetris game mechanics and gesture detection.
settings.py: Configuration and settings for the game.
button.py: Manages button functionalities in the game.
cam_testing.py: Tests camera functionalities.

### 3.4 Video Extractor Folder
main.py: Extracts frames from videos and saves them as images, aiding in the data collection process for training the deep learning models.
