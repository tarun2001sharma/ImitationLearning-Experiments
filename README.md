# Imitation Learning Experiments

This repository contains my project implementation for exploring supervised learning techniques in decision making using imitation learning algorithms. The work involves implementing and comparing various strategies for behavior cloning and multimodal policy learning.

## Project Overview

The goal of this project is to deepen the understanding of imitation learning by implementing several approaches:

- **Conceptual Questions:**  
  Theoretical answers on topics including deterministic vs. probabilistic encoders, multimodal action distributions, continuous vs. discrete action spaces, dataset coverage, and generative models (Gaussian Mixture Models, Variational Auto Encoders, Generative Adversarial Networks, Energy-Based Models, and Diffusion Models).

- **Coding Tasks:**  
  Implementation and evaluation of different imitation learning methods:
  
  1. **Behavior Cloning (BC), Unimodal:**  
     - Files: `policy/agent/bc.py` and `policy/agent/gcbc.py`  
     - Task: Complete the provided templates to implement standard behavior cloning and goal-conditioned behavior cloning. Train these models on both fixed-goal and changing-goal datasets, and analyze the loss curves.

  2. **Behavior Transformer (BeT), Multimodal:**  
     - Files: `policy/agent/bet.py` and `policy/agent/networks/kmeans_discretizer.py`  
     - Task: Implement a behavior transformer style multimodal policy, incorporating a KMeans clustering-based discretizer. Train this model on a multimodal dataset and observe how well the approach captures the multimodality in the data.

  3. **Behavior Cloning (BC), Multimodal:**  
     - Task: Train the standard BC approach on the multimodal dataset and compare its performance against the BeT approach.

  4. **Bonus Task (Nearest Neighbor / VINN):**  
     - Task: Implement a nearest neighbor algorithm inspired by VINN to further compare performance on the multimodal dataset. This approach introduces multimodality by randomly selecting among multiple nearest neighbor matches.

## Environment and Datasets

- **Environment:**  
  The experiments are conducted on a 2D goal-reaching environment following the OpenAI Gym API. The environment simulates an agent that starts at a variable location and attempts to reach a goal position.

- **Datasets:**  
  Three demonstration datasets are used:
  - **Fixed Goal:** Fixed goal position with variable start positions.
  - **Changing Goal:** Both start and goal positions are variable.
  - **Multimodal:** Fixed start/goal positions with variable intermediate waypoints.
  
  Each dataset consists of 1000 demonstrations, split into 800 training examples and 200 validation examples.

## Setup and Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<yourusername>/ImitationLearning-Experiments.git
   cd ImitationLearning-Experiments
