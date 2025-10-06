# 🌕 Lunar Lander Project

Welcome to the **Lunar Lander Project**! 🚀 This repository contains code and resources for training, evaluating, and playing with a reinforcement learning agent designed to solve the classic Lunar Lander environment from OpenAI Gym.

---

## 📖 Table of Contents
- [📘 Overview](#-overview)
- [✨ Features](#-features)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
  - [🧠 Training the Agent](#-training-the-agent)
  - [📊 Evaluating the Agent](#-evaluating-the-agent)
  - [🎮 Playing with the Agent](#-playing-with-the-agent)
- [📂 File Descriptions](#-file-descriptions)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 📘 Overview
The **Lunar Lander Project** demonstrates the application of reinforcement learning to control a lunar lander. The goal is to train an agent to land the spacecraft safely on the moon's surface by optimizing its actions based on rewards.

This project leverages Python and popular machine learning libraries to implement and evaluate the agent.

---

## ✨ Features
- 🌌 **Train** a reinforcement learning agent to solve the Lunar Lander environment.
- 📈 **Evaluate** the agent's performance using pre-trained models.
- 🎮 **Play** the Lunar Lander game interactively with the trained agent.
- 💾 **Save and Load** model parameters for reproducibility.

---

## ⚙️ Installation
To get started, clone this repository and install the required dependencies:

### Prerequisites
Ensure you have Python 3.8 or higher installed. You can download it from [python.org](https://www.python.org/).

### Install Dependencies
This project uses the following Python libraries:
- `gym`: For the Lunar Lander environment.
- `numpy`: For numerical computations.
- `matplotlib`: For visualizations (optional).

Install the dependencies using the following command:

```bash
# Clone the repository
git clone https://github.com/Piyush12-kumar/Lunar-Lander-Project
cd lunar-lander-project

# Install dependencies
pip install gym numpy matplotlib
```

---

## 🚀 Usage

### 🧠 Training the Agent
Train the agent using the `new_agent2.py` script. This script trains the agent using reinforcement learning algorithms and saves the best parameters:

```bash
python new_agent2.py
```

### 📊 Evaluating the Agent
Evaluate the agent's performance with the `evaluate_agent.py` script. This script loads the saved parameters and evaluates the agent in the Lunar Lander environment:

```bash
python evaluate_agent.py
```

### 🎮 Playing with the Agent
Play the Lunar Lander game interactively with the trained agent using the `play_lunar_lander.py` script:

```bash
python play_lunar_lander.py
```

---

## 📂 File Descriptions
- **`new_agent2.py`**: Script for training the reinforcement learning agent.
- **`evaluate_agent.py`**: Script for evaluating the trained agent.
- **`play_lunar_lander.py`**: Script for playing the Lunar Lander game interactively with the trained agent.
- **`my_policy.py`**: Contains the policy implementation for the agent.
- **`best_policy_final.npy`**: File containing the best policy parameters.
- **`checkpoint_best_params.npy`**: Checkpoint file for saving intermediate parameters during training.
- **`extracted_params.npy`**: File containing extracted parameters for analysis.

---

## 🤝 Contributing
Contributions are welcome! 🎉 If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your branch.
4. Open a pull request.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

🌟 **Happy coding and enjoy exploring the Lunar Lander Project!** 🌟
