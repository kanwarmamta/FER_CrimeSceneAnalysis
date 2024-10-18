Hostage Crime Scene Detection Using Deep Learning

Mentored by: Dr. Raju Halder
Course: CS389: Innovative Design Lab
Video Demonstration: https://drive.google.com/file/d/1UPjHciPHlFgRKOiYOFvEr8jMNTn2aKFT/view?usp=sharing
Presentation: https://docs.google.com/presentation/d/1-wqyl_ew_erSJ5sCK8KA83lIR-wx3_pC/edit?usp=sharing&ouid=106073243158613320134&rtpof=true&sd=true

Project Overview

This project aims to develop a deep learning-based system for identifying hostage situations in crime scene footage, including video clips and images. The goal is to enhance crime detection efficiency by using machine learning techniques for facial expression recognition, weapon detection, and audio processing.
Motivation

    Security forces often face challenges in obtaining precise details during hostage crises, affecting their decision-making. Accurate, automated analysis can help reduce risks and save lives.
    While CCTV cameras are widely deployed globally, manual monitoring is impractical, necessitating intelligent crime detection systems.

Work Done So Far

    Data Collection:
        Curated a dataset of 100 video clips (50 hostage and 50 non-hostage) for training and testing.

    Facial Expression Recognition:
        Implemented Patt-Lite and EfficientFace models for detecting emotions using the FER2013 and RAF-DB datasets, achieving 74% accuracy.
        Trained the model to classify emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

    Model Design:
        Fine-tuned MobileNetV1 architecture with additional techniques like Patch Extraction, Global Average Pooling, and an Attention Classifier.
        Developed a lightweight CNN architecture for use on mobile and embedded devices.

    Challenges Faced:
        Limited data availability for negative reactions.
        High inter-class similarities led to reduced accuracy.
        RAM constraints during training with large datasets.
        Occlusions and partially covered faces affected recognition accuracy.

Key Features

    Real-Time Detection: Capable of detecting emotions in both pre-recorded and live video streams.
    Robust Facial Expression Recognition: Leveraged FER2013 and RAF-DB datasets to improve recognition in diverse conditions.
    Lightweight Architecture: Optimized for deployment on mobile and edge devices.
