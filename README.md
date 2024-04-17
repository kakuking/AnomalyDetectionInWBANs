# Slide 1 - Intro
Welcome Everyone to the next presentation, I would have loved to meet everyone in person, unfortunately, stuff happened, and here I am presenting on Zoom. 

SO I am **Kartikeya Dubey**, a 4th year undergraduate student at BITS Hyderabad, and I am here to present the paper "**Anomaly Detection in WBANs using CNN-Autoencoders and LSTMs**". that I co-authored with my professor, **Chittaranjan Hota**. Let's begin -

# Slide 2 - What is a WBAN
Some of you here might know what a WBAN is, or at least might be able to guess that the AN part is **Area Network**. You would be right, a WBAN is a **Wireless Body Area Network**. WBANs are wireless networks, consisting of microscopic sensors. These sensors are placed either on a subject's body or attached to their clothes. They record **physiological readings**, such as Pulse or Heart Rate. 

These sensors are extremely small, and also have an extremely small power consumption. As such, in most cases, they cannot transmit the data they collect directly to the internet. So, instead there is a central unit, usually kept on the person's body or in their vicinity. The sensors first transmit the data to this central unit, and this central unit forwards it to internet, such as  aremote data centre which doctors worldwide can access. This data can also be processed, to improve patient healthcare.  Another wrench in the process is that in some instances, however, what is considered an **anomaly may not actually be one, and the readings are correct**. To mitigate such cases, we must detect anomalies promptly.

It is obvious that these WBANs can improve patient-care dramatically, providing insights even when a patient is not in a healthcare facility. WBANs find various uses, ranging from remote medical monitoring to early detese detection. For viability in these critical area, the accuracy of these readings must be high. This is where we focused our research, on the detection and handling of these anomalies in WBAN outputs. It might seem like a easy task, however rather than just classifying a model to be an anomaly, our model(s) can **also replace these with predicted actual values to ease downstream processing.** 

# Slide 3 - Related Work
As with any study, we first undertook a literature review of this research area 

The paper by MU Harun Al Rasyid et al., titled **Anomalous data detection in WBAN measurements** employs a sliding window to predict the next reading of a sensor. The paper proposes using the errors from this regressed data to predict whether a reading is erroneous. But, it considers that  if the readings are considered erroneous, it must be a sensor fault, disregarding the posibbility that the readings are actually accurate and must be reported as such.

Next, is the paper **continuous change detection mechanism to identify anomalies in ECG signals for WBAN-based health-care environments** by F. A. Khan et al. This paper uses a Markov Model to classify the readings. Here, the features of the ECG dataset are extracted using a Discrete Wavelet Transform (DWT). The features are divided into feature sequences. The probability for each feature sequence is then calculated, and the system decides if an anomaly has occurred. The paper, however, focuses on ECG data. We aim to develop a more generic model. the model implemented in this paper is for a single time series data while We aim to find anomalies across multiple time series datasets

Finally, O.I Provotar et al. in **Unsupervised anomaly detection in time series using LSTM-based autoencoders** propose using a LSTM-based encoder to tackle this problem. They determine the dominant frequency then, the signal is decomposed into seasonal, trend, and residual. Finally, the decomposed components are used to decide whether an anomaly occurs. This paper again focuses on a single sensor reading.

With the literature review in mind, we developed our approach to anomaly detection, capable of detecting anomalies in multiple physiological readings. 

# Slide 4 - Data Preparation