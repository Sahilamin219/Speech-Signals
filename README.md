# Speech-Signals  <img src=https://user-images.githubusercontent.com/48405411/145718412-43e74d9d-9757-4808-ab62-2e659335f566.png width="100" height="100">
Emotion Recognition From Speech Signals

# 1.Proposed Work Plan 

This study focuses on identifying the best audio feature and model architecture for emotion recognition in speech. 
The experiments were carried out on "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" dataset. 
The robustness of the model was assessed by predicting the emotions of speech utterances on a completely different dataset, the "Toronto Emotional Speech Set (TESS)" dataset. 


Audio features can be broadly classified into two categories, namely time-domain features and frequency-domain features. Time-domain features include the short-term energy of the signal, zero crossing rate, maximum amplitude, minimum energy, entropy of energy. 

Frequency-domain features include spectrograms, Mel-Frequency Cepstral Coefficients (MFCCs), spectral centroid, spectral rolloff, spectral entropy and chroma coefficients [11]. For the purpose of this synopsis, we are going to restrict ourselves to two main features, namely Mel-Frequency Cepstral Coefficients and Mel-spectrograms. 



# 2. Progress Report 

## 2.1 Accomplished Assignment
## 2.1.1 Project Description:
To build a model to recognize emotion from speech using the librosa and sklearn libraries and the RAVDESS dataset of the audio recordings. 
### 2.1.2 Major Obstacles:
Emotions are subjective, people would interpret it differently. It is hard to define the notion of emotions.
Annotating an audio recording is challenging. Should we label a single word, sentence or a whole conversation? How many emotions should we define to recognize?
Collecting data is complex. There is lots of audio data that can be achieved from films or news. However, both of them are biased since news reporting has to be neutral and actors’ emotions are imitated. It is hard to look for neutral audio recording without any bias.
Labeling data requires a high human and time cost. Unlike drawing a bounding box on an image, it requires trained personnel to listen to the whole audio recording, analyze it and give an annotation. The annotation result has to be evaluated by multiple individuals due to its subjectivity.


## 2.2 Data Set Description:

These are two datasets originally made use in the repository RAVDESS and SAVEE, and we only adopted RAVDESS in my model. In the RAVDESS, there are two types of data: speech and song.
Data Set: The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
Description
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7356 files (total size: 24.8 GB). 

The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent.
Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and the song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. 
All conditions are available in three modality formats: 
    Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).  Note, there are no song files for Actor_18.
File Summary
In total, the RAVDESS collection includes 7356 files (2880+2024+1440+1012 files).
File naming convention
Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics: 

Filename identifiers 

    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
![image](https://user-images.githubusercontent.com/48405411/145718778-5c024389-9383-4fab-83a5-df7ed322dc71.png)


Here is the emotion class distribution bar chart.
![image](https://user-images.githubusercontent.com/48405411/145718772-d7b85463-a6bc-49b8-88a0-be2171ac0408.png)

                   
We tested out one of the audio files to know its features by plotting its waveform and spectrogram.
![image](https://user-images.githubusercontent.com/48405411/145718765-7cea6971-b403-4609-a38f-b3f75ec975c8.png)

      Waveform
      
This shape determines what sound comes out. If we can determine the shape accurately, this should give us an accurate representation of the phoneme being produced. The shape of the vocal tract manifests itself in the envelope of the short time power spectrum, and the job of MFCCs is to accurately represent this envelope.
![image](https://user-images.githubusercontent.com/48405411/145718758-d54f0195-9dbb-45b9-a2b7-bec947b73285.png)

      Spectrogram

We would use MFCCs to be our input feature. If you want a thorough understanding of MFCCs, Loading audio data and converting it to MFCCs format can be easily done by the Python package librosa.
3. Designing And Implementing Of Work Flow

# 3.1. Studying On Audio PreProcessing 

## 3.1.1 Fourier Transform

An audio signal is a complex signal composed of multiple ‘single-frequency sound waves’ which travel together as a disturbance(pressure-change) in the medium. When sound is recorded we only capture the resultant amplitudes of those multiple waves. Fourier Transform is a mathematical concept that can decompose a signal into its constituent frequencies. Fourier transform does not just give the frequencies present in the signal, It also gives the magnitude of each frequency present in the signal.
![image](https://user-images.githubusercontent.com/48405411/145718734-d45089cc-6aef-4f24-8b0c-2807e6f84d77.png)
  
The Inverse Fourier Transform is just the opposite of the Fourier Transform. It takes the frequency-domain representation of a given signal as input and does mathematically synthesize the original signal.
We see how we can use Fourier transformation to convert our audio signal into its frequency components in our next topic :-

### 3.1.2 Discrete Fourier Transform (DFT)

Next, we apply DFT to extract information in the frequency domain.
![image](https://user-images.githubusercontent.com/48405411/145718727-d8d6a76f-88a4-48e6-a0c5-98cc23354565.png)

The discrete Fourier transform expresses a signal as a sum of sinusoids. Because the time duration of the sinusoids is infinite, the discrete Fourier transform of the signal reflects the spectral content of an entire signal over time but does not indicate when the spectral content occurs.
However, in our cases, evaluating the spectral content of a signal over a short time scale can be useful. You can use the STFT to evaluate spectral content over short time scales.
![image](https://user-images.githubusercontent.com/48405411/145718720-b3d542c9-fadd-441b-96f6-58120bb509f0.png)

### 3.1.3 Fast Fourier Transform (FFT)

Fast Fourier Transformation(FFT) is a mathematical algorithm that calculates Discrete Fourier Transform(DFT) of a given sequence. The only difference between FT(Fourier Transform) and FFT is that FT considers a continuous signal while FFT takes a discrete signal as input. DFT converts a sequence (discrete signal) into its frequency constituents just like FT does for a continuous signal. In our case, we have a sequence of amplitudes that were sampled from a continuous audio signal. A DFT or FFT algorithm can convert this time-domain discrete signal into a frequency-domain.

### 3.1.4 Short Time Fourier Transform (STFT)

The STFT, also called the windowed Fourier transform or the sliding Fourier transform, partitions the time-domain input signal into several disjointed or overlapped blocks by multiplying the signal with a window function and then applies the discrete Fourier transform to each block. Window functions, also called sliding windows, are functions in which the amplitude tapers gradually and smoothly toward zero at the edges. Because each block occupies different time periods, the resulting STFT indicates the spectral content of the signal at each corresponding time period. When you move the sliding window, you obtain the spectral content of the signal over different time intervals. Therefore, the STFT is a function of time and frequency that indicates how the spectral content of a signal evolves over time. A complex-valued, 2-D array called the STFT coefficients stores the results of windowed Fourier transforms. The magnitudes of the STFT coefficients form a magnitude time-frequency spectrum, and the phases of the STFT coefficients form a phase time-frequency spectrum.
The STFT is one of the most straightforward approaches for performing time-frequency analysis and can help you easily understand the concept of time-frequency analysis. The STFT is computationally efficient because it uses the fast Fourier transform (FFT).
![image](https://user-images.githubusercontent.com/48405411/145718709-ad75e0c4-31bb-4713-9dbd-f300d935243a.png)



## 3.2. Loading Our Data Set 

This is explained in section - 2.2 where we went with the Audio only zip file because we are dealing with finding emotions from speech. The zip file consisted of around 1500 audio files which were in wav format.
 
## 3.3. Defining a Dictionary to hold emotions 
                   
The next step involves organizing the audio files. Each audio file has a unique identifier at the 6th position of the file name which can be used to determine the emotion the audio file consists of. We have 8 different emotions in our dataset.
        1.Calm
        2.Happy 
        3.Sad
        4.Angry
        5.Fearful
        6.Neutral
        7.Disgust
        8.Surprised


## 3.4. Extracting Input Features 
                   
We used Librosa library in Python to process and extract features from the audio files. Librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems. Using the librosa library we were able to extract features i.e MFCC(Mel Frequency Cepstral Coefficient). MFCCs are a feature widely used in automatic speech and speaker recognition. We also separated out the females and males voice by using the identifiers provided in the website. This was because as an experiment we found out that separating male and female voices increased by 15%. It could be because the pitch of the voice was affecting the results.
Each audio file gave us many features which were basically an array of many values. These features were then appended by the labels which we created in the previous step.

### 3.4.1 Understanding Input Features 

#### 3.4.1.1 Mel Cepstrum
![image](https://user-images.githubusercontent.com/48405411/145718693-5d9bd12b-689e-44c8-a9b9-b8cc91bd7e1e.png)

Understanding Block Diagram :-

#### 3.4.1.1.1 Windowing and Framing
Windowing involves the slicing of the audio waveform into sliding frames.
![image](https://user-images.githubusercontent.com/48405411/145718677-2cc052d0-1cde-44a7-8f0c-8f2614b6cb21.png)


But we cannot just chop it off at the edge of the frame. The sudden fall in amplitude will create a lot of noise that shows up in the high-frequency. To slice the audio, the amplitude should gradually drop off near the edge of a frame.
![image](https://user-images.githubusercontent.com/48405411/145718673-073cd1b1-de5a-46aa-a8b2-382d4b39acc7.png)


Let’s say w is the window applied to the original audio clip in the time domain.
![image](https://user-images.githubusercontent.com/48405411/145718670-9d6f09f5-948c-4189-a8c7-87c6dde2d8b2.png)


#### 3.4.1.1.2 Mel filterbank
The equipment measurements are not the same as our hearing perception. For humans, the perceived loudness changes according to frequency. Also, perceived frequency resolution decreases as frequency increases. i.e. humans are less sensitive to higher frequencies. The diagram on the left indicates how the Mel scale maps the measured frequency to that we perceive in the context of frequency resolution.
![image](https://user-images.githubusercontent.com/48405411/145718646-acbee18c-6716-47e9-a977-7d295fd4d709.png)


All these mappings are non-linear. In feature extraction, we apply triangular band-pass filters to convert the frequency information to mimic what a human perceived.

![image](https://user-images.githubusercontent.com/48405411/145718639-7d76a12f-a862-4b61-a261-052b19b4bd69.png)


First, we square the output of the DFT. This reflects the power of the speech at each frequency (x[k]²) and we call it the DFT power spectrum. We apply these triangular Mel-scale filter banks to transform it to Mel-scale power spectrum. The output for each Mel-scale power spectrum slot represents the energy from a number of frequency bands that it covers. This mapping is called the Mel Binning. The precise equations for slot m will be:
![image](https://user-images.githubusercontent.com/48405411/145718636-6354f65a-e07f-4166-bba8-1554b38b7c82.png)


The Triangular bandpass is wider at the higher frequencies to reflect human hearing and has less sensitivity in high frequency. Specifically, it is linearly spaced below 1000 Hz and turns logarithmically afterward.
All these efforts try to mimic how the basilar membrane in our ear senses the vibration of sounds. The basilar membrane has about 15,000 hairs inside the cochlear at birth. The diagram below demonstrates the frequency response of those hairs. So the curve-shape response below is simply approximated by triangles in Mel filterbank.
![image](https://user-images.githubusercontent.com/48405411/145718632-2aea3997-7278-41a3-a8b5-e54c8221ff2a.png)

We imitate how our ears perceive sound through those hairs. In short, it is modeled by the triangular filters using Mel filtering banks.
![image](https://user-images.githubusercontent.com/48405411/145718624-0f974988-150d-430c-9013-9e403ca9f04a.png)


##### 3.4.1.1.3 Log
Mel filterbank outputs a power spectrum. Humans are less sensitive to small energy changes at high energy than small changes at a low energy level. In fact, it is logarithmic. So our next step will take the log out of the output of the Mel filterbank. This also reduces the acoustic variants that are not significant for speech recognition. Next, we need to address two more requirements. First, we need to remove the F0 information (the pitch) and make the extracted features independent of others.

##### 3.4.1.2 Mel-spectrogram


![image](https://user-images.githubusercontent.com/48405411/145718618-65582d47-02da-4c4a-a6f4-97ee99e4ad5f.png)

        Fig:Computing of Mel-spectrogram 
![image](https://user-images.githubusercontent.com/48405411/145718613-028d79f7-3dbc-4d56-a6f5-e42e057eb4e8.png)

      Fig:Mel Spectrogram After taking log of filter bank energy
      
#####  3.4.1.3 Mel Frequency Cepstral Coefficient
MFCCs for inputs to speech Detection Neural Network
MFCCs are commonly derived as follows:
Take the Fourier transform of (a windowed excerpt of) a signal.
Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows.
Take the logs of the powers at each of the mel frequencies.
Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
The MFCCs are the amplitudes of the resulting spectrum.

![image](https://user-images.githubusercontent.com/48405411/145718609-dbecd989-190a-4c35-868e-bb451ddb866e.png)

      Fig:MFCC After Discrete Cosine Transform (DCT)

##### 3.4.1.4  Cepstrum Analysis Power Spectrum and Delta-Cepstrum 
The idea of the cepstrum is that the above spectrum, in log scale, can be treated as a periodic signal itself. And in fact its periodicity, in this case, is linked to the spacing between the different peaks.
![image](https://user-images.githubusercontent.com/48405411/145718600-3ab4c7cc-4447-4f03-afef-f4be818d8195.png)

To study cepstrum analysis of a signal :-
<strong> A. we first create a sinusoidal signal </strong> 

![image](https://user-images.githubusercontent.com/48405411/145718591-2f6e24b8-62eb-4354-9867-a26767a15db6.png)

<strong> B. we go to Fourier space, which we see the same signal to be full of peaks.As expected, we find 20 harmonics in the spectrum </strong> 

![image](https://user-images.githubusercontent.com/48405411/145718584-9dd7f93d-61f8-47e8-ae84-9b7b74f0f7fd.png)

<strong> C. Then we compute and visualize the cepstrum of this spectrum.
NOTE:  we expect a peak at a frequency of 440 Hz, hence a quefrency of 1/440 s which is approximately 0.0022 seconds.</strong> 

![image](https://user-images.githubusercontent.com/48405411/145718573-d114ec98-edeb-4bfe-baa2-7a0ecd914f27.png)

The cepstrum is defined as the power spectrum of the logarithmic power spectrum (i.e. in dB amplitude form), and is thus related to the autocorrelation function, which can be obtained by inverse Fourier transformation of the power spectrum with linear ordinates. In other applications such as speech analysis, the advantage is perhaps more that multiplicative relationships in the spectrum (e.g. by transfer functions) become additive on taking logarithms, and this additive relationship is maintained by the further Fourier transformation, 
Cepstrum Analysis is a tool for the detection of periodicity in a frequency spectrum, and seems so far to have been used mainly in speech analysis for voice pitch determination and related questions. In that case the periodicity in the spectrum is given by the many harmonics of the fundamental voice frequency, but another form of periodicity which can also be detected by cepstrum analysis is the presence of sidebands spaced at equal intervals around one or a number of carrier frequencies. 
In an amplitude modulated signal, the carrier signal is modulated by the baseband signal and the resulting modulated signal consists of sidebands along with carrier frequency. Sideband of an AM signal is the part which contains the information signal. The carrier frequency fc does not contain any information of the baseband signal. 
![image](https://user-images.githubusercontent.com/48405411/145718564-60b20bf1-0a33-4ac3-a673-8599cb96b8e6.png)

![image](https://user-images.githubusercontent.com/48405411/145718560-5d8ff8cd-93d6-4323-b074-9ef3359fc991.png)

For a short-time cepstral sequence C[n], the delta-cepstral features are typically defined as

        D[n] = C[n+ m] −C[n−m]  ………………………… ….. (1)
        
where n is the index of the analysis frames and in practice m is approximately 2 or 3. Similarly, double-delta cepstral features are defined in terms of a subsequent delta-operation on the delta - cepstral features
                    
# 4. Models 
                    
## 4.1 Study On Model 
A multilayer perceptron (MLP) is a class of a feedforward artificial neural network (ANN). MLPs models are the most basic deep neural network, which is composed of a series of fully connected layers. Today, MLP machine learning methods can be used to overcome the requirement of high computing power required by modern deep learning architectures.
Each new layer is a set of nonlinear functions of a weighted sum of all outputs (fully connected) from the prior one.
![image](https://user-images.githubusercontent.com/48405411/145718551-913ad04a-31f1-4e7c-a74a-6245ca36f5d2.png)

## 4.2 Implementation 

The next steps involve shuffling the data, splitting into train and test and then building a model to train our data.
We built a Multi Perceptron model. The MLP is not suitable as it gives us low accuracy. As our project is a classification problem where we categorize the different emotions, CNN would work best for us.
     
MLP Model: The MLP model we created had a very low validation accuracy of around 25% with 8 layers, softmax function at the output, batch size of 32 and 550 epochs.                   

Analysis:
![image](https://user-images.githubusercontent.com/48405411/145718524-d40c1122-3004-4ca1-b14e-c19f99697f3b.png)

# 5. Conclusion
After building numerous different models, we have found our best CNN model for our emotion classification problem. We achieved a validation accuracy of 25% with our existing model. Our model could perform better if we have more data to work on. What’s more surprising is that the model performed excellently when distinguishing between a males and female voice. We can also see above how the model predicted against the actual values. In the future we could build a sequence to sequence model to generate voice based on different emotions. E.g. A happy voice, A surprised one etc.

# 6. Key Takeaway
    Emotions are subjective and it is hard to notate them.
    We should define the emotions that are suitable for our own project objective.
    Do not always trust the content from GitHub even though it has lots of stars.
    Be aware of the data splitting.
    Exploratory Data Analysis always grants us good insight, and you have to be patient when you work on audio data!
    Deciding the input for your model: a sentence, a recording or an utterance?
    Lack of data is a crucial factor to achieve success in SER, however, it is complex and very expensive to build a good speech emotion dataset.
    Simplified your model when you lack data.

# 7. Further Improvement Possible 
    We only selected the first 3 seconds to be the input data since it would reduce the dimension, the original notebook used 2.5 sec only. We would like to use the full length of the audio to do the experiment.
    Preprocess the data like cropping silence voice, normalize the length by zero padding, etc.
    Experiment the Recurrent Neural Network approach on this topic.
    Experiment the Convolutional Neural Network approach on this topic.
    Experiment the LSTM Neural Network approach on this topic.
    Experiment the Deep Recurrent Neural Network approach on this topic.
