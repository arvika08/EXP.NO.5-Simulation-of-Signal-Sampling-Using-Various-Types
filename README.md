# EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as
    i) Ideal Sampling
    ii) Natural Sampling
    iii) Flat Top Sampling

## EXP. NO. 1: Simulation of Signal Sampling Using Various Types

## AIM
To simulate signal sampling using three different methods:

Ideal Sampling

Natural Sampling

Flat-Top Sampling

## SOFTWARE REQUIRED
Python with libraries: NumPy, Matplotlib

MATLAB

## ALGORITHMS
i) Ideal Sampling

Define the input signal x(t).

Set the sampling interval Ts.

Generate an impulse train at intervals of Ts.

Multiply x(t) with the impulse train to get the sampled output.

ii) Natural Sampling

Define the input signal x(t).

Set the sampling period Ts.

Set the pulse width τ (where τ < Ts).

Generate a periodic pulse train with width τ.

Multiply x(t) by the pulse train.

iii) Flat-Top Sampling

Define the input signal x(t).

Set the sampling period Ts.

Set the pulse width τ.

Generate a rectangular pulse train with width τ.

Multiply x(t) by the pulse train.

Hold the sampled value constant for duration τ.



## Ideal sampling code:

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import resample

fs = 100

t = np.arange(0, 1, 1/fs) 

f = 5

signal = np.sin(2 * np.pi * f * t)

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal')

plt.title('Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

t_sampled = np.arange(0, 1, 1/fs)

signal_sampled = np.sin(2 * np.pi * f * t_sampled)

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')

plt.title('Sampling of Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

reconstructed_signal = resample(signal_sampled, len(t))

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')

plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

## Output:

![impulse signal](https://github.com/user-attachments/assets/1ccec161-55db-4d0c-aedd-7de16a696bb2)


## Natural sampling code:

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter

fs = 1000  

T = 1  

t = np.arange(0, T, 1/fs)  

fm = 5 

message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50  # pulses per second

pulse_train = np.zeros_like(t)

pulse_width = int(fs / pulse_rate / 2)

for i in range(0, len(t), int(fs / pulse_rate)):

pulse_train[i:i+pulse_width] = 1

nat_signal = message_signal * pulse_train

sampled_signal = nat_signal[pulse_train == 1]

sample_times = t[pulse_train == 1]

reconstructed_signal = np.zeros_like(t)

for i, time in enumerate(sample_times):

    index = np.argmin(np.abs(t - time))
    
    reconstructed_signal[index:index+pulse_width] = sampled_signal[i]

def lowpass_filter(signal, cutoff, fs, order=5):

nyquist = 0.5 * fs

normal_cutoff = cutoff / nyquist

b, a = butter(order, normal_cutoff, btype='low', analog=False)

return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal,10, fs)

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)

plt.plot(t, message_signal, label='Original Message Signal')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 2)

plt.plot(t, pulse_train, label='Pulse Train')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 3)

plt.plot(t, nat_signal, label='Natural Sampling')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 4)

plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

## Output:

![natural sampling](https://github.com/user-attachments/assets/324941ea-284c-4a37-bd42-4f52c9a308ec)


## Flat top sampling code:

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import resample


fs = 1000                

f = 5                    

T = 1                    

ts = 1/fs                

t = np.arange(0, T, ts)   

x = np.sin(2 * np.pi * f * t)

fs_sampled = 50       

Ts = 1 / fs_sampled      

tau = Ts / 4               

samples_idx = np.arange(0, len(t), int(Ts / ts))

pulse_train = np.zeros_like(t)

for idx in samples_idx:

    pulse_train[idx:idx + int(tau / ts)] = 1

flat_top = x * pulse_train

num_points = len(t)

reconstructed = resample(flat_top, num_points)

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)

plt.plot(t, x, label='Original Message Signal')

plt.title('Original Message Signal')

plt.grid(True)

plt.legend()

plt.subplot(4, 1, 2)

plt.plot(t, pulse_train, label='Pulse Train')

plt.title('Pulse Train')

plt.grid(True)

plt.legend()

plt.subplot(4, 1, 3)

plt.plot(t, flat_top, color='orange', label='Flat-Top Sampled Signal')

plt.title('Flat-Top Sampled Signal')

plt.grid(True)

plt.legend()

plt.subplot(4, 1, 4)

plt.plot(t, reconstructed, color='green', label='Reconstructed Signal')

plt.title('Reconstructed Signal')

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()

## Output:

![flat top sampling](https://github.com/user-attachments/assets/7fe64abc-0fe8-49c8-b8c0-d415fc1af736)

## RESULT / CONCLUSION
The simulation successfully demonstrates three types of signal sampling.

Ideal Sampling gives exact values at discrete time intervals but is idealized.

Natural Sampling reflects the analog nature by allowing pulse width but may distort shape.

Flat-Top Sampling provides practical representation for ADC systems by holding sampled values constant.

This helps understand how real-world signals are digitized in electronics and communication systems.
