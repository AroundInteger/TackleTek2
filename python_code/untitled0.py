#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:32:15 2024

@author: Rowan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def generate_signal(t, frequencies, amplitudes):
    return np.sum([a * np.sin(2 * np.pi * f * t) for f, a in zip(frequencies, amplitudes)], axis=0)

def update(val):
    f1, f2, f3 = slider_f1.val, slider_f2.val, slider_f3.val
    a1, a2, a3 = slider_a1.val, slider_a2.val, slider_a3.val
    
    y = generate_signal(t, [f1, f2, f3], [a1, a2, a3])
    line_signal.set_ydata(y)
    
    Y = np.fft.fft(y)
    line_magnitude.set_ydata(np.abs(Y[:N//2]) * 2 / N)
    
    fig.canvas.draw_idle()

# Time domain
N = 1000
T = 1.0
t = np.linspace(0.0, T, N, endpoint=False)

# Initial frequencies and amplitudes
initial_freq = [3.0, 7.0, 10.0]
initial_amp = [1.0, 0.5, 0.3]

y = generate_signal(t, initial_freq, initial_amp)

# Frequency domain
Y = np.fft.fft(y)
xf = np.fft.fftfreq(N, T / N)[:N//2]

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)

# Plot time domain signal
line_signal, = ax1.plot(t, y)
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.set_title('Time Domain Signal')

# Plot frequency domain magnitude
line_magnitude, = ax2.plot(xf, np.abs(Y[:N//2]) * 2 / N)
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Magnitude')
ax2.set_title('Frequency Domain Magnitude')

# Add sliders for frequencies and amplitudes
ax_f1 = plt.axes([0.1, 0.25, 0.8, 0.03])
ax_f2 = plt.axes([0.1, 0.20, 0.8, 0.03])
ax_f3 = plt.axes([0.1, 0.15, 0.8, 0.03])
ax_a1 = plt.axes([0.1, 0.10, 0.8, 0.03])
ax_a2 = plt.axes([0.1, 0.05, 0.8, 0.03])
ax_a3 = plt.axes([0.1, 0.00, 0.8, 0.03])

slider_f1 = Slider(ax_f1, 'Freq 1', 0.1, 20.0, valinit=initial_freq[0])
slider_f2 = Slider(ax_f2, 'Freq 2', 0.1, 20.0, valinit=initial_freq[1])
slider_f3 = Slider(ax_f3, 'Freq 3', 0.1, 20.0, valinit=initial_freq[2])
slider_a1 = Slider(ax_a1, 'Amp 1', 0.0, 2.0, valinit=initial_amp[0])
slider_a2 = Slider(ax_a2, 'Amp 2', 0.0, 2.0, valinit=initial_amp[1])
slider_a3 = Slider(ax_a3, 'Amp 3', 0.0, 2.0, valinit=initial_amp[2])

slider_f1.on_changed(update)
slider_f2.on_changed(update)
slider_f3.on_changed(update)
slider_a1.on_changed(update)
slider_a2.on_changed(update)
slider_a3.on_changed(update)

plt.show()