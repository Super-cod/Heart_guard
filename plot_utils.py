# plot_utils.py
import matplotlib.pyplot as plt

def plot_ecg_signal(signal):
    plt.figure(figsize=(15, 7))
    plt.plot(signal)
    plt.title('ECG Signal')
    plt.grid(True)
    plt.show()