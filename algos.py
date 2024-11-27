
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

input_file = 'input_audio.wav'  # Название входного файла
output_file = 'processed_audio.wav' # Результат

# Загрузка звука
y, sr = librosa.load(input_file, sr=None)  # Чтение с исходной частотой дискретизации

#  Эквализация (усиление низких частот)
def apply_equalization(signal, sr):
    low_freq = librosa.effects.preemphasis(signal, coef=0.97)  # Усиление низких частот
    return low_freq

#  Реверберация
def apply_reverb(signal, decay=0.3):
    impulse_response = np.zeros(5000) #Добавление реверберации с простым импульсным откликом
    impulse_response[0] = 1  # Начальный импульс
    impulse_response[1000:] = decay
    return np.convolve(signal, impulse_response, mode='full')[:len(signal)]

#  Дилей (задержка сигнала)
def apply_delay(signal, delay_seconds=0.2, sr=44100, decay=0.5):
    delay_samples = int(delay_seconds * sr)
    delayed_signal = np.zeros_like(signal)
    delayed_signal[delay_samples:] = signal[:-delay_samples] * decay
    return signal + delayed_signal

# Изменение высоты тона
def pitch_shift(signal, sr, n_steps):
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)

# Применение функций
equalized_signal = apply_equalization(y, sr)
reverb_signal = apply_reverb(equalized_signal)
delayed_signal = apply_delay(equalized_signal, delay_seconds=0.25, sr=sr, decay=0.4)
final_signal = pitch_shift(delayed_signal, sr, n_steps=2)  # Сдвиг на 2 полутона вверх

# Нормализация результата
final_signal = final_signal / np.max(np.abs(final_signal))  # Приведение амплитуды в [-1, 1]

# Сохранение результата
sf.write(output_file, final_signal, sr)
print(f"Файл сохранён как: {output_file}")

# Визуализация
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.5, label='Исходный сигнал')
plt.title("Исходный сигнал")
plt.legend()
plt.subplot(2, 1, 2)
librosa.display.waveshow(final_signal, sr=sr, alpha=0.5, color='r', label='Обработанный сигнал')
plt.title("Обработанный сигнал")
plt.legend()
plt.tight_layout()
plt.show()




