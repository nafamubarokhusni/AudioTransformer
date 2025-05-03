import os
import numpy as np # type: ignore
import librosa # type: ignore
import tensorflow as tf # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# Constants (define these based on your project requirements)
N_MFCC = 13
SEQ_LENGTH = 100
BATCH_SIZE = 32
NUM_FEATURE = 8
SOUND_FOLDER = "path_to_sound_folder"

def segment_cough_sound(signal, sr, cough_threshold=0.05, min_cough_duration=0.1, padding=0.05):
    hop_length = int(min_cough_duration * sr)
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    energy = librosa.feature.rms(y=signal, hop_length=hop_length)[0]

    # Normalize the energy values
    normalized_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

    # Set the energy threshold for event detection
    cough_threshold = np.max(normalized_energy) * cough_threshold
    min_cough_samples = round(sr * min_cough_duration)

    # Find the cough segments
    cough_segments = []
    event_start = None

    for i, value in enumerate(normalized_energy):
        if value >= cough_threshold:
            if event_start is None:
                event_start = i * hop_length
        else:
            if event_start is not None:
                cough_duration = i * hop_length - event_start
                if cough_duration >= min_cough_samples:
                    event_end = i * hop_length + int(padding * sr)
                    event_start -= int(padding * sr)
                    event_start = max(event_start, 0)
                    cough_segments.append(signal[event_start: event_end + 1])
                event_start = None

    return cough_segments

def extract_mfcc(file_path, n_mfcc=N_MFCC, target_length=SEQ_LENGTH):
    try:
        audio, sr = librosa.load(file_path)
        cough_segments = segment_cough_sound(audio, sr)

        # If no cough segments found, use full audio
        if not cough_segments:
            print(f"No cough segments detected for {file_path}, using full signal.")
            segmented_audio = audio
        else:
            segmented_audio = np.concatenate(cough_segments)

        # Extract MFCC from the segmented audio
        mfcc = librosa.feature.mfcc(y=segmented_audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T  # shape: (time_steps, n_mfcc)

        # Pad or trim to target length
        if len(mfcc) > target_length:
            mfcc = mfcc[:target_length]
        elif len(mfcc) < target_length:
            pad_width = target_length - len(mfcc)
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

        return tf.convert_to_tensor(mfcc, dtype=tf.float32)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return tf.zeros((target_length, n_mfcc), dtype=tf.float32)

def preprocess_features(row):
    age = row["age"] / 100.0
    pack_years = row["packYears"] / 100.0
    gender = row["gender"]
    tb_contact_history = row["tbContactHistory"]
    wheezing_history = row["wheezingHistory"]
    phlegm_cough = row["phlegmCough"]
    family_asthma_history = row["familyAsthmaHistory"]
    fever_history = row["feverHistory"]

    features = [
        age, gender, pack_years, tb_contact_history, wheezing_history,
        phlegm_cough, family_asthma_history, fever_history
    ]
    features = np.array(features).reshape(-1, 1)
    return features

def process_row(row):
    candidate_id = row["candidateID"]
    audio_path = os.path.join(SOUND_FOLDER, str(candidate_id), "cough.wav")
    mfcc = extract_mfcc(audio_path)
    features = preprocess_features(row)
    label = row["disease"]
    label = tf.one_hot(label, depth=3)
    return mfcc, features, label

def data_generator(data):
    for _, row in data.iterrows():
        mfcc, features, label = process_row(row)
        yield mfcc, features, label

def create_dataset(data, batch_size=BATCH_SIZE):
    output_signature = (
        tf.TensorSpec(shape=(SEQ_LENGTH, N_MFCC), dtype=tf.float32),
        tf.TensorSpec(shape=(NUM_FEATURE, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(3,), dtype=tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(data),
        output_signature=output_signature
    )
    return dataset.batch(batch_size).shuffle(256).prefetch(tf.data.AUTOTUNE)

def split_data(data, test_size=0.2, random_state=42):
    train_data, valid_data = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data['disease']
    )
    return train_data, valid_data