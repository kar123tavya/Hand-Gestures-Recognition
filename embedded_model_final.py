
import json
import numpy as np
import tensorflow as tf
import cv2
from sklearn.preprocessing import LabelEncoder

# Embedded landmarks data
landmarks_data = {
    "closed_fist/1.jpg": {
        "class": "closed_fist",
        "landmarks": [
            [0.0, 0.0, 0.0],
            [0.2857, -0.1175, -0.1635],
            [0.5724, -0.3704, -0.2461],
            [0.5509, -0.6208, -0.3083]
        ],
        "reference_point": "wrist",
        "scale_reference": "wrist_to_middle_mcp"
    },
    "closed_fist/10.jpg": {
        "class": "closed_fist",
        "landmarks": [
            [0.0, 0.0, 0.0],
            [0.2893, -0.1038, -0.2004],
            [0.5687, -0.3456, -0.3033],
            [0.5581, -0.6253, -0.3846]
        ],
        "reference_point": "wrist",
        "scale_reference": "wrist_to_middle_mcp"
    }
    # (Include the full dataset here)
}

# Model training function
def train_model(model_output_path, encoder_output_path):
    X = []
    y = []

    for image_path, data in landmarks_data.items():
        landmarks_flat = np.array(data['landmarks']).flatten()
        X.append(landmarks_flat)
        y.append(data['class'])

    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_encoded, epochs=50, batch_size=8, validation_split=0.2)

    model.save(model_output_path)
    np.save(encoder_output_path, label_encoder.classes_)

# Real-time gesture detection
def real_time_detection(model_path, encoder_path):
    model = tf.keras.models.load_model(model_path)
    label_encoder = np.load(encoder_path, allow_pickle=True)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for dummy landmark detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        landmarks = resized.flatten().reshape(1, -1)

        if landmarks.shape[1] != model.input_shape[1]:
            landmarks = np.resize(landmarks, (1, model.input_shape[1]))

        predictions = model.predict(landmarks)
        predicted_class = label_encoder[np.argmax(predictions)]

        cv2.putText(frame, f'Gesture: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-Time Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
train_model('gesture_model.h5', 'label_encoder.npy')
real_time_detection('gesture_model.h5', 'label_encoder.npy')
