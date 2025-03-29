import pandas as pd
import numpy as np
import pennylane as qml
import tensorflow as tf
from pennylane import numpy as pnp
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D, MultiHeadAttention, Add, concatenate
from tensorflow.keras.models import Model
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


data = pd.read_csv('scaled_dataset.csv')
labels = tf.keras.utils.to_categorical(data['ec class'] - 1)

# Normalize QM/MM descriptors
augmented_features = data[['scf total energy', 'maximum gradient', 'nuclear repulsion energy']].values
scaler = StandardScaler()
augmented_features = scaler.fit_transform(augmented_features)

# Tokenize enzyme sequences
tokenizer_smiles = Tokenizer(char_level=True, filters='', lower=False)
tokenizer_smiles.fit_on_texts(data['reconstructed_sequences'].tolist())
smiles_sequences = tokenizer_smiles.texts_to_sequences(data['reconstructed_sequences'].tolist())
smiles_sequences = pad_sequences(smiles_sequences, padding='post', maxlen=1024)

# Amplitude encoding (zero-padding to 2^n dim)
def amplitude_encoding(features):
    target_dim = 2 ** int(np.ceil(np.log2(len(features))))
    padded = np.zeros(target_dim)
    padded[:len(features)] = features
    norm = np.linalg.norm(padded)
    return padded / norm if norm > 0 else padded

quantum_encoded_features = np.apply_along_axis(amplitude_encoding, 1, augmented_features)
n_qubits = int(np.log2(quantum_encoded_features.shape[1]))


dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="autograd")
def quantum_circuit(inputs):
    qml.MottonenStatePreparation(inputs, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

# Simulate quantum feature encoding
print(f"Encoding quantum features on PennyLane simulator using {n_qubits} qubits...")
quantum_features = np.array([quantum_circuit(f) for f in quantum_encoded_features])


smiles_input = Input(shape=(1024,), dtype='int32', name='SMILES_Input')
quantum_input = Input(shape=(quantum_features.shape[1],), dtype='float32', name='Quantum_Input')
smiles_embedding = Embedding(input_dim=len(tokenizer_smiles.word_index) + 1, output_dim=1024)(smiles_input)


def transformer_block(x, num_heads, ff_dim):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attention_output = Dropout(0.2)(attention_output)
    attention_output = Add()([x, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
    feedforward_output = Dense(ff_dim * 4, activation='relu')(attention_output)
    feedforward_output = Dropout(0.2)(feedforward_output)
    feedforward_output = Dense(ff_dim)(feedforward_output)
    
    final_output = Add()([attention_output, feedforward_output])
    final_output = LayerNormalization(epsilon=1e-6)(final_output)
    return final_output


x = smiles_embedding
for _ in range(6):
    x = transformer_block(x, num_heads=16, ff_dim=1024)

pooled_output = GlobalAveragePooling1D()(x)


quantum_dense = Dense(1024, activation='relu')(quantum_input)
quantum_dense = Dense(512, activation='relu')(quantum_dense)


combined = concatenate([pooled_output, quantum_dense])
dense = Dense(1024, activation='relu')(combined)
dense = Dropout(0.4)(dense)
dense = Dense(1024, activation='relu')(dense)
dense = Dropout(0.4)(dense)
dense = Dense(512, activation='relu')(dense)
output = Dense(labels.shape[1], activation='softmax')(dense)

# Compile model
qvt_model = Model(inputs=[smiles_input, quantum_input], outputs=output)
qvt_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# Model overview
qvt_model.summary()

# Train
qvt_model.fit([smiles_sequences, quantum_features], labels,
              epochs=50, batch_size=16, validation_split=0.2)
