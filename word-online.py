import numpy as np
import gensim
import time  # Added for the delay parameter

from collections import deque
from tqdm import tqdm
from scipy.special import softmax
from sklearn.linear_model import SGDClassifier


# Sample text 
text = """Hello world, this is an online learning example with word embeddings.
          It learns words and generates text incrementally using an SGD classifier."""

def debug_print(x):
    print(f"{x}")

# Tokenization (simple space-based)
words = text.lower().split()
vocab = sorted(set(words))
vocab.append("<UNK>")  # Add unknown token for OOV words

# Train Word2Vec model (or load pretrained embeddings)
embedding_dim = 50  # Change to 100/300 if using a larger model
word2vec = gensim.models.Word2Vec([words], vector_size=embedding_dim, window=5, min_count=1, sg=0)

# Create word-to-index mapping
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# Hyperparameters
context_size = 12  # Default 10, Words used for prediction context
learning_rate = 0.005
epochs = 10

# Prepare training data
X_train, y_train = [], []

for i in tqdm(range(len(words) - context_size)):
    context = words[i:i + context_size]
    target = words[i + context_size]
    # Convert context words to embeddings
    context_embedding = np.concatenate([word2vec.wv[word] for word in context])
    X_train.append(context_embedding)
    y_train.append(word_to_idx[target])

X_train, y_train = np.array(X_train), np.array(y_train)

# Initialize SGD-based classifier
clf = SGDClassifier(loss="hinge", max_iter=1, learning_rate="constant", eta0=learning_rate)

# Online training (stochastic updates, multiple passes)
for epoch in tqdm(range(epochs)):
    for i in range(len(X_train)):
        clf.partial_fit([X_train[i]], [y_train[i]], classes=np.arange(len(vocab)))

# 🔥 **Softmax function for probability scaling**
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Stability trick
    return exp_logits / np.sum(exp_logits)


def sample_from_logits(logits, k=5, temperature=1.0, random_seed=123):
    """ Applies Top-K sampling & Temperature scaling """
    logits = np.array(logits) / temperature  # Apply temperature scaling
    probs = softmax(logits)  # Convert logits to probabilities
    # Select top-K indices
    top_k_indices = np.argsort(probs)[-k:]
    top_k_probs = probs[top_k_indices]
    top_k_probs /= top_k_probs.sum()  # Normalize
    # Sample from Top-K distribution
    np.random.seed(random_seed)
    return np.random.choice(top_k_indices, p=top_k_probs)


def generate_text(seed="this is", length=20, k=5, temperature=1.0, random_state=123, delay=3):
    seed_words = seed.lower().split()

    # Ensure context has `context_size` words (pad with zero vectors if needed)
    while len(seed_words) < context_size:
        seed_words.insert(0, "<PAD>")

    context = deque(
        [word_to_idx[word] if word in word_to_idx else -1 for word in seed_words[-context_size:]],
        maxlen=context_size
    )

    generated = seed
    previous_word = seed

    for _ in range(length):
        # Generate embeddings, use a zero vector if word is missing
        context_embedding = np.concatenate([
            word2vec.wv[idx_to_word[idx]] if idx in idx_to_word else np.zeros(embedding_dim)
            for idx in context
        ])
        logits = clf.decision_function([context_embedding])[0]  # Get raw scores
        # Sample next word using Top-K & Temperature scaling
        pred_idx = sample_from_logits(logits, k=k, temperature=temperature)
        next_word = idx_to_word.get(pred_idx, "<PAD>")
        
        print(f"Generating next word: {next_word}")  # Added this line
        time.sleep(delay)  # Added this line
        
        if previous_word[-1] == "." and previous_word[-1] != "" and previous_word[-1] != seed:
          generated += " " + next_word.capitalize()
        else: 
          generated += " " + next_word
        previous_word = next_word
        context.append(pred_idx)

    return generated

# 🔥 Generate text
print("\n\n Generated Text:")
seed = "This is a"
print(seed)
print(generate_text(seed, length=12, k=1, delay=5)) # delay seconds for next word generation, optimal for delay=0 seconds 