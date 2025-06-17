import random
import re
import argparse

def preprocess_text(text):
    """
    Normalize text: lowercase, remove extra spaces, and preserve sentence punctuation.
    """
    text = text.lower()
    text = re.sub(r'([.!?])', r' \1', text)  # Space out punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def build_markov_chain(text, order=1):
    """
    Build a Markov chain from the preprocessed text.
    """
    words = text.split()
    chain = {}
    
    for i in range(len(words) - order):
        state = tuple(words[i:i + order])
        next_word = words[i + order]
        chain.setdefault(state, []).append(next_word)
    
    return chain

def generate_text(chain, length=50, order=1, seed=None):
    """
    Generate text using the Markov chain with optional seeding.
    """
    if seed is not None:
        random.seed(seed)
    
    state = random.choice(list(chain.keys()))
    output = list(state)

    for _ in range(length - order):
        next_words = chain.get(state)
        
        if not next_words:
            state = random.choice(list(chain.keys()))
            output.extend(list(state))
            continue
        
        next_word = random.choice(next_words)
        output.append(next_word)
        state = tuple(output[-order:])
        
        # Optional early stopping on punctuation (if sentence ends)
        if next_word in ('.', '?', '!') and len(output) >= 10:
            break
    
    return ' '.join(output).replace(' .', '.').replace(' ?', '?').replace(' !', '!')

def main():
    parser = argparse.ArgumentParser(description="Enhanced Markov Chain Text Generator")
    parser.add_argument('--order', type=int, default=2, help="Order of the Markov chain (default: 2)")
    parser.add_argument('--length', type=int, default=50, help="Length of generated text (default: 50 words)")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Sample text
    sample_text = (
        "I love to explore creative text generation with Markov chains. "
        "They are an interesting way to generate surprising and fun outputs. "
        "With Markov models, the past influences the future in unpredictable ways."
    )

    clean_text = preprocess_text(sample_text)
    chain = build_markov_chain(clean_text, order=args.order)
    generated = generate_text(chain, length=args.length, order=args.order, seed=args.seed)
    
    print("\nGenerated Text:")
    print(generated)

if __name__ == "__main__":
    main()
