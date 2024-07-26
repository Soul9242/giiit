import heapq
from collections import Counter
import numpy as np
from PIL import Image
import pickle
# Node class for Huffman tree
class Node:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None
# Comparison operators for priority queue
    def __lt__(self, other):
        return self.freq < other.freq
# Build Huffman tree
def build_huffman_tree(freqs):
    heap = [Node(value, freq) for value, freq in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]
# Generate Huffman codes
def generate_codes(node, prefix='', codebook={}):
    if node.value is not None:
        codebook[node.value] = prefix
    else:
        generate_codes(node.left, prefix + '0', codebook)
        generate_codes(node.right, prefix + '1', codebook)
    return codebook
# Encode image
def encode_image(image, codebook):
    encoded = ''.join(codebook[pixel] for pixel in image.flatten())
    return encoded
# Decode image
def decode_image(encoded, root, shape):
    decoded = []
    node = root
    for bit in encoded:
        node = node.left if bit == '0' else node.right
        if node.value is not None:
            decoded.append(node.value)
            node = root
    return np.array(decoded).reshape(shape)
# Save the encoded data and codebook
def save_encoded_data(encoded, codebook, shape, filename):
    with open(filename, 'wb') as f:
        pickle.dump((encoded, codebook, shape), f)
# Load the encoded data and codebook
def load_encoded_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
# Main function
def main():
    mode = input("Enter 'encode' to encode an image or 'decode' to decode an image: ").strip().lower()
if mode == 'encode':
        input_image_path = r'input_image.jpg'  # Specify your image path
        encoded_data_path = r'encoded_image.pkl'  # Specify your encoded data path
        
        # Debug: Print the path being used
        print(f"Trying to open image at: {input_image_path}")

        # Open and convert image to grayscale
        try:
            image = Image.open(input_image_path).convert('L')
            image_data = np.array(image)
# Calculate frequencies
            freqs = Counter(image_data.flatten())
# Build Huffman tree and generate codes
            huffman_tree = build_huffman_tree(freqs)
            codebook = generate_codes(huffman_tree)
# Encode image
            encoded_image = encode_image(image_data, codebook)
# Print length of encoded image to verify encoding
            print(f"Length of encoded image: {len(encoded_image)} bits")
# Save encoded image data and codebook
            save_encoded_data(encoded_image, codebook, image_data.shape, en-coed_data_path)
            print("Encoded image data saved successfully.")
except FileNotFoundError as e:
            print(f"File not found: {e}")
except Exception as e:
            print(f"An error occurred: {e}")
elif mode == 'decode':
        encoded_data_path = r'encoded_image.pkl'  # Specify your encoded data path
        output_image_path = r'decoded_image.png'  # Specify your output image path
try:
            encoded_image, codebook, shape = load_encoded_data(encoded_data_path)
# Print some details to verify loading
            print(f"Length of encoded image: {len(encoded_image)} bits")
            print(f"Image shape: {shape}")
# Rebuild Huffman tree
            freqs = Counter({k: len(v) for k, v in codebook.items()})
            huffman_tree = build_huffman_tree(freqs)
# Decode image
            decoded_image_data = decode_image(encoded_image, huffman_tree, shape)
            decoded_image = Image.fromarray(decoded_image_data)
# Save the decoded image
            decoded_image.save(output_image_path)
            print("Decoded image saved successfully.")
except FileNotFoundError as e:
            print(f"File not found: {e}")
except Exception as e:
            print(f"An error occurred: {e}")
else:
        print("Invalid mode. Please enter 'encode' or 'decode'.")
if __name__ == "__main__":
    main()
