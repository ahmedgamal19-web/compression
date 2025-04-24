import streamlit as st
import plotly.graph_objects as go
import math
from collections import Counter
import ast
import heapq
from collections import Counter
import math
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import base64
import warnings
from io import StringIO
import ast
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
import time
import math
from collections import Counter
from PIL import Image
import plotly.express as px
import io



st.title("Compression Algorithms")
tabs = st.tabs(["RLE Algorithm", "Huffmancode Algorith", 'Golombcode Algorithm',"Arthimitic Algorithm","LZW Lgorithm","Uniform Algorithm(lossy)"])


# RLE encoding and decoding functions
def rle_encode(data):
    """Encodes data using Run-Length Encoding (RLE)."""
    encoding = ""
    i = 0
    while i < len(data):
        count = 1
        while i + 1 < len(data) and data[i] == data[i + 1]:
            count += 1
            i += 1
        encoding += data[i] + str(count)
        i += 1
    return encoding

def decode_tuples(encoded_data):
    
   decoded = ""
    
   if isinstance(encoded_data[0][0], int):
        # Decode binary tuples, e.g., [(1, 2), (0, 3), (1, 7)] -> "1100001111111"
        for digit, freq in encoded_data:
            decoded += str(digit) * freq  # Repeat the digit based on frequency
   elif isinstance(encoded_data[0][0], str):
        # Decode character tuples, e.g., [('a', 3), ('b', 4), ('d', 2), ('c', 3), ('a', 2)] -> "aaabbbbddccaa"
        for char, freq in encoded_data:
            decoded += char * freq  # Repeat the character based on frequency
   else:
        raise ValueError("Invalid input format. The first element should be an integer or a string.")

   return decoded

def decode_string(encoded_str):
    decoded_str = ""
    i = 0
    while i < len(encoded_str):
        char = encoded_str[i]
        i += 1
        num = ""
        while i < len(encoded_str) and encoded_str[i].isdigit():
            num += encoded_str[i]
            i += 1
        decoded_str += char * int(num)
    return decoded_str


def is_binary(text):
    """Checks if the input text is binary (only contains '0' and '1')."""
    return all(char in '01' for char in text)

def calculate_bit_size(text, is_binary):
    """Calculates the original bit size based on the input type."""
    if is_binary:
        return len(text) * 1  # 1 bit per character for binary input
    else:
        return len(text) * 8  # 8 bits per character for regular text
    
def num_terms(text):
    
    result = []
    count = 1
    
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1
        else:
            result.append((text[i - 1], count))
            count = 1
            
    result.append((text[-1], count))
    
    # Find the maximum frequency
    max_freq_num = max(count for _, count in result)
    
    return  result,len(result), max_freq_num


def calculate_compressed_size(text, compressed_text, is_binary):
    """Calculates the compressed size in bits using RLE with specific encoding rules."""
    # Count occurrences of each character in the text
     # Bits needed for count
    results,num_termss,max_freq = num_terms(text)
    num_of_bits_for_count = math.ceil(math.log2(max_freq + 1))  # Number of unique characters/terms in the compressed text
    bits_per_char = 1 if is_binary else 8  # Determine bits per character based on input type
    compressed_size = num_termss * (bits_per_char + num_of_bits_for_count)  # Each term: bits for char + count bits
    return compressed_size,results
def calculate_CR(text, compressed_text, is_binary):
    """Calculates the Compression Ratio (CR) as a fraction."""
    original_size = calculate_bit_size(text, is_binary)
    compressed_size,results = calculate_compressed_size(text, compressed_text, is_binary)

    # Provide CR as a fraction (numerator and denominator)
    numerator = original_size
    denominator = compressed_size if compressed_size > 0 else 1  # Avoid division by zero
    return numerator, denominator,results


def compression_percentage(original, compressed):
    """Calculates compression percentage."""
    return 100 * (1-(compressed / original))

def plot_compression_gauge(percentage):
    """Plots a half-circle gauge to show compression percentage."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 50], 'color': "lightgray"},
                   {'range': [50, 100], 'color': "skyblue"}]},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(
        title={'text': "Compression Efficiency (%)"},
        height=400,
        margin=dict(t=40, b=0, l=0, r=0)
    )
    return fig

#huffman functions

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    heap = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def generate_huffman_codes(node, prefix="", huffman_dict={}):
    if node:
        if node.char is not None:
            huffman_dict[node.char] = prefix
        generate_huffman_codes(node.left, prefix + "0", huffman_dict)
        generate_huffman_codes(node.right, prefix + "1", huffman_dict)
    return huffman_dict

def encode_text(text, huffman_dict):
    return ''.join(huffman_dict[char] for char in text)

def decode_text(encoded_text, huffman_dict):
    reverse_dict = {v: k for k, v in huffman_dict.items()}
    
    decoded_text = ""
    buffer = ""
    for bit in encoded_text:
        buffer += bit
        if buffer in reverse_dict:
            decoded_text += reverse_dict[buffer]
            buffer = ""
    return decoded_text

def plot_tree(node, x=0, y=0, dx=2, dy=2, fig=None, ax=None):
    if fig is None:  # Create a new figure and axes if they are not provided
        fig, ax = plt.subplots(figsize=(15, 10))
    
    if node:
        label = f"{node.char or ''}\n{node.freq}"
        node_color = 'lightgreen' if node.char is not None else 'lightblue'
        ax.text(x, y, label, ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor=node_color, edgecolor='black'))

        if node.left:
            ax.plot([x, x - dx], [y, y - dy], color='blue', linewidth=1.5)
            plot_tree(node.left, x - dx, y - dy, dx / 1.5, dy, fig, ax)

        if node.right:
            ax.plot([x, x + dx], [y, y - dy], color='red', linewidth=1.5)
            plot_tree(node.right, x + dx, y - dy, dx / 1.5, dy, fig, ax)

    ax.axis('off')
    return fig  # Return the figure object for later use

def avg_l(frequency,huffman_dict):
    total_characters = sum(frequency.values())
    weighted_sum = sum(frequency[char] * len(huffman_dict[char]) for char in frequency)
    return weighted_sum / total_characters

def entropy(frequency):
    total_characters = sum(frequency.values())
    return -sum((freq / total_characters) * math.log2(freq / total_characters) for freq in frequency.values())

def encode_and_display_metrics(text):
    frequency = Counter(text)
    huffman_tree_root = build_huffman_tree(frequency)
    huffman_dict = generate_huffman_codes(huffman_tree_root)
    
    avg_length = avg_l(frequency, huffman_dict)
    entrop = entropy(frequency)
    efficiency = (entrop / avg_length) * 100
    
    encoded_text = encode_text(text, huffman_dict)
    
    return encoded_text, huffman_dict, avg_length, entrop, efficiency, huffman_tree_root

def file_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def display_gauge(value, label):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': label},
        gauge={'axis': {'range': [None, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

#Golombcode functions

def golomb_encode(number, m):
    quotient = number // m
    remainder = number % m

    # Encode the quotient in unary
    unary_code = '1' * quotient + '0'

    # Encode the remainder in truncated binary
    b = math.ceil(math.log2(m))
    if 2**b - m > remainder:
        binary_code = format(remainder, f'0{b - 1}b')
    else:
        binary_code = format(remainder + 2**b - m, f'0{b}b')

    # Final code word (unary part + binary part)
    code_word = unary_code + binary_code

    return quotient, remainder, unary_code, binary_code, code_word


def golomb_decode(code, m):
    # Decode the unary part (quotient)
    quotient = 0
    while code[quotient] == '1':
        quotient += 1
    code = code[quotient + 1:]  # Remove unary part and the zero separator

    # Decode the binary part (remainder)
    b = math.ceil(math.log2(m))
    cutoff = 2**b - m
    remainder_bits = b - 1 if len(code) >= b - 1 and int(code[:b - 1], 2) < cutoff else b
    remainder = int(code[:remainder_bits], 2)
    if remainder_bits == b:
        remainder -= cutoff

    # Combine quotient and remainder to get the original number
    number = quotient * m + remainder
    return number


# --- Shannon Entropy Function ---
def calculate_entropy(m):
    # For a uniform distribution, the entropy is H(X) = log2(m)
    return math.log2(m)

#arthimitic functions
def calculate_probabilities(data):
    total_symbols = len(data)
    symbol_counts = Counter(data)
    probabilities = {symbol: count / total_symbols for symbol, count in symbol_counts.items()}
    
    cumulative = {}
    cumulative_value = 0.0
    for symbol, prob in sorted(probabilities.items()):
        cumulative[symbol] = cumulative_value
        cumulative_value += prob

    return probabilities, cumulative

# Function to perform arithmetic encoding
def arithmetic_encode(sequence, probabilities, cum_probs):
    low = 0.0
    high = 1.0
    for symbol in sequence:
        range_width = high - low
        high = low + range_width * (cum_probs[symbol] + probabilities[symbol])
        low = low + range_width * cum_probs[symbol]
    encoded_value = (low + high) / 2
    return encoded_value

# Function to perform arithmetic decoding
def arithmetic_decode(encoded_value, probabilities, cum_probs, sequence_length):
    low = 0.0
    high = 1.0
    decoded_sequence = []

    for _ in range(sequence_length):
        range_width = high - low
        for symbol, prob in probabilities.items():
            symbol_low = low + range_width * cum_probs[symbol]
            symbol_high = symbol_low + range_width * prob
            if symbol_low <= encoded_value < symbol_high:
                decoded_sequence.append(symbol)
                low = symbol_low
                high = symbol_high
                break

    return ''.join(decoded_sequence)

#uniform functions

def plot_compression_gauge(percentage):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        title={'text': "Compression Ratio"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "skyblue"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 75}
        }
    ))
    return fig

def plot_mse_gauge(mse):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mse,
        title={'text': "Mean Squared Error (MSE)"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "orange"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "red"}
            ],
            'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': 25}
        }
    ))
    return fig

# Function to apply uniform scalar quantization
def uniform_quantizer(image, num_bits):
    img_array = np.array(image)
    min_val = np.min(img_array)
    max_val = np.max(img_array)

    # Calculate number of levels and step size
    levels = 2 ** num_bits
    step_size = (max_val - min_val) / levels

    # Quantize the image
    quantized_img = np.floor((img_array - min_val) / step_size) * step_size + min_val
    quantized_img = np.clip(quantized_img, min_val, max_val)

    return quantized_img

# Function to decompress the quantized image
def decompress_image(compressed_image, num_bits):
    # Assuming compression is quantization, we reverse the process by applying the inverse of quantization.
    img_array = np.array(compressed_image)
    min_val = np.min(img_array)
    max_val = np.max(img_array)

    # Calculate number of levels and step size
    levels = 2 ** num_bits
    step_size = (max_val - min_val) / levels

    # Decompress by rounding back to the closest quantized level
    decompressed_img = np.round((img_array - min_val) / step_size) * step_size + min_val
    decompressed_img = np.clip(decompressed_img, min_val, max_val)

    return decompressed_img

# Function to calculate compression ratio
def calculate_compression_ratio(original_image, num_bits):
    original_size = original_image.size[0] * original_image.size[1] * 3  # Assuming 3 channels (RGB)
    compressed_size = original_image.size[0] * original_image.size[1] * num_bits / 8 * 3
    compression_ratio = original_size / compressed_size
    return compression_ratio

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(original_image, compressed_image):
    original_array = np.array(original_image)
    compressed_array = np.array(compressed_image)
    mse = np.mean((original_array - compressed_array) ** 2)
    return mse

# Function to convert image to bytes for downloading
def pil_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

# LZW Compression Function
def lzw_compress(input_string):
    dictionary = {chr(i): i for i in range(128)}
    dict_size = 128
    current_string = ""
    compressed_data = []

    for char in input_string:
        combined_string = current_string + char
        if combined_string in dictionary:
            current_string = combined_string
        else:
            compressed_data.append(dictionary[current_string])
            dictionary[combined_string] = dict_size
            dict_size += 1
            current_string = char

    if current_string:
        compressed_data.append(dictionary[current_string])

    return compressed_data

# LZW Decompression Function
def lzw_decompress(compressed_data):
    dictionary = {i: chr(i) for i in range(128)}
    dict_size = 128
    current_code = compressed_data.pop(0)
    pre_string = dictionary[current_code]
    decompressed_data = [pre_string]

    for code in compressed_data:
        if code in dictionary:
            current = dictionary[code]
        elif code == dict_size:
            current = pre_string + pre_string[0]
        else:
            raise ValueError("Invalid compressed data.")
        
        decompressed_data.append(current)
        dictionary[dict_size] = pre_string + current[0]
        dict_size += 1
        pre_string = current

    return "".join(decompressed_data)



with tabs[0]:

    # Streamlit app layout
    st.title("Run-Length Encoding (RLE) App")
    st.write("Encode and decode strings using Run-Length Encoding (RLE).")

    # Input section for encoding
    st.header("Encode a String")
    uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded File Content:", file_content, height=150)
        input_text = file_content
    else:
        input_text = st.text_input("Or enter the text to encode:")
    is_binary_code=is_binary(input_text)
    if st.button("Encode"):
        if input_text:
            encoded_text = rle_encode(input_text)
            st.write("Encoded Text:", encoded_text)

            # Calculate bit sizes and compression percentage
            original_bits = calculate_bit_size(input_text,is_binary_code)
            compressed_bits = calculate_compressed_size(input_text,encoded_text,is_binary_code)[0]
            numerator, denominator,results = calculate_CR(input_text, encoded_text,is_binary_code)
            compression_pct = compression_percentage(numerator,denominator)
            
            st.write(f"encoded terms:{results}")
            st.write(f"Original Bit Size: {original_bits} bits")
            st.write(f"Compressed Bit Size: {compressed_bits} bits")
            st.write(f"compression ratio: {numerator}\{denominator}")
            st.write(f"in RLE algorith each {numerator} bits will replace by {denominator} bits")

            # Display compression gauge
            fig = plot_compression_gauge(compression_pct)
            st.plotly_chart(fig)
            st.download_button(
                label="Download Encoded File",
                data=encoded_text,
                file_name="encoded_text.txt",
                mime="text/plain"
            )
        
        else:
            st.warning("Please enter text to encode.")
            

    # Input section for decoding
    st.header("Decode a String")
    uploaded_decode_file = st.file_uploader("Upload a file containing encoded text:", type=["txt"], key="decode_file")
    if uploaded_decode_file is not None:
        decode_file_content = uploaded_decode_file.read().decode("utf-8").strip()
        st.text_area("Uploaded Encoded Content:", decode_file_content, height=100)
        encoded_input = decode_file_content
    else:
        encoded_tuples_input = st.text_input("Or enter the RLE-encoded text to decode:")
    if st.button("Decode_tuble"):
        if encoded_tuples_input:
            try:
                # Safely evaluate the input string as a Python literal
                encoded_data = ast.literal_eval(encoded_tuples_input)
                if not isinstance(encoded_data, list) or not all(isinstance(i, tuple) and len(i) == 2 for i in encoded_data):
                    raise ValueError("Input must be a list of tuples, each containing a character and its frequency.")

                decoded_text = decode_tuples(encoded_data)
                decoded_text
                st.download_button(
                label="Download Decode File",
                data=decoded_text,
                file_name="encoded_text.txt",
                mime="text/plain"
            )
                #st.write("Decoded Text:", decoded_text)
            except Exception as e:
                st.error(f"Error decoding: {e}")
        else:
            st.warning("Please enter tuples to decode.")

    if st.button("Decode_text"):
        decoded_text = decode_string(encoded_input)
        decoded_text
        st.download_button(
            label="Download Decode File",
            data=decoded_text,
            file_name="Decoded_text.txt",
            mime="text/plain"
        )

with tabs[1]:

    st.title("Huffman Encoding and Decoding")

    st.title("Upload Text File")

    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.subheader("Original Text")
        st.text(text)

        if st.button("encode"):
            encoded_text, huffman_dict, avg_length, entrop, efficiency, huffman_tree_root = encode_and_display_metrics(text)


    
        

            # Display Huffman Metrics
            st.subheader("Encoding Results")
            st.write(f"Average Length: {avg_length:.2f} bits")
            st.write(f"Entropy: {entrop:.2f} bits/character")
            st.write(f"Compression Efficiency: {efficiency:.2f}%")
            st.write(f"huffman dictionarry:{huffman_dict}")

            display_gauge(efficiency, "efficiency of Huffman Codes")

            st.subheader("Huffman Tree Visualization")
            fig = plot_tree(huffman_tree_root)  # Get the figure from plot_tree
            st.pyplot(fig) 


            # Display Encoded Text
            st.subheader("Encoded Text")
            st.text(encoded_text)
            st.download_button("Download Encoded File", data=encoded_text, file_name="encoded_text.txt", mime="text/plain")

            # Decoding Process
        huffman_dictt = st.text_input("Enter Huffman dictionary to decode:")
        if st.button("Decode") and huffman_dictt:
            try:
                # Convert the input string to a dictionary
                huffman_dict = ast.literal_eval(huffman_dictt)
                
                if isinstance(huffman_dict, dict):
                    decoded_text = decode_text(text, huffman_dict)
                    st.subheader("Decoded Text")
                    st.text(decoded_text)
                    st.download_button("Download Decoded File", data=decoded_text, file_name="decoded_text.txt", mime="text/plain")
                else:
                    st.error("The input is not a valid dictionary.")
            except Exception as e:
                st.error(f"Error: {e}")
        
with tabs[2]:

    st.title("Golomb code to Compression and Decompression the integers 🔢💾")

    try:
        # Get user input for action
        action = st.selectbox("Choose an action:", ['Encode', 'Decode'])
        # Get user input for divisor (m)
        m = st.number_input("Enter the divisor (m): ", min_value=1)
        if m <= 0:
            raise ValueError("Divisor (m) must be a positive integer.")

        # Initialize placeholders for inputs
        if action == 'Encode':
                input_numbers = st.text_area("Enter a set of numbers to encode (comma separated or single number): ", placeholder="e.g., 5, 12, 15 or just 5")

        elif action == 'Decode':
            code = st.text_input("Enter the binary code to decode: ").strip()

        # Compute button at the end after user inputs
        if st.button(f"Compute {action}ing results"):
            CR = None
            avg_length = None
            efficiency = None

            with st.spinner("Computing...."):
                time.sleep(1)
                
                # Validate inputs
                if action == 'Encode':
                    try:
                        # Check if the user input is a single number or a list
                        numbers = list(map(int, input_numbers.split(','))) if ',' in input_numbers else [int(input_numbers)]

                        # Ensure all numbers are non-negative
                        for num in numbers:
                            if num < 0:
                                st.write(f"Number {num} to encode must be a non-negative integer.")
                            

                        encoded_results = []
                        for n in numbers:
                            quotient, remainder, unary_code, binary_code, encoded_code = golomb_encode(n, m)
                            encoded_results.append({
                                "number": n,
                                "quotient": quotient,
                                "remainder": remainder,
                                "unary_code": unary_code,
                                "binary_code": binary_code,
                                "encoded_code": encoded_code
                            })
                            

                        # Display the encoded results for each number
                        for result in encoded_results:
                            st.write(f"Encoded result for {result['number']}: {result['encoded_code']}")
                            st.write(f"Quotient (q): {result['quotient']} 🔢")
                            st.write(f"Remainder (r): {result['remainder']} 🔢")
                            st.write(f"Unary code for q: {result['unary_code']} 🔲")
                            st.write(f"Binary code for r: {result['binary_code']} 💾")
                            st.write(f"Final code word: {result['encoded_code']} 💻")
                            

                        # Compression Ratio, Average Length, and Efficiency for each number
                        
                            n = result["number"]
                            CR = len(bin(n)[2:]) / len(result['encoded_code'])
                            st.write(f"Compression ratio for {n}: {CR:.2f} (bits/symbol)")
                            avg_length = len(result['encoded_code']) / 1  # As we're encoding one number
                            st.write(f"Average length for {n}: {avg_length:.2f} bits")
                            entropy = calculate_entropy(m)
                            st.write(f"Entropy for {n}: {entropy:.2f} bits/symbol")
                            efficiency = (entropy / len(result['encoded_code'])) * 100
                            st.write(f"Efficiency for {n}: {efficiency:.2f}%")
                            st.write("-------------------------------------------------------------------")

                    except ValueError:
                        st.write("Please enter valid integers separated by commas or a single number.")


                elif action == 'Decode':
                    if not code:
                        st.write("Please enter a binary string to decode.")
                    elif not all(c in '01' for c in code):
                        st.write("Encoded code must be a binary string composed of '0' and '1'.")
                    else:
                        decoded_number = golomb_decode(code, m)
                        st.write(f"Decoded result: {decoded_number}")

                    
                        

    except Exception as e:
        st.write(f"An unexpected error occurred: {e}")

with tabs[3]:
    st.title("Arithmetic Encoding Application")


# Step 1: Upload text file for compression
    st.header("Step 1: Upload Text File for Compression")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

    if uploaded_file:
        # Read the input text
        input_sequence = uploaded_file.read().decode('utf-8')
        st.write(f"**Input Text:** {input_sequence}")
        st.write(f"**Original Length:** {len(input_sequence)}")

        # Calculate probabilities and cumulative probabilities
        probabilities, cum_probs = calculate_probabilities(input_sequence)

        # Encode the sequence
        encoded_value = arithmetic_encode(input_sequence, probabilities, cum_probs)
        st.write(f"**Encoded Value:** {encoded_value}")

        # Save the encoded value to a file for download
        compressed_file_name = "compressed.txt"
        with open(compressed_file_name, "w", encoding="utf-8") as f:
            f.write(str(encoded_value))
        st.download_button(
            label="Download Compressed File",
            data=open(compressed_file_name, "r").read(),
            file_name=compressed_file_name,
            mime="text/plain"
        )

    # Step 2: Decode the compressed file
    st.header("Step 2: Decode the Compressed File")
    uploaded_compressed_file = st.file_uploader("Upload the compressed file", type=["txt"], key="decode")

    if uploaded_compressed_file:
        # Read the compressed value
        compressed_value = float(uploaded_compressed_file.read().decode('utf-8'))
        st.write(f"**Uploaded Encoded Value:** {compressed_value}")

        # Input sequence length
        sequence_length = st.number_input("Enter the sequence length for decoding:", min_value=1, step=1)

        if sequence_length and uploaded_file:
            # Decode the sequence
            decoded_sequence = arithmetic_decode(compressed_value, probabilities, cum_probs, sequence_length)
            st.write(f"**Decoded Sequence:** {decoded_sequence}")

            # Validation
            if input_sequence == decoded_sequence:
                st.success("The decoding was successful. The input sequence matches the decoded sequence.")
            else:
                st.error("The decoding failed. The input sequence does not match the decoded sequence.")

with tabs[4]:
    st.title("LZW Algorithm")
    
    uploaded_file = st.file_uploader("Upload a file to compress", type=["txt"])

    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        st.subheader("Original Data:")
        st.text(file_content)

        # Compress the file content
        compressed_data = lzw_compress(file_content)

        st.subheader("Compressed Data:")
        st.text(compressed_data)  # Display compressed data as a list

        st.subheader("table of numbers")
        st.write(compressed_data)

        # Save compressed data to a text file
        compressed_file_name = "compressed.txt"
        with open(compressed_file_name, "w", encoding="utf-8") as f:
            f.write(" ".join(map(str, compressed_data)))

        # Provide download button for the compressed file
        st.download_button(
            label="Download Compressed File",
            data=open(compressed_file_name, "r").read(),
            file_name=compressed_file_name,
            mime="text/plain",
        )

        # Calculate sizes for evaluation
        original_size_bits = len(file_content) * 8  # Original size in bits
        max_compressed_value = max(compressed_data)
        max_binary_length = len(bin(max_compressed_value)[2:]) 
        compressed_size_bits = len(compressed_data) * max_binary_length  # Compressed size in bits


        st.write(f"Original Size: {original_size_bits} bytes")
        st.write(f"len of max num bin code:{max_binary_length}")
        st.write(f"Compressed Size: {compressed_size_bits} bytes")
        st.write(f"Compression Ratio: {original_size_bits} / {compressed_size_bits}")
        cr=original_size_bits/compressed_size_bits
        rounded_compression_ratio = round(cr,2)
        
        fig=plot_compression_gauge(rounded_compression_ratio )
        st.plotly_chart(fig)

    # Decompression Tab
    st.header("LZW Decompression")
    uploaded_compressed_file = st.file_uploader("Upload a compressed file to decompress", type=["txt"])

    if uploaded_compressed_file is not None:
        # Read the compressed data from the file
        compressed_file_content = uploaded_compressed_file.read().decode("utf-8")
        compressed_data = list(map(int, compressed_file_content.split()))

        # Decompress the data
        decompressed_data = lzw_decompress(compressed_data)

        # Display decompressed content
        st.subheader("Decompressed Data:")
        st.text(decompressed_data)

        # Save decompressed data to a text file
        decompressed_file_path = "decompressed.txt"
        with open(decompressed_file_path, "w", encoding="utf-8") as f:
            f.write(decompressed_data)

        # Provide download button for the decompressed file
        st.download_button(
            label="Download Decompressed File",
            data=open(decompressed_file_path, "r").read(),
            file_name="decompressed.txt",
            mime="text/plain",
        )

with tabs[5]:
    st.title("Image Compression and Decompression using Uniform Scalar Quantization 📸")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

        # Input the number of bits for quantization
        num_bits = st.slider("Select the number of bits for quantization", min_value=1, max_value=8, value=4)

        # Button to apply quantization and display the result
        if st.button("Apply Uniform Scalar Quantization"):
            # Apply uniform quantization
            quantized_image = uniform_quantizer(image, num_bits)

            # Convert quantized image back to PIL Image for display
            quantized_image_pil = Image.fromarray(np.uint8(quantized_image))

            # Display the quantized image
            st.image(quantized_image_pil, caption=f"Compressed Image with {num_bits} bits", use_container_width=True)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            axes[1].imshow(quantized_image_pil)
            axes[1].set_title(f"Quantized Image ({num_bits} bits)")
            axes[1].axis("off")

            st.pyplot(fig)

            # Calculate and display compression ratio
            compression_ratio = calculate_compression_ratio(image, num_bits)
            st.write(f"Compression Ratio: {compression_ratio:.2f}")
            fig = plot_compression_gauge(compression_ratio)
            st.plotly_chart(fig,use_container_width=True)

            # Calculate and display Mean Squared Error (MSE)
            mse = calculate_mse(image, quantized_image_pil)
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            fig = plot_mse_gauge(mse)
            st.plotly_chart(fig,use_container_width=True)


            # Convert quantized image to bytes for download
            quantized_img_bytes = pil_to_bytes(quantized_image_pil)

            
            
            # Provide a download button for the compressed image
            st.download_button(
                label="Download Compressed Image",
                data=quantized_img_bytes,
                file_name="compressed_image.png",
                mime="image/png"
            )

        # Upload compressed image for decompression

        st.title("Decompression")
        uploaded_compressed_file = st.file_uploader("Upload a compressed image", type=["png"])

        if uploaded_compressed_file is not None:
            compressed_image = Image.open(uploaded_compressed_file)
            st.image(compressed_image, caption="Compressed Image", use_container_width=True)

            # Button to decompress the uploaded image
            if st.button("Decompress Image"):
                decompressed_image = decompress_image(compressed_image, num_bits)

                # Convert decompressed image back to PIL
                decompressed_image_pil = Image.fromarray(np.uint8(decompressed_image))

                # Display decompressed image
                st.image(decompressed_image_pil, caption="Decompressed Image", use_container_width=True)

                # Calculate and display MSE for decompressed image
                mse_decompression = calculate_mse(image, decompressed_image_pil)
                st.write(f"Mean Squared Error (MSE) after Decompression: {mse_decompression:.2f}")

