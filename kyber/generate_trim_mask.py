import numpy as np

############################
index_threshold = 0.995
message_threshold = 0.99
############################

def generate_index_trim_mask():
    # Calculate what indexes to keep
    index_sample_importance = np.load("sample_importance/index_sample_importance.npy")
    index_sample_importance = np.min(index_sample_importance, axis=0)
    index_keep = np.where(index_sample_importance<=index_threshold)[0]
    
    # Print what to keep
    print("Index samples to keep")
    print("Length =", len(index_keep))
    print(index_keep)

    # Save mask
    np.save("index_trim_mask.npy", index_keep)

def generate_message_trim_mask(bit):
    # Calculate what indexes to keep
    message_sample_importance = np.load(f"sample_importance/message_bit{bit}_sample_importance.npy")
    message_sample_importance = np.min(message_sample_importance, axis=0)
    message_keep = np.where(message_sample_importance<=message_threshold)[0]
    
    # Adjust for full synchronized traces
    message_keep = message_keep + (145+225)

    # Print what to keep
    print("Bit", bit)
    print("Message samples to keep")
    print("Length =", len(message_keep))
    print(message_keep)

    # Save mask
    np.save(f"message_bit{bit}_trim_mask.npy", message_keep)

def main():
    print("Generating trim masks with")
    print("    index threshold:", index_threshold)
    print("  message threshold:", message_threshold)
    print()
    generate_index_trim_mask()
    print()
    generate_message_trim_mask(bit=0)
    print()
    generate_message_trim_mask(bit=7)

main()
