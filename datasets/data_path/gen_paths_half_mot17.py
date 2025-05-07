
import os
import os.path as osp

def get_seq_length(seqinfo_path):
    with open(seqinfo_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("seqLength="):
            return int(line.strip().split("=")[1])
    raise ValueError(f"seqLength not found in {seqinfo_path}")

def generate_split_files(root_dir, output_dir):
    train_lines = []
    val_lines = []

    train_root = osp.join(root_dir, "MOT17", "images", "train")
    sequences = sorted(os.listdir(train_root))

    for seq in sequences:
        print(seq)
        seq_path = osp.join(train_root, seq)
        seqinfo_path = osp.join(seq_path, "seqinfo.ini")
        if not osp.exists(seqinfo_path):
            print(f"Skipping {seq}: no seqinfo.ini found")
            continue
        
        seq_len = get_seq_length(seqinfo_path)
        midpoint = seq_len // 2 + 1  # Include frame 000001 to midpoint in train
        img_dir = osp.join(seq_path, "img1")

        for i in range(1, seq_len + 1):
            frame_name = f"{i:06d}.jpg"
            line = osp.join("MOT17", "images", "train", seq, "img1", frame_name)
            if i <= midpoint:
                train_lines.append(line)
            else:
                val_lines.append(line)

    # Write to files
    with open(osp.join(output_dir, "mot17_half.train"), "w") as f:
        f.write("\n".join(train_lines))
    with open(osp.join(output_dir, "mot17_half.val"), "w") as f:
        f.write("\n".join(val_lines))

    print(f"Done. Wrote {len(train_lines)} train and {len(val_lines)} val entries.")

# Example usage
if __name__ == "__main__":
    generate_split_files(root_dir="/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/datasets/mot",output_dir="/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/datasets/data_path")  
