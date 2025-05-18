import os
import os.path as osp

def get_seq_length(seqinfo_path):
    with open(seqinfo_path, "r") as f:
        for line in f:
            if line.startswith("seqLength="):
                return int(line.strip().split("=")[1])
    raise ValueError(f"seqLength not found in {seqinfo_path}")

def split_gt_file(gt_path, seq_len):
    midpoint = seq_len // 2 + 1

    gt_half_train = []
    gt_half_val = []

    with open(gt_path, "r") as f:
        for line in f:
            frame_id = int(line.split(',')[0])
            if frame_id <= midpoint:
                gt_half_train.append(line)
            else:
                gt_half_val.append(line)
    
    return gt_half_train, gt_half_val

def generate_half_gt_files(root_dir):
    train_root = osp.join(root_dir, "MOT17", "images", "train")
    sequences = sorted(os.listdir(train_root))

    for seq in sequences:
        print(f"Processing sequence: {seq}")
        seq_path = osp.join(train_root, seq)
        seqinfo_path = osp.join(seq_path, "seqinfo.ini")
        gt_path = osp.join(seq_path, "gt", "gt.txt")

        if not osp.exists(seqinfo_path) or not osp.exists(gt_path):
            print(f"Skipping {seq}: missing seqinfo.ini or gt.txt")
            continue

        seq_len = get_seq_length(seqinfo_path)
        gt_half_train, gt_half_val = split_gt_file(gt_path, seq_len)

        # Save the new GT files
        gt_dir = osp.join(seq_path, "gt")
        train_out = osp.join(gt_dir, "gt_half_train.txt")
        val_out = osp.join(gt_dir, "gt_half_val.txt")

        with open(train_out, "w") as f:
            f.writelines(gt_half_train)
        with open(val_out, "w") as f:
            f.writelines(gt_half_val)

        print(f"  → Saved {len(gt_half_train)} train lines and {len(gt_half_val)} val lines.")

    print("Done generating half GT files.")

# Example usage
if __name__ == "__main__":
    generate_half_gt_files(
        root_dir="/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/datasets/mot"
    )
