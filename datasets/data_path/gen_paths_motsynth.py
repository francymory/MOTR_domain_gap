import os
import os.path as osp
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motsynth-path', default='/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/datasets/mot/MOTSYNTH', help="Path to MOTSynth dataset root")
    parser.add_argument('--seqmaps-dir', default='seqmaps', help="Directory where split txt files are stored")
    parser.add_argument('--save-dir', default='/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/datasets/data_path', help="Directory where the new train.txt and half.txt will be saved")
    parser.add_argument('--train-split', default='train', help="Name of the txt file (without .txt) that contains train sequence IDs")
    parser.add_argument('--val-split', default='val', help="Name of the txt file (without .txt) that contains test sequence IDs (used for half.txt)")
    parser.add_argument('--subsample', default=10, type=int, help="Frame subsampling rate (e.g., 10 means 1 every 10 frames)")
    return parser.parse_args()

def read_split_file(path):
    with open(path, 'r') as f:
        return [line.strip().zfill(3) for line in f if line.strip()]

def get_image_paths(motsynth_path, seq, subsample):
    image_dir = osp.join(motsynth_path, 'images', seq, 'rgb')
    if not osp.exists(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    
    all_files = sorted(f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png')))
    return [osp.join('MOTSYNTH','images',seq,'rgb', f) for i, f in enumerate(all_files) if i % subsample == 0]

def generate_file_list(motsynth_path, seqs, subsample):
    all_paths = []
    for seq in tqdm(seqs, desc="Processing sequences"):
        paths = get_image_paths(motsynth_path, seq, subsample)
        all_paths.extend(paths)
    return all_paths

def save_paths(save_dir, filename, paths):
    os.makedirs(save_dir, exist_ok=True)
    with open(osp.join(save_dir, filename), 'w') as f:
        f.writelines(f"{p}\n" for p in paths)

def main():
    args = parse_args()

    train_seqs = read_split_file(osp.join(args.motsynth_path, args.seqmaps_dir, f"{args.train_split}.txt"))
    test_seqs = read_split_file(osp.join(args.motsynth_path, args.seqmaps_dir, f"{args.val_split}.txt"))

    train_paths = generate_file_list(args.motsynth_path, train_seqs, args.subsample)
    test_paths = generate_file_list(args.motsynth_path, test_seqs, args.subsample)

    save_paths(osp.join(args.motsynth_path, args.save_dir), 'motsynth_10.train', train_paths)
    save_paths(osp.join(args.motsynth_path, args.save_dir), 'motsynth_10.val', test_paths)

if __name__ == '__main__':
    main()
