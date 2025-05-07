import os
import os.path as osp
import numpy as np

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

# === PATHS ===
out_labels = '/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/datasets/motsynth.train'
seq_root = '/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/datasets/mot/MOTSYNTH'
frame_root = osp.join(seq_root, 'images')
ann_root = osp.join(seq_root, 'mot_annotations')
label_root= '/leonardo_work/IscrB_FeeCO/fmorandi/datasets/motsynth/labels_with_ids'

mkdirs(label_root)

# === LISTA SEQUENZE ===
seqs = sorted(os.listdir(frame_root))

tid_curr = 0
tid_last = -1

for seq in seqs:
    print(f"Processing {seq}...")
    
    seq_path = osp.join(frame_root, seq, 'rgb')
    ann_path = osp.join(ann_root, seq)

    # === Carica dimensioni immagine da seqinfo.ini ===
    seqinfo_path = osp.join(ann_path, 'seqinfo.ini')
    with open(seqinfo_path, 'r') as f:
        lines = f.readlines()
    info = {}
    for line in lines:
        if '=' in line:
            key, value = line.strip().split('=')
            info[key] = value
    seq_width = int(info['imWidth'])
    seq_height = int(info['imHeight'])

    # === Carica ground truth ===
    gt_txt = osp.join(ann_path, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    # === Crea directory di destinazione ===
    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

        # === Scrivi etichette per ogni frame ===
    tid_mapping = {}             #qui mi serve un'altra mappatura perche le righe sono ordinate per frame e non per stesso track id come in mot17
    next_tid = 0

    for fid, tid, bb_left, bb_top, bb_width, bb_height, conf, cls_id, vis, x3d, y3d, z3d in gt:
        if conf == 0 or cls_id != 1:
            continue  # Salta oggetti non validi o non pedoni
        fid = int(fid)
        tid = int(tid)
        if tid not in tid_mapping:
            tid_mapping[tid] = next_tid
            next_tid += 1
        new_tid = tid_mapping[tid]

        # Converti bbox da top-left a center
        x = bb_left + bb_width / 2
        y = bb_top + bb_height / 2
        
        label_fpath = osp.join(seq_label_root, '{:04d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            new_tid, x / seq_width, y / seq_height, bb_width / seq_width, bb_height / seq_height
        )
        with open(label_fpath, 'a') as f:
            f.write(label_str)
                #ogni frame è un txt con gli oggetti che contiene. gli stessi oggetti hann stesso track_id se compaiono in piu frames

# === Genera lista immagini corrispondenti (opzionale) ===
tmp_list = 'motsynth.train.tmp'
os.system(f'find {label_root} -type f > {tmp_list}')

with open(tmp_list, 'r') as f:
    s = f.read().replace('.txt', '.jpg').replace('labels', 'motsynth/images')

with open(out_labels, 'w') as f:
    f.write(s)

os.remove(tmp_list)
