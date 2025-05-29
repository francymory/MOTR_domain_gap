import os

# === CONFIGURAZIONE ===
base_path = "/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/datasets/mot"  # radice del dataset
input_file = "/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/datasets/data_path/motsynth_50_train.train"         # file con i percorsi originali (.jpg)
output_file = "/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/datasets/data_path/motsynth_50_train_correct.train"  # file di output

# === FUNZIONE PRINCIPALE ===
def filter_valid_image_paths(input_file, output_file, base_path):
    with open(input_file, "r") as f:
        img_paths = [line.strip() for line in f if line.strip()]

    valid_paths = []

    for img_rel_path in img_paths:
        img_full_path = os.path.join(base_path, img_rel_path)
        label_rel_path = img_rel_path.replace("images", "labels_with_ids").replace("rgb", "img1").replace(".jpg", ".txt")
        label_full_path = os.path.join(base_path, label_rel_path)

        if os.path.isfile(img_full_path) and os.path.isfile(label_full_path):
            valid_paths.append(img_rel_path)

    print(f"Totale immagini valide trovate: {len(valid_paths)}")

    with open(output_file, "w") as f_out:
        for path in valid_paths:
            f_out.write(path + "\n")

# === ESECUZIONE ===
if __name__ == "__main__":
    filter_valid_image_paths(input_file, output_file, base_path)
