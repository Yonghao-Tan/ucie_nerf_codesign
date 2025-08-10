import os
import ast
import sys

def read_psnr_files(pth):
    scenes = ["trex", "fern", "flower", "leaves", "room", "fortress", "horns", "orchids"]
    base_path = "./llff_test/eval_llff/"
    psnr_values = []

    for scene in scenes:
        file_path = os.path.join(base_path, f"psnr_{scene}_{pth}.txt")
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                try:
                    data_dict = ast.literal_eval(content)
                    scene_dict = data_dict.get(scene)
                    if scene_dict:
                        fine_mean_psnr = scene_dict.get('fine_mean_psnr')
                        if fine_mean_psnr is not None:
                            psnr_values.append(fine_mean_psnr)
                except:
                    print(f"Error reading or parsing file: {file_path}")

    if psnr_values:
        average_psnr = sum(psnr_values) / len(psnr_values)
        return average_psnr
    else:
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_llff_all_statistics.py <pth>")
        sys.exit(1)

    pth = sys.argv[1]
    average_psnr = read_psnr_files(pth)
    if average_psnr is not None:
        print(f"Average Fine Mean PSNR: {average_psnr}")
    else:
        print("No valid PSNR data found.")

if __name__ == "__main__":
    main()
