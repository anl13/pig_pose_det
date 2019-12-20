import numpy as np 
import json 

univ_cat_info = {
    'keypoints': [
        "nose", "eye_left", "eye_right", "ear_root_left", "ear_root_right", 
        "shoulder_left", "shoulder_right", "elbow_left", "elbow_right", "paw_left", "paw_right", 
        "hip_left", "hip_right", "knee_left", "knee_right", "foot_left", "foot_right", 
        "neck", "tail_root", "withers", "center", 
        "tail_middle", "tail_end"
    ], # totally 23 joints for now [20191220]
    'name': "animal", 
    'id': 1, 
    'skeleton': [
        [0,1], [0,2], [1,2], [1,3], [2,4],
        [0,17], [17,5],[17,6], [5,7], [7,9], [6,8], [8,10],
        [17,19], [19,20], [20,18], [18,21], [21,22],
        [18,11], [18,12], [11,13], [13,15], [12,14], [14,16]
    ] # 23 limbs in total 
}

def read_old_json(filename):
    with open(filename, 'r') as f: 
        data = json.load(f) 
    return data 

def map_keypoints(keypoints, mapping):
    new_keypoints = [0] * 69
    for i in range(23):
        mapped_id = mapping[i]
        if mapped_id >= 0: 
            new_keypoints[i*3:i*3+3] = keypoints[mapped_id*3:mapped_id*3+3]
    return new_keypoints

def convert_atrw(infile, outfile):
    data = read_old_json(infile) 
    mapping_atrw = [2, -1, -1, 0, 1, 5, 3, -1, -1, 6, 4, 10, 7, 11, 8, 12, 9, -1, 13, -1, 14, -1, -1]
    new_cat = univ_cat_info.copy() 
    new_cat['name'] = "tiger"
    new_anns = [] 
    for ann in data['annotations']:
        kpts = ann['keypoints'].copy()
        ann['keypoints'] = map_keypoints(kpts, mapping_atrw) 
        new_anns.append(ann) 
    data['categories'] = new_cat 
    data['annotations'] = new_anns 
    with open(outfile, 'w') as f: 
        json.dump(data, f)   

def convert_pig(infile, outfile):
    data = read_old_json(infile) 
    mapping_pig20 = [0, 1, 10, 2, 11, 3,12,4,13, 5, 14, 6,15,7,16,8,17,-1, 9, -1, 18,-1,-1]
    new_cat = univ_cat_info.copy() 
    new_cat['name'] = "pig"
    new_anns = [] 
    for ann in data['annotations']:
        kpts = ann['keypoints'].copy()
        ann['keypoints'] = map_keypoints(kpts, mapping_pig20) 
        new_anns.append(ann) 
    data['categories'] = new_cat 
    data['annotations'] = new_anns 
    with open(outfile, 'w') as f: 
        json.dump(data, f)   
    from IPython import embed; embed() 

if __name__ == "__main__":
    infile = "data/pig20/annotations/train_pig_cocostyle.json"
    outfile = "data/pig_univ/annotations/train_pig_cocostyle.json" 
    convert_pig(infile, outfile) 

