# Copyright 2021 - Valeo Comfort and Driving Assistance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import pathlib
import argparse
import detectron2.data
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepares the CAD gt for WiderPerson"
                    "dataset in the data format expected from detectron2.")
    parser.add_argument("--wider_person_dir", type=str, default='../datasets/WiderPerson',
                        help="Path to where the WiderPerson dataset is.")
    parser.add_argument("--file_wider_person", type=str, default='../datasets/wider_person_filenames.txt',
                        help="Location of WiderPerson subset.")
    args = parser.parse_args()

    print('Prepare Class-Agnostic WiderPerson in the data format expected from detectron2.')

    # Load WiderPerson images
    wider_person_f = '../datasets/wider_person_filenames.txt'
    with open(args.file_wider_person, "r") as f:
        sel_20k = f.readlines()
        sel_20k = [s.replace("\n", "") for s in sel_20k]
    im20k = [str(int(s.split("_")[-1].split(".")[0])) for s in sel_20k]

    # Load annotations
    annotation_file = pathlib.Path(args.wider_person_dir) / "annotations" / "instances_train2014.json"
    with open(annotation_file) as json_file:
        annot = json.load(json_file)

    wider_person_data_gt_train14 = detectron2.data.DatasetCatalog.get("wider_person_2014_train")
    ann_to_img_ids = [x['id'] for ind, x in enumerate(annot['images'])]
    map_id_to_annot = [x['image_id'] for x in wider_person_data_gt_train14]

    data_gt_20k = []
    for file_name in tqdm(sel_20k):

        image_name = file_name[:-len('.jpg')]
        image_id = image_name.split('_')[-1].split('.')[0]
        image_id_int = int(image_id)
        
        full_img_path = pathlib.Path(args.wider_person_dir) / "images" / file_name
        ann_id = ann_to_img_ids.index(image_id_int)
        assert full_img_path.is_file()
        annotations = wider_person_data_gt_train14[map_id_to_annot.index(image_id_int)]["annotations"]
        ca_annotations = [{'iscrowd':v['iscrowd'], 'bbox':v['bbox'], 'category_id': 0, 'bbox_mode':v['bbox_mode']} for v in annotations]

        data_gt_20k.append({
            "file_name": str(full_img_path),
            "image_id": image_id,
            "height": annot['images'][ann_id]['height'],
            "width": annot['images'][ann_id]['width'],
            "annotations": ca_annotations,
        })

    print("Dataset WiderPerson CAD-gt has been saved.")

    json_data = {"dataset": data_gt_20k,}
    with open(f'./datasets/wider_person_trainval_CAD_gt.json', 'w') as outfile:
        json.dump(json_data, outfile)
