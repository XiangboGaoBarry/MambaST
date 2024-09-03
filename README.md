


# MambaST
MambaST: A Plug-and-Play Cross-Spectral Spatial-Temporal Fuser for Efficient Pedestrian Detection
## Usage

1. **Build Docker Image**
   ```bash
   cd docker
   bash build.sh
   ```
   *Note: Make sure to change `USERNAME` and `USER_UID` in the script.*

2. **Run Docker Container**
   ```bash
   cd ..
   bash docker/run.sh
   ```
   *Note: Update `your_username` and `path_to_project_directory` accordingly.*

3. **Install Third Party Libraries**
   ```bash
   cd thirdparty/Vim
   pip install -e mamba-1p1p1
   cd ../..
   pip install causal-conv1d==1.1.0
   ```

4. **Download and Prepare Dataset**
   - Download the KAIST-CVPR15 dataset from [here](https://soonminhwang.github.io/rgbt-ped-detection/) and place it in your dataset directory.
   - Copy the dataset:
     ```bash
     cp -r ~/dataset/* path_to_your_dataset_directory
     ```
   - Copy the sanitized annotations format:
     ```bash
     cp sanitized_annotations_format_all path_to_your_dataset_directory
     ```

5. **Update Annotation Path**
   - Modify `KAIST_ANNOTATION_PATH` in `utils/datasets_vid.py` to the absolute path of `sanitized_annotations_format_all`.

6. **Training**
   ```bash
   bash train.sh
   ```
   *Sample training code.*

7. **Testing**
   ```bash
   bash test.sh
   ```

 ## Citation 
If you found repo useful, feel free to cite.
```
@article{gao2024mambast,
  title={MambaST: A Plug-and-Play Cross-Spectral Spatial-Temporal Fuser for Efficient Pedestrian Detection},
  author={Gao, Xiangbo and Kanu-Asiegbu, Asiegbu Miracle and Du, Xiaoxiao},
  journal={arXiv preprint arXiv:2408.01037},
  year={2024}
}
```
