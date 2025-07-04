# FACE-INTELLIGENCE-TASK[COMSYS]


##  How to Run Task A Notebook

1. Open `TaskA_Gender_Classification.ipynb` in Google Colab.
2. Upload the following datasets when prompted:
   - `train.zip` containing: `train/male/`, `train/female/`
   - `val.zip` containing: `val/male/`, `val/female/`
3. Run all cells from top to bottom.
4. The model will be trained and metrics will be printed and saved.

Model and results will be saved as:
- `resnet18_gender.pth`
- `results_task_a.json`

##  How to Run Task B Notebook

1. Upload `Task_B.zip` to your Google Drive (e.g., inside `MyDrive`).
2. Open `TaskB_Face_Recognition.ipynb` in Google Colab.
3. Follow the notebook prompt to mount your Google Drive.
4. The notebook will automatically unzip `Task_B.zip` and extract the dataset.

The following structure will be created:
/content/Task_B/
├── train/
│ ├── person_name/
│ │ ├── frontal.jpg
│ │ └── distortion/
│ │ ├── blurry.jpg, foggy.jpg, ...
│ └── ...
└── val/
└── ...
5. Run all notebook cells. The Siamese model will be trained and evaluated.
6. Model and results will be saved as:
   - `siamese_facecon_resnet18.pt`
   - `results_task_b.json`
   - 'results_task_b_train.json'
