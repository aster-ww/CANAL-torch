# RapidVideOCR
[简体中文](./README.md) | English

<p align="left">
    <a href="https://colab.research.google.com/github/SWHL/RapidVideOCR/blob/main/RapidVideOCR.ipynb" target="_blank"><img src="./assets/colab-badge.svg" alt="Open in Colab"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/LICENSE-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python-3.6+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
</p>

- Drive from [videocr](https://github.com/apm1467/videocr)
- Extract subtitles embedded in the video faster and more accurately, and provide three formats of `txt|SRT|docx`
  - **Faster**:
    - Adapted the [Decord](https://github.com/dmlc/decord), which is dedicated to processing videos.
    - Only extract the key frames of the whole video.
  - **More accurately**:
    - The entire project is completely offline CPU running.
    - The OCR part is from [RapidOCR](https://github.com/RapidAI/RapidOCR), relying on the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/).

#### TODO
- [ ] Add specific parameter descrition.
- [ ] Make the project logo.
- [x] Add sample vidoe to run time-consuming benchmark.

#### The benchmark of costing time
|Env|Test MP4| Cost(s)|
|:---:|:---:|:---:|
|`Intel(R) Core(TM) i7-6700 CPU @3.40GHz 3.41 GHz`|`assets/test_video/2.mp4`|4.681s|


#### Use
1. Download the OCR models and dictionary keys used by RapidOCR. ([Baidu:drf1](https://pan.baidu.com/s/103kx0ABtU7Lif57cv397oQ) | [Google Drive](https://drive.google.com/drive/folders/1cjfawIhIP0Yq7_HjX4wtr_obcz7VTFtg?usp=sharing))

2. Put the downloaded models and `ppocr_keys_v1.txt` under the `resources`, the specific directories are as follows:
   ```text
   resources
      - models
        - ch_mobile_v2.0_rec_infer.onnx
        - ch_PP-OCRv2_det_infer.onnx
        - ch_ppocr_mobile_v2.0_cls_infer.onnx
      - ppocr_keys_v1.txt
   ```

4. Install the run envirement.
   - Recommend the Window OS, because the entire project has only been tested under Windows now.
   - Install the relative packages as follows.
      ```bash
      cd RapidVideOCR

      pip install -r requirements.txt -i https://pypi.douban.com/simple/
      ```

5. Run
   - The code:
      ```bash
      cd RapidVideOCR

      python main.py
      ```
    - The output log is as follows：
        ```text
        Loading assets/test_video/2.mp4
        Get the key point: 100%|██████| 71/71 [00:03<00:00, 23.46it/s]
        Extract content: 100%|██████| 4/4 [00:03<00:00,  1.32it/s]
        The srt has been saved in the assets\test_video\2.srt.
        The txt has been saved in the assets\test_video\2.txt.
        The docx has been saved in the assets\test_video\2.docx.
        ```
   - Also run on the [Google Colab](https://colab.research.google.com/github/SWHL/RapidVideOCR/blob/main/RapidVideOCR.ipynb).

6. Look the output files where the video is located.