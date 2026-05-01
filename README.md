## Demo
### Input 1
[input1.mp4](input1.mp4)
The generated subtitle files from demo inference are provided below:

- [output1.srt](test_materials/output1.srt)
### Input 2
[input2.mp4](input2.mp4)
The generated subtitle files from demo inference are provided below:

- [output2.srt](test_materials/output2.srt)
## Environment Requirements
- Conda installed
- Python: **3.11.x**

## Getting Started

### Local Run (With GPU)
1. Create and activate the Conda environment
```bash
conda create -n transcriber python=3.11
conda activate transcriber
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Configure file paths
Open audio_transcriber.py and modify the lines below:

input_file = "test_materials/input1.mp4"   # your MP4 video path

output_file = "output1.srt"                # custom SRT output name

5. Run the script
```bash
python audio_transcriber.py
```
An .srt subtitle file will be generated automatically.
