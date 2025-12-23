To setup the environment and create the script, there are three steps (For computer with local GPU resourses):
First, ensure that your python version in conda environment is python3.11.x, and download the python packages listed in the "requirements.txt" in your conda environment.
Second, open the "audio_transcriber.py" file and modify the code to your path of  mp4 video file.( If the path is "test_materials/input1.mp4",then you should modify (input_file = "test_materials/input1.mp4")) And you can also change the path and the name of output file in the code, such as (output_file = "output1.srt")
Third, run the "audio_transcriber.py" file and wait for the file "output.srt" to be created.

For computer without local GPU resourses, there are steps:
First, open Colab and create a new ipynb file and change the state to "T4 GPU".
Second, download the packages in the "requirements.txt".You can use the following command to download on the Colab:
!pip install -q openai-whisper matplotlib ipython
!pip install openai==1.98.0
!pip install safetensors==0.5.3
!pip install transformers==4.38.2 --no-cache-dir
!pip install facenet_pytorch
!pip install 'opencv-python==4.9.0.80'
!pip install 'pandas==2.2.1'
Third, copy the code of "audio_transcriber.py" to the ipynb platform.
Fourth, upload the mp4 file onto the Colab.(The "input.mp4" file is stored in the folder test_materials)
Fifth, modify the code to your path of  mp4 video file.
(For example, input_file = "input1.mp4").And you can also change the path and the name of output file in the code, such as (output_file = "output1.srt")
Sixth, run the "ipynb" file and wait for the file "output.srt" to be created.

 