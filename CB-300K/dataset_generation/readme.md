# CB-MRG dataset generation
1. Collect Visual Genome information by running the following command ([image_info.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/CB-300K/dataset_generation/MRG/image_info/image_info.py)):
```
python image_info.py
```
- Remember to replace path in our provided file with the specific path on your machine.

2. Clean Visual Genome data by running the following command ([data_clean.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/CB-300K/dataset_generation/MRG/data_generation.py)):
```
python data_clean.py
```
3. Generate conversation with gpt-4 by running the following command ([recurrent_data_generation.sh](https://github.com/sunsmarterjie/ChatterBox/blob/main/CB-300K/dataset_generation/MRG/recurrent_data_generation.sh)):
```
bash recurrent_data_generation.sh
```
-  It's noted that openai package should be installed and user's OPENAI_KEY and OPENAI_API should be provided in [chat.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/CB-300K/dataset_generation/MRG/chat.py).  
